# Copyright 2024-2026 IBM Corporation

"""
Overlap verification for accelerator compute events.

Compute events of the same stream cannot overlap in time: the stream processes them one after
another. Any detected overlap therefore indicates a data problem (e.g. inaccurate timestamps)
and is reported as an error-level finding of the verification report.

This module reuses the overlap detection of the regular processing pipeline
(OverlapDetectionContext) and only replaces the parts that differ in verification mode:
  * overlaps are reported, never resolved
  * the finding is error-level and records each offending event as an instance
  * the streams are identified per input dialect instead of by pid+tid
"""

import copy

from aiu_trace_analyzer.types import TraceEvent, TraceWarning
from aiu_trace_analyzer.pipeline.context import AbstractContext, AbstractVerificationContext
from aiu_trace_analyzer.pipeline.barrier import TwoPhaseWithBarrierContext
from aiu_trace_analyzer.pipeline.overlap import OverlapDetectionContext
from aiu_trace_analyzer.pipeline.tools import PipelineContextTool


def _is_memory_event(event: TraceEvent) -> bool:
    """Return whether an accelerator event represents a memory operation."""
    dialect = PipelineContextTool.get_dialect_of_event(event)
    if dialect is None:
        return False

    if dialect.get("NAME") == "TORCH":
        return event.get("cat") in {"gpu_memcpy", "gpu_memset"}

    return (
        PipelineContextTool.is_category(event, "acc_datatransfer_HtoD")
        or PipelineContextTool.is_category(event, "acc_datatransfer_DtoH")
    )



def _is_memory_superevent(event: TraceEvent) -> bool:
    return event.get("args", {}).get("_memory_superevent") is True


class MemoryOverlapContext(TwoPhaseWithBarrierContext):
    """Collect occupied memory intervals for each accelerator stream."""

    _DEFAULT_STREAM = 0

    def __init__(self) -> None:
        super().__init__()
        self.memory_intervals = {}

    def _stream_id(self, event: TraceEvent):
        dialect = PipelineContextTool.get_dialect_of_event(event)
        assert dialect is not None, "Cannot determine input dialect for memory event"

        if dialect.get("NAME") == "TORCH":
            stream = event["args"].get("stream", self._DEFAULT_STREAM)
            return event["pid"], stream

        return (event["pid"],)

    def add_memory_event(self, event: TraceEvent) -> None:
        stream_id = self._stream_id(event)
        start = event["ts"]
        end = round(event["ts"] + event["dur"], 4)

        intervals = self.memory_intervals.setdefault(stream_id, [])

        if intervals and start < intervals[-1]["end"]:
            intervals[-1]["end"] = max(intervals[-1]["end"], end)
            return

        intervals.append({
            "start": start,
            "end": end,
            "event": event,
        })

    def make_memory_superevents(self) -> list[TraceEvent]:
        superevents = []

        for intervals in self.memory_intervals.values():
            for interval in intervals:
                event = copy.deepcopy(interval["event"])

                event["name"] = "Memory operations"
                event["cat"] = "gpu_memory"
                event["ts"] = interval["start"]
                event["dur"] = interval["end"] - interval["start"]
                event["args"]["_memory_superevent"] = True

                superevents.append(event)

        return superevents

    def drain(self) -> list[TraceEvent]:
        if self.collection_phase():
            TwoPhaseWithBarrierContext.drain(self)
            return self.make_memory_superevents()

        return super().drain()



def memory_overlap_collect(event: TraceEvent, context: AbstractContext) -> list[TraceEvent]:
    assert isinstance(context, MemoryOverlapContext)

    if event["ph"] != "X" or not _is_memory_event(event):
        return [event]

    context.add_memory_event(event)
    return [event]


class OverlapVerificationContext(OverlapDetectionContext, AbstractVerificationContext):
    '''
    Verification-mode variant of the overlap detection: it only reports overlapping compute
    events, it never resolves them (always OVERLAP_RESOLVE_WARN). Every detected overlap is
    recorded as an instance of an error-level finding and the accumulated findings are emitted
    as verification meta-data events by AbstractVerificationContext.drain() (reached via the MRO,
    the parent drain chains up to it; no two-phase barrier required for this context).

    Compute streams are identified per dialect: FLEX events have a single stream per pid, while
    TORCH events separate the streams within a pid by their 'args.stream' entry.
    '''
    test_name = "Compute Overlap Check"

    _OVERLAP_WARNING = "overlaps"
    _STREAM_KEY = "stream"
    _STREAM_ARG = "args." + _STREAM_KEY
    _DEFAULT_STREAM = 0  # convention: actual stream numbers start at 1, 0 means 'no stream entry'

    def __init__(self, strict=False) -> None:
        super().__init__(self.OVERLAP_RESOLVE_WARN, strict=strict)
        # replace the resolution-oriented warning of the parent: in verification mode nothing is
        # resolved, a detected overlap is a finding that has to fail the test
        self.add_warning(
            TraceWarning(
                name=self._OVERLAP_WARNING,
                text="Overlapping accelerator events detected: {d[count]}",
                data={"count": 0},
                is_error=True,
            )
        )

    def _select_queue_id_keys(self, event: TraceEvent) -> list[str]:
        dialect = PipelineContextTool.get_dialect_of_event(event)
        assert dialect is not None, \
            "OVL: cannot determine the dialect of the first event." \
            " Register this stage before any stage that removes the jobhash."
        if dialect.get("NAME") == "TORCH":
            return ["pid", self._STREAM_ARG]
        return ["pid"]

    def _record_overlap(self, oevent: TraceEvent) -> None:
        super()._record_overlap(oevent)
        self.warnings[self._OVERLAP_WARNING].add_instance({
            "name": oevent["name"],
            "pid": oevent["pid"],
            "tid": oevent["tid"],
            "stream": oevent["args"].get(self._STREAM_KEY),
            "ts": oevent["ts"],
            "dur": oevent["dur"],
        })

    def add_default_stream(self, event: TraceEvent) -> bool:
        '''
        the stream-based separation of queues requires the stream entry to exist. Events without
        one are all attributed to the same default stream. Returns whether a default was added.
        '''
        if self._STREAM_KEY in event["args"]:
            return False
        event["args"][self._STREAM_KEY] = self._DEFAULT_STREAM
        return True

    def remove_default_stream(self, event: TraceEvent) -> None:
        del event["args"][self._STREAM_KEY]


def verify_kernel_overlap(event: TraceEvent, context: AbstractContext) -> list[TraceEvent]:
    assert isinstance(context, OverlapVerificationContext)

    # only compute kernels and temporary memory superevents are checked
    if event["ph"] not in "X":
        return [event]

    if not PipelineContextTool.is_acc_kernel(event) and not _is_memory_superevent(event):
        return [event]

    # the default is only required to determine the queue/stream of this event: drop it again to
    # keep the event unchanged for any downstream stage
    default_added = context.add_default_stream(event)
    revents = context.overlap_detection(event)
    if default_added:
        context.remove_default_stream(event)
    return revents
