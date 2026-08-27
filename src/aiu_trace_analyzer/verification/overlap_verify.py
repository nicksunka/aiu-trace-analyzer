# Copyright 2024-2026 IBM Corporation

"""
Overlap verification for accelerator events.

Accelerator events on the same stream are checked for invalid overlaps. Detected kernel/kernel,
kernel/memory, and memory/memory overlaps are reported as error-level findings of the verification
report.

This module reuses the overlap detection of the regular processing pipeline
(OverlapDetectionContext) and only replaces the parts that differ in verification mode:
  * overlaps are reported, never resolved
  * the finding is error-level and records each offending event as an instance
  * the streams are identified per input dialect instead of by pid+tid
"""

import copy

from aiu_trace_analyzer.pipeline.context import (
    AbstractContext,
    AbstractVerificationContext,
)
from aiu_trace_analyzer.pipeline.hashqueue import AbstractHashQueueContext
from aiu_trace_analyzer.pipeline.overlap import OverlapDetectionContext
from aiu_trace_analyzer.pipeline.tools import PipelineContextTool
from aiu_trace_analyzer.types import TraceEvent, TraceWarning


def _is_memory_event(event: TraceEvent) -> bool:
    """Return whether an accelerator event represents a memory operation."""
    dialect = PipelineContextTool.get_dialect_of_event(event)
    if dialect is None:
        return False

    if dialect.get("NAME") == "TORCH":
        return event.get("cat") in {"gpu_memcpy", "gpu_memset"}

    return PipelineContextTool.is_category(
        event, "acc_datatransfer_HtoD"
    ) or PipelineContextTool.is_category(event, "acc_datatransfer_DtoH")


def _is_memory_superevent(event: TraceEvent) -> bool:
    return event.get("args", {}).get("_memory_superevent") is True


class MemoryOverlapContext(AbstractHashQueueContext):
    """Coalesce memory operations while preserving event order."""

    _DEFAULT_STREAM = 0

    def __init__(self) -> None:
        super().__init__()
        self.hold = []

    def _stream_id(self, event: TraceEvent):
        dialect = PipelineContextTool.get_dialect_of_event(event)
        assert dialect is not None, "Cannot determine input dialect for memory event"

        if dialect.get("NAME") == "TORCH":
            stream = event["args"].get("stream", self._DEFAULT_STREAM)
            return event["pid"], stream

        return (event["pid"],)

    def _make_superevent(self, event: TraceEvent) -> TraceEvent:
        superevent = copy.deepcopy(event)
        superevent["name"] = "Memory operations"
        superevent["cat"] = "gpu_memory"
        superevent["args"]["_memory_superevent"] = True
        return superevent

    def _open_memory_group(self, event: TraceEvent) -> None:
        stream_id = self._stream_id(event)
        end = round(event["ts"] + event["dur"], 4)

        superevent = self._make_superevent(event)

        self.queues[stream_id] = {
            "start": event["ts"],
            "end": end,
            "superevent": superevent,
        }

        # The temporary event has to appear before the original memory event.
        self.hold.append(superevent)

    def _extend_memory_group(self, stream_id, event: TraceEvent) -> None:
        group = self.queues[stream_id]
        end = round(event["ts"] + event["dur"], 4)

        group["end"] = max(group["end"], end)
        group["superevent"]["dur"] = group["end"] - group["start"]

    def _close_finished_groups(self, timestamp: float) -> None:
        finished = [
            stream_id
            for stream_id, group in self.queues.items()
            if group["end"] <= timestamp
        ]

        for stream_id in finished:
            del self.queues[stream_id]

    def _release_ready_events(self) -> list[TraceEvent]:
        if not self.queues:
            ready = self.hold
            self.hold = []
            return ready

        watermark = min(group["start"] for group in self.queues.values())

        split = 0
        while split < len(self.hold) and self.hold[split]["ts"] < watermark:
            split += 1

        ready = self.hold[:split]
        self.hold = self.hold[split:]
        return ready

    def process_event(self, event: TraceEvent) -> list[TraceEvent]:
        self._close_finished_groups(event["ts"])

        if event["ph"] == "X" and _is_memory_event(event):
            stream_id = self._stream_id(event)

            if stream_id in self.queues:
                self._extend_memory_group(stream_id, event)
            else:
                self._open_memory_group(event)

        self.hold.append(event)
        return self._release_ready_events()

    def drain(self) -> list[TraceEvent]:
        self.queues.clear()

        remaining = self.hold
        self.hold = []
        return remaining


def memory_overlap_collect(
    event: TraceEvent,
    context: AbstractContext,
) -> list[TraceEvent]:
    assert isinstance(context, MemoryOverlapContext)
    return context.process_event(event)


class _MemoryOverlapDetectionContext(OverlapDetectionContext):
    """Detect overlaps between original memory events on the same stream."""

    _STREAM_ARG = "args.stream"

    def __init__(self, verification_context, strict=False) -> None:
        super().__init__(self.OVERLAP_RESOLVE_WARN, strict=strict)
        self.verification_context = verification_context

    def _select_queue_id_keys(self, event: TraceEvent) -> list[str]:
        dialect = PipelineContextTool.get_dialect_of_event(event)
        assert dialect is not None, (
            "OVL: cannot determine the dialect of the first event."
        )

        if dialect.get("NAME") == "TORCH":
            return ["pid", self._STREAM_ARG]

        return ["pid"]

    def _record_overlap(self, event: TraceEvent) -> None:
        self.verification_context._record_overlap(event)


class OverlapVerificationContext(OverlapDetectionContext, AbstractVerificationContext):
    """
    Verification-mode variant of the overlap detection: it reports overlapping accelerator
    events without resolving them (always OVERLAP_RESOLVE_WARN). Every detected overlap is
    recorded as an instance of an error-level finding and the accumulated findings are emitted
    as verification meta-data events by AbstractVerificationContext.drain() (reached via the MRO,
    the parent drain chains up to it; no two-phase barrier required for this context).

    Compute streams are identified per dialect: FLEX events have a single stream per pid, while
    TORCH events separate the streams within a pid by their 'args.stream' entry.
    """

    test_name = "Compute Overlap Check"

    _OVERLAP_WARNING = "overlaps"
    _STREAM_KEY = "stream"
    _STREAM_ARG = "args." + _STREAM_KEY
    _DEFAULT_STREAM = (
        0  # convention: actual stream numbers start at 1, 0 means 'no stream entry'
    )

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
        self.memory_overlap_context = _MemoryOverlapDetectionContext(
            self,
            strict=strict,
        )

    def _select_queue_id_keys(self, event: TraceEvent) -> list[str]:
        dialect = PipelineContextTool.get_dialect_of_event(event)
        assert dialect is not None, (
            "OVL: cannot determine the dialect of the first event."
            " Register this stage before any stage that removes the jobhash."
        )
        if dialect.get("NAME") == "TORCH":
            return ["pid", self._STREAM_ARG]
        return ["pid"]

    def _record_overlap(self, oevent: TraceEvent) -> None:
        super()._record_overlap(oevent)
        self.warnings[self._OVERLAP_WARNING].add_instance(
            {
                "name": oevent["name"],
                "pid": oevent["pid"],
                "tid": oevent["tid"],
                "stream": oevent["args"].get(self._STREAM_KEY),
                "ts": oevent["ts"],
                "dur": oevent["dur"],
            }
        )

    def add_default_stream(self, event: TraceEvent) -> bool:
        """
        the stream-based separation of queues requires the stream entry to exist. Events without
        one are all attributed to the same default stream. Returns whether a default was added.
        """
        if self._STREAM_KEY in event["args"]:
            return False
        event["args"][self._STREAM_KEY] = self._DEFAULT_STREAM
        return True

    def remove_default_stream(self, event: TraceEvent) -> None:
        del event["args"][self._STREAM_KEY]


def verify_kernel_overlap(
    event: TraceEvent,
    context: AbstractContext,
) -> list[TraceEvent]:
    assert isinstance(context, OverlapVerificationContext)

    if event["ph"] != "X":
        return [event]

    if _is_memory_event(event):
        default_added = context.add_default_stream(event)
        context.memory_overlap_context.overlap_detection(event)

        if default_added:
            context.remove_default_stream(event)

        return [event]

    if not PipelineContextTool.is_acc_kernel(event) and not _is_memory_superevent(
        event
    ):
        return [event]

    default_added = context.add_default_stream(event)
    revents = context.overlap_detection(event)

    if default_added:
        context.remove_default_stream(event)

    if _is_memory_superevent(event):
        return []

    return revents
