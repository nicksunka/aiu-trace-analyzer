# Copyright 2024-2026 IBM Corporation

from aiu_trace_analyzer.types import (
    GlobalIngestData,
    InputDialectTORCH,
    TraceEvent,
)
from aiu_trace_analyzer.verification.overlap_verify import (
    MemoryOverlapContext,
    OverlapVerificationContext,
    _is_memory_event,
    verify_kernel_overlap,
)


def _torch_jobhash():
    return GlobalIngestData.add_job_info(
        source_uri="test_torch_trace.json",
        data_dialect=InputDialectTORCH(),
    )


def _event(name, cat, ts, dur, stream):
    return TraceEvent(
        {
            "name": name,
            "cat": cat,
            "ph": "X",
            "pid": 0,
            "tid": 0,
            "ts": ts,
            "dur": dur,
            "args": {
                "stream": stream,
                "jobhash": _torch_jobhash(),
            },
        }
    )


def _run(events):
    context = OverlapVerificationContext(strict=True)

    for event in events:
        verify_kernel_overlap(event, context)

    return context.warnings["overlaps"].args_list["count"]


def test_same_stream_kernel_overlap_is_detected():
    events = [
        _event("kernel_1", "kernel", 100.0, 20.0, 1),
        _event("kernel_2", "kernel", 110.0, 20.0, 1),
    ]

    assert _run(events) == 1


def test_cross_stream_kernel_overlap_is_allowed():
    events = [
        _event("kernel_1", "kernel", 100.0, 20.0, 1),
        _event("kernel_2", "kernel", 110.0, 20.0, 2),
    ]

    assert _run(events) == 0


def test_original_memory_events_are_not_checked_directly():
    events = [
        _event("Memcpy (HtoD)", "gpu_memcpy", 100.0, 20.0, 1),
        _event("Memcpy (DtoH)", "gpu_memcpy", 110.0, 20.0, 1),
    ]

    assert _run(events) == 0


def test_same_stream_kernel_memory_superevent_overlap_is_detected():
    memory_context = MemoryOverlapContext()

    memory_context.add_memory_event(
        _event("Memcpy (DtoH)", "gpu_memcpy", 110.0, 20.0, 1)
    )

    memory_superevents = memory_context.make_memory_superevents()

    events = [
        _event("kernel_1", "kernel", 100.0, 20.0, 1),
        memory_superevents[0],
    ]

    assert _run(events) == 1




def _interval_bounds(context, stream_id):
    bounds = []

    for interval in context.memory_intervals[stream_id]:
        bounds.append({
            "start": interval["start"],
            "end": interval["end"],
        })

    return bounds

def test_memory_event_classification():
    memcpy = _event("Memcpy (DtoH)", "gpu_memcpy", 100.0, 10.0, 1)
    memset = _event("Memset (Device)", "gpu_memset", 100.0, 10.0, 1)
    kernel = _event("kernel_1", "kernel", 100.0, 10.0, 1)

    assert _is_memory_event(memcpy)
    assert _is_memory_event(memset)
    assert not _is_memory_event(kernel)


def test_memory_context_merges_overlapping_events_on_same_stream():
    context = MemoryOverlapContext()

    context.add_memory_event(
        _event("Memcpy (HtoD)", "gpu_memcpy", 100.0, 20.0, 1)
    )
    context.add_memory_event(
        _event("Memset (Device)", "gpu_memset", 110.0, 20.0, 1)
    )

    assert _interval_bounds(context, (0, 1)) == [
        {"start": 100.0, "end": 130.0}
    ]


def test_memory_context_keeps_separate_intervals_on_same_stream():
    context = MemoryOverlapContext()

    context.add_memory_event(
        _event("Memcpy (HtoD)", "gpu_memcpy", 100.0, 10.0, 1)
    )
    context.add_memory_event(
        _event("Memcpy (DtoH)", "gpu_memcpy", 120.0, 10.0, 1)
    )

    assert _interval_bounds(context, (0, 1)) == [
        {"start": 100.0, "end": 110.0},
        {"start": 120.0, "end": 130.0},
    ]


def test_memory_context_keeps_streams_separate():
    context = MemoryOverlapContext()

    context.add_memory_event(
        _event("Memcpy (HtoD)", "gpu_memcpy", 100.0, 20.0, 1)
    )
    context.add_memory_event(
        _event("Memcpy (DtoH)", "gpu_memcpy", 110.0, 20.0, 2)
    )

    assert _interval_bounds(context, (0, 1)) == [
        {"start": 100.0, "end": 120.0}
    ]
    assert _interval_bounds(context, (0, 2)) == [
        {"start": 110.0, "end": 130.0}
    ]


def test_memory_context_uses_default_stream_when_missing():
    context = MemoryOverlapContext()

    event = _event("Memcpy (DtoH)", "gpu_memcpy", 100.0, 20.0, 1)
    event["args"].pop("stream")

    context.add_memory_event(event)

    assert _interval_bounds(context, (0, 0)) == [
        {"start": 100.0, "end": 120.0}
    ]



def test_memory_context_keeps_back_to_back_events_separate():
    context = MemoryOverlapContext()

    context.add_memory_event(
        _event("Memcpy (HtoD)", "gpu_memcpy", 100.0, 20.0, 1)
    )
    context.add_memory_event(
        _event("Memcpy (DtoH)", "gpu_memcpy", 120.0, 20.0, 1)
    )

    assert _interval_bounds(context, (0, 1)) == [
        {"start": 100.0, "end": 120.0},
        {"start": 120.0, "end": 140.0},
    ]



def test_memory_context_creates_one_superevent_for_merged_interval():
    context = MemoryOverlapContext()

    context.add_memory_event(
        _event("Memcpy (HtoD)", "gpu_memcpy", 100.0, 20.0, 1)
    )
    context.add_memory_event(
        _event("Memset (Device)", "gpu_memset", 110.0, 20.0, 1)
    )

    superevents = context.make_memory_superevents()

    assert len(superevents) == 1

    event = superevents[0]
    assert event["name"] == "Memory operations"
    assert event["cat"] == "gpu_memory"
    assert event["ts"] == 100.0
    assert event["dur"] == 30.0
    assert event["args"]["stream"] == 1
    assert event["args"]["_memory_superevent"] is True


def test_memory_context_creates_superevent_for_each_interval():
    context = MemoryOverlapContext()

    context.add_memory_event(
        _event("Memcpy (HtoD)", "gpu_memcpy", 100.0, 10.0, 1)
    )
    context.add_memory_event(
        _event("Memcpy (DtoH)", "gpu_memcpy", 120.0, 10.0, 1)
    )

    superevents = context.make_memory_superevents()

    assert len(superevents) == 2

    assert superevents[0]["ts"] == 100.0
    assert superevents[0]["dur"] == 10.0

    assert superevents[1]["ts"] == 120.0
    assert superevents[1]["dur"] == 10.0


def test_memory_superevent_does_not_modify_original_event():
    context = MemoryOverlapContext()

    original = _event(
        "Memcpy (HtoD)",
        "gpu_memcpy",
        100.0,
        20.0,
        1,
    )

    context.add_memory_event(original)
    context.make_memory_superevents()

    assert original["name"] == "Memcpy (HtoD)"
    assert original["cat"] == "gpu_memcpy"
    assert original["ts"] == 100.0
    assert original["dur"] == 20.0
    assert "_memory_superevent" not in original["args"]



def test_memory_context_starts_in_collection_phase():
    context = MemoryOverlapContext()

    assert context.collection_phase()


def test_memory_context_drain_creates_superevents():
    context = MemoryOverlapContext()

    context.add_memory_event(
        _event("Memcpy (HtoD)", "gpu_memcpy", 100.0, 20.0, 1)
    )
    context.add_memory_event(
        _event("Memset (Device)", "gpu_memset", 110.0, 20.0, 1)
    )

    events = context.drain()

    assert not context.collection_phase()
    assert len(events) == 1
    assert events[0]["name"] == "Memory operations"
    assert events[0]["ts"] == 100.0
    assert events[0]["dur"] == 30.0
