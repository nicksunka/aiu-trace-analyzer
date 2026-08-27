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
    _is_memory_superevent,
    memory_overlap_collect,
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


def _run_memory_stage(events):
    context = MemoryOverlapContext()
    output = []

    for event in events:
        output.extend(memory_overlap_collect(event, context))

    output.extend(context.drain())
    return output


def _memory_superevents(events):
    return [event for event in events if _is_memory_superevent(event)]


def test_same_stream_kernel_memory_superevent_overlap_is_detected():
    memory = _event(
        "Memcpy (DtoH)",
        "gpu_memcpy",
        110.0,
        20.0,
        1,
    )

    output = _run_memory_stage([memory])
    superevent = _memory_superevents(output)[0]

    events = [
        _event("kernel_1", "kernel", 100.0, 20.0, 1),
        superevent,
    ]

    assert _run(events) == 1


def test_cross_stream_kernel_memory_superevent_overlap_is_allowed():
    memory = _event(
        "Memcpy (DtoH)",
        "gpu_memcpy",
        110.0,
        20.0,
        2,
    )

    output = _run_memory_stage([memory])
    superevent = _memory_superevents(output)[0]

    events = [
        _event("kernel_1", "kernel", 100.0, 20.0, 1),
        superevent,
    ]

    assert _run(events) == 0


def test_memory_event_classification():
    memcpy = _event("Memcpy (DtoH)", "gpu_memcpy", 100.0, 10.0, 1)
    memset = _event("Memset (Device)", "gpu_memset", 100.0, 10.0, 1)
    kernel = _event("kernel_1", "kernel", 100.0, 10.0, 1)

    assert _is_memory_event(memcpy)
    assert _is_memory_event(memset)
    assert not _is_memory_event(kernel)


def test_memory_context_merges_overlapping_events_on_same_stream():
    events = [
        _event("Memcpy (HtoD)", "gpu_memcpy", 100.0, 20.0, 1),
        _event("Memset (Device)", "gpu_memset", 110.0, 20.0, 1),
    ]

    output = _run_memory_stage(events)
    superevents = _memory_superevents(output)

    assert len(superevents) == 1
    assert superevents[0]["ts"] == 100.0
    assert superevents[0]["dur"] == 30.0
    assert superevents[0]["args"]["stream"] == 1


def test_memory_context_keeps_back_to_back_events_separate():
    events = [
        _event("Memcpy (HtoD)", "gpu_memcpy", 100.0, 20.0, 1),
        _event("Memcpy (DtoH)", "gpu_memcpy", 120.0, 20.0, 1),
    ]

    output = _run_memory_stage(events)
    superevents = _memory_superevents(output)

    assert len(superevents) == 2
    assert superevents[0]["ts"] == 100.0
    assert superevents[0]["dur"] == 20.0
    assert superevents[1]["ts"] == 120.0
    assert superevents[1]["dur"] == 20.0


def test_memory_context_keeps_separate_groups_on_same_stream():
    events = [
        _event("Memcpy (HtoD)", "gpu_memcpy", 100.0, 10.0, 1),
        _event("Memcpy (DtoH)", "gpu_memcpy", 120.0, 10.0, 1),
    ]

    output = _run_memory_stage(events)
    superevents = _memory_superevents(output)

    assert len(superevents) == 2
    assert superevents[0]["ts"] == 100.0
    assert superevents[1]["ts"] == 120.0


def test_memory_context_keeps_streams_separate():
    events = [
        _event("Memcpy (HtoD)", "gpu_memcpy", 100.0, 20.0, 1),
        _event("Memcpy (DtoH)", "gpu_memcpy", 110.0, 20.0, 2),
    ]

    output = _run_memory_stage(events)
    superevents = _memory_superevents(output)

    assert len(superevents) == 2
    assert superevents[0]["args"]["stream"] == 1
    assert superevents[1]["args"]["stream"] == 2


def test_memory_context_uses_default_stream_when_missing():
    first = _event(
        "Memcpy (HtoD)",
        "gpu_memcpy",
        100.0,
        20.0,
        1,
    )
    second = _event(
        "Memcpy (DtoH)",
        "gpu_memcpy",
        110.0,
        20.0,
        1,
    )

    first["args"].pop("stream")
    second["args"].pop("stream")

    output = _run_memory_stage([first, second])
    superevents = _memory_superevents(output)

    assert len(superevents) == 1
    assert superevents[0]["ts"] == 100.0
    assert superevents[0]["dur"] == 30.0
    assert "stream" not in superevents[0]["args"]


def test_memory_superevent_does_not_modify_original_event():
    original = _event(
        "Memcpy (HtoD)",
        "gpu_memcpy",
        100.0,
        20.0,
        1,
    )

    output = _run_memory_stage([original])
    superevent = _memory_superevents(output)[0]

    assert superevent is not original
    assert original["name"] == "Memcpy (HtoD)"
    assert original["cat"] == "gpu_memcpy"
    assert original["ts"] == 100.0
    assert original["dur"] == 20.0
    assert "_memory_superevent" not in original["args"]


def test_memory_context_holds_events_while_group_is_open():
    context = MemoryOverlapContext()

    memory = _event(
        "Memcpy (HtoD)",
        "gpu_memcpy",
        100.0,
        50.0,
        1,
    )
    kernel = _event(
        "kernel_1",
        "kernel",
        110.0,
        10.0,
        1,
    )

    assert memory_overlap_collect(memory, context) == []
    assert memory_overlap_collect(kernel, context) == []

    output = context.drain()

    assert [(event["name"], event["ts"]) for event in output] == [
        ("Memory operations", 100.0),
        ("Memcpy (HtoD)", 100.0),
        ("kernel_1", 110.0),
    ]


def test_memory_context_releases_finished_group_before_later_kernel():
    context = MemoryOverlapContext()

    memory = _event(
        "Memcpy (HtoD)",
        "gpu_memcpy",
        100.0,
        20.0,
        1,
    )
    kernel = _event(
        "kernel_1",
        "kernel",
        150.0,
        10.0,
        1,
    )

    assert memory_overlap_collect(memory, context) == []

    output = memory_overlap_collect(kernel, context)

    assert [(event["name"], event["ts"]) for event in output] == [
        ("Memory operations", 100.0),
        ("Memcpy (HtoD)", 100.0),
        ("kernel_1", 150.0),
    ]


def test_memory_context_preserves_overlap_verifier_ordering():
    memory_context = MemoryOverlapContext()
    overlap_context = OverlapVerificationContext(strict=True)

    events = [
        _event(
            "Memcpy (HtoD)",
            "gpu_memcpy",
            100.0,
            20.0,
            1,
        ),
        _event(
            "kernel_1",
            "kernel",
            150.0,
            10.0,
            1,
        ),
    ]

    for event in events:
        for emitted in memory_overlap_collect(
            event,
            memory_context,
        ):
            verify_kernel_overlap(emitted, overlap_context)

    for emitted in memory_context.drain():
        verify_kernel_overlap(emitted, overlap_context)

    assert overlap_context.warnings["overlaps"].args_list["count"] == 0


def test_memory_context_drain_flushes_unfinished_group():
    context = MemoryOverlapContext()

    memory = _event(
        "Memcpy (HtoD)",
        "gpu_memcpy",
        100.0,
        20.0,
        1,
    )

    assert memory_overlap_collect(memory, context) == []

    output = context.drain()

    assert [(event["name"], event["ts"]) for event in output] == [
        ("Memory operations", 100.0),
        ("Memcpy (HtoD)", 100.0),
    ]

    assert context.queues == {}
    assert context.hold == []


def test_memory_context_releases_only_safe_events_with_multiple_streams():
    context = MemoryOverlapContext()

    stream_1_memory = _event(
        "Memcpy stream 1",
        "gpu_memcpy",
        100.0,
        20.0,
        1,
    )
    stream_2_memory = _event(
        "Memcpy stream 2",
        "gpu_memcpy",
        110.0,
        90.0,
        2,
    )
    kernel = _event(
        "kernel_1",
        "kernel",
        150.0,
        10.0,
        1,
    )

    assert memory_overlap_collect(stream_1_memory, context) == []
    assert memory_overlap_collect(stream_2_memory, context) == []

    output = memory_overlap_collect(kernel, context)

    assert [(event["name"], event["ts"]) for event in output] == [
        ("Memory operations", 100.0),
        ("Memcpy stream 1", 100.0),
    ]

    remaining = context.drain()

    assert [(event["name"], event["ts"]) for event in remaining] == [
        ("Memory operations", 110.0),
        ("Memcpy stream 2", 110.0),
        ("kernel_1", 150.0),
    ]


def test_same_stream_memory_overlap_is_detected():
    events = [
        _event("Memcpy (HtoD)", "gpu_memcpy", 100.0, 20.0, 1),
        _event("Memcpy (DtoH)", "gpu_memcpy", 110.0, 20.0, 1),
    ]

    assert _run(events) == 1


def test_cross_stream_memory_overlap_is_allowed():
    events = [
        _event("Memcpy (HtoD)", "gpu_memcpy", 100.0, 20.0, 1),
        _event("Memcpy (DtoH)", "gpu_memcpy", 110.0, 20.0, 2),
    ]

    assert _run(events) == 0


def test_back_to_back_same_stream_memory_is_allowed():
    events = [
        _event("Memcpy (HtoD)", "gpu_memcpy", 100.0, 20.0, 1),
        _event("Memcpy (DtoH)", "gpu_memcpy", 120.0, 20.0, 1),
    ]

    assert _run(events) == 0


def test_nested_same_stream_memory_overlap_is_detected():
    events = [
        _event("Memcpy (HtoD)", "gpu_memcpy", 100.0, 40.0, 1),
        _event("Memcpy (DtoH)", "gpu_memcpy", 110.0, 10.0, 1),
    ]

    assert _run(events) == 1


def test_missing_stream_memory_overlap_uses_default_stream():
    first = _event(
        "Memcpy (HtoD)",
        "gpu_memcpy",
        100.0,
        20.0,
        1,
    )
    second = _event(
        "Memcpy (DtoH)",
        "gpu_memcpy",
        110.0,
        20.0,
        1,
    )

    first["args"].pop("stream")
    second["args"].pop("stream")

    assert _run([first, second]) == 1


def test_memory_superevent_is_consumed_after_overlap_check():
    context = OverlapVerificationContext(strict=True)

    event = _event(
        "Memory operations",
        "gpu_memory",
        100.0,
        20.0,
        1,
    )
    event["args"]["_memory_superevent"] = True

    assert verify_kernel_overlap(event, context) == []
