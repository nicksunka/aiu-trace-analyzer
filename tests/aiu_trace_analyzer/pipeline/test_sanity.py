# Copyright 2024-2026 IBM Corporation

import pytest

from aiu_trace_analyzer.pipeline.sanity import (
    PipelineSanityCheckContext,
    VerificationSanityCheckContext,
    sanity_check,
)
from aiu_trace_analyzer.types import TraceEvent


def _duration_event(dur, cycles_ts=None) -> TraceEvent:
    args = {}
    if cycles_ts is not None:
        args["cycles_ts"] = cycles_ts

    return TraceEvent({
        "ph": "X",
        "ts": 1.0,
        "pid": 0,
        "tid": 1,
        "name": "test_event",
        "dur": dur,
        "args": args,
    })


def _valid_event() -> TraceEvent:
    return TraceEvent({
        "ph": "X",
        "ts": 1.0,
        "pid": 0,
        "name": "test_event",
    })


def test_valid_event_passes_unchanged():
    ctx = PipelineSanityCheckContext()
    event = _valid_event()

    result = sanity_check(event, ctx)

    assert result == [event]
    assert ctx.warnings["missing_required_key"].to_finding_args()["count"] == 0


@pytest.mark.parametrize("missing_key", ["ph", "ts", "pid", "name"])
def test_missing_required_key_drops_event_and_records_finding(missing_key):
    ctx = PipelineSanityCheckContext()
    event = _valid_event()
    del event[missing_key]

    result = sanity_check(event, ctx)

    finding = ctx.warnings["missing_required_key"].to_finding_args()

    assert result == []
    assert finding["count"] == 1
    assert finding["is_error"] is True
    assert finding["instances"][0]["missing_key"] == missing_key


def test_each_malformed_event_counts_once():
    ctx = PipelineSanityCheckContext()

    first = _valid_event()
    del first["ts"]

    second = _valid_event()
    del second["pid"]

    sanity_check(first, ctx)
    sanity_check(second, ctx)

    finding = ctx.warnings["missing_required_key"].to_finding_args()

    assert finding["count"] == 2
    assert [instance["missing_key"] for instance in finding["instances"]] == [
        "ts",
        "pid",
    ]


def test_event_with_multiple_missing_keys_counts_once():
    ctx = PipelineSanityCheckContext()
    event = _valid_event()
    del event["ph"]
    del event["ts"]

    result = sanity_check(event, ctx)

    finding = ctx.warnings["missing_required_key"].to_finding_args()

    assert result == []
    assert finding["count"] == 1
    assert len(finding["instances"]) == 1


def test_event_processor_has_no_implicit_sanity_stage():
    from aiu_trace_analyzer.core.processing import EventProcessor

    processor = EventProcessor()

    assert processor.stages == []


def _processor_with_sanity_stage(context=None):
    from aiu_trace_analyzer.core.processing import EventProcessor
    from aiu_trace_analyzer.core.stage_profile import StageProfile

    profile = StageProfile(
        {"stages": [{"sanity_check": True}]},
        {"stages": [{"sanity_check": True}]},
    )
    processor = EventProcessor(profile=profile)
    processor.register_stage(
        callback=sanity_check,
        context=context if context is not None else PipelineSanityCheckContext(),
    )
    return processor


def test_missing_required_key_finding_is_emitted_on_drain():
    from aiu_trace_analyzer.types import TRACE_FINDING_EVENT_NAME

    processor = _processor_with_sanity_stage()
    event = _valid_event()
    del event["name"]

    assert processor.process(event) == []

    drained = processor.drain()
    findings = [
        event for event in drained
        if event.name == TRACE_FINDING_EVENT_NAME
    ]

    missing_key = next(
        finding
        for finding in findings
        if finding.args["finding"] == "missing_required_key"
    )
    assert missing_key.args["count"] == 1
    assert missing_key.args["is_error"] is True
    assert missing_key.args["instances"] == [{"missing_key": "name"}]


def test_missing_required_key_fails_verification_report():
    from aiu_trace_analyzer.export.exporter import VerificationReportExporter

    processor = _processor_with_sanity_stage(VerificationSanityCheckContext())
    event = _valid_event()
    del event["pid"]

    assert processor.process(event) == []

    exporter = VerificationReportExporter(
        "unused.json",
        settings={"save_to_file": False},
    )
    exporter.export(processor.drain())

    assert exporter.has_errors is True

    report = exporter.get_data()
    assert report["result"] == "FAIL"
    assert [finding["finding"] for finding in report["errors"]] == [
        "missing_required_key",
    ]
    assert {
        "test": "Event Sanity Check",
        "result": "fail",
    } in report["test_results"]


def test_pipeline_sanity_does_not_emit_verification_result():
    from aiu_trace_analyzer.types import VERIFICATION_TEST_RESULT_NAME

    processor = _processor_with_sanity_stage(PipelineSanityCheckContext())

    drained = processor.drain()

    assert all(
        event.name != VERIFICATION_TEST_RESULT_NAME
        for event in drained
    )


def test_verification_sanity_reports_pass_when_no_findings():
    from aiu_trace_analyzer.export.exporter import VerificationReportExporter

    processor = _processor_with_sanity_stage(VerificationSanityCheckContext())

    exporter = VerificationReportExporter(
        "unused.json",
        settings={"save_to_file": False},
    )
    exporter.export(processor.drain())

    report = exporter.get_data()

    assert {
        "test": "Event Sanity Check",
        "result": "pass",
    } in report["test_results"]


def test_negative_duration_warns_and_drops_in_pipeline():
    ctx = PipelineSanityCheckContext()
    event = _duration_event(-1.0)

    result = sanity_check(event, ctx)

    finding = ctx.warnings["negative_duration"].to_finding_args()
    assert result == []
    assert finding["count"] == 1
    assert finding["is_error"] is False
    assert finding["instances"][0]["dur"] == -1.0


def test_negative_duration_fails_verification_and_drops():
    ctx = VerificationSanityCheckContext()
    event = _duration_event(-1.0)

    result = sanity_check(event, ctx)

    finding = ctx.warnings["negative_duration"].to_finding_args()
    assert result == []
    assert finding["count"] == 1
    assert finding["is_error"] is True


def test_invalid_duration_is_error_and_drops():
    ctx = PipelineSanityCheckContext()
    event = _duration_event("invalid")

    result = sanity_check(event, ctx)

    finding = ctx.warnings["invalid_duration"].to_finding_args()
    assert result == []
    assert finding["count"] == 1
    assert finding["is_error"] is True


def test_zero_duration_warns_and_is_kept():
    ctx = PipelineSanityCheckContext()
    event = _duration_event(0.0)

    result = sanity_check(event, ctx)

    finding = ctx.warnings["zero_duration"].to_finding_args()
    assert result == [event]
    assert finding["count"] == 1
    assert finding["is_error"] is False


def test_zero_duration_with_nonzero_hw_time_warns_and_is_kept():
    ctx = PipelineSanityCheckContext()
    event = _duration_event(0.0, cycles_ts=[10, 10, 10, 10, 20])

    result = sanity_check(event, ctx)

    assert result == [event]
    assert ctx.warnings["zero_duration"].to_finding_args()["count"] == 1
    assert ctx.warnings["zero_with_hw_time"].to_finding_args()["count"] == 1


def test_zero_duration_with_zero_hw_time_has_no_hw_warning():
    ctx = PipelineSanityCheckContext()
    event = _duration_event(0.0, cycles_ts=[10, 10, 10, 10, 10])

    result = sanity_check(event, ctx)

    assert result == [event]
    assert ctx.warnings["zero_duration"].to_finding_args()["count"] == 1
    assert ctx.warnings["zero_with_hw_time"].to_finding_args()["count"] == 0


def test_duration_checks_are_independent_of_phase():
    ctx = PipelineSanityCheckContext()
    event = _duration_event(-1.0)
    event["ph"] = "i"

    assert sanity_check(event, ctx) == []
    assert ctx.warnings["negative_duration"].to_finding_args()["count"] == 1


def test_complete_event_without_duration_passes_unchanged():
    ctx = PipelineSanityCheckContext()
    event = _duration_event(1.0)
    del event["dur"]

    assert sanity_check(event, ctx) == [event]
    assert ctx.warnings["negative_duration"].to_finding_args()["count"] == 0


def test_negative_duration_fails_verification_report():
    from aiu_trace_analyzer.export.exporter import VerificationReportExporter

    processor = _processor_with_sanity_stage(VerificationSanityCheckContext())

    assert processor.process(_duration_event(-1.0)) == []

    exporter = VerificationReportExporter(
        "unused.json",
        settings={"save_to_file": False},
    )
    exporter.export(processor.drain())

    report = exporter.get_data()

    assert report["result"] == "FAIL"
    assert "negative_duration" in [
        finding["finding"] for finding in report["errors"]
    ]


def test_zero_hw_time_check_supports_attr():
    ctx = PipelineSanityCheckContext()
    event = _duration_event(0.0)
    event["attr"] = {"cycles_ts": [10, 10, 10, 10, 20]}
    del event["args"]

    assert sanity_check(event, ctx) == [event]
    finding = ctx.warnings["zero_with_hw_time"].to_finding_args()
    assert finding["count"] == 1
    assert finding["instances"][0]["ts1"] == 10
    assert finding["instances"][0]["ts5"] == 20


def test_duration_finding_contains_event_location():
    from aiu_trace_analyzer.dialect import InputDialectTORCH
    from aiu_trace_analyzer.types import GlobalIngestData

    jobhash = GlobalIngestData.add_job_info(
        source_uri="sanity-location.pt.trace.json",
        data_dialect=InputDialectTORCH(),
    )
    event = _duration_event(-1.0)
    event["pid"] = 4
    event["args"] = {
        "jobhash": jobhash,
        "rank": 4,
        "stream": 2,
    }

    ctx = PipelineSanityCheckContext()
    sanity_check(event, ctx)

    instance = ctx.warnings["negative_duration"].to_finding_args()["instances"][0]
    assert instance["rank"] == 4
    assert instance["pid"] == 4
    assert instance["stream"] == 2
    assert instance["source"] == "sanity-location.pt.trace.json"


def test_duration_instances_are_capped_but_count_is_not():
    ctx = PipelineSanityCheckContext()
    warning = ctx.warnings["zero_duration"]

    for _ in range(warning.max_instances + 1):
        sanity_check(_duration_event(0.0), ctx)

    finding = warning.to_finding_args()
    assert finding["count"] == warning.max_instances + 1
    assert len(finding["instances"]) == warning.max_instances


def test_silenced_negative_duration_is_still_dropped():
    ctx = PipelineSanityCheckContext(
        silenced_findings={"negative_duration"},
    )

    assert sanity_check(_duration_event(-1.0), ctx) == []
    assert ctx.warnings["negative_duration"].to_finding_args()["count"] == 0


def test_silenced_zero_duration_is_kept_without_finding():
    ctx = PipelineSanityCheckContext(
        silenced_findings={"zero_duration", "zero_with_hw_time"},
    )
    event = _duration_event(0.0, cycles_ts=[10, 10, 10, 10, 20])

    assert sanity_check(event, ctx) == [event]
    assert ctx.warnings["zero_duration"].to_finding_args()["count"] == 0
    assert ctx.warnings["zero_with_hw_time"].to_finding_args()["count"] == 0


def test_errors_cannot_be_silenced():
    ctx = PipelineSanityCheckContext(
        silenced_findings={"missing_required_key", "invalid_duration"},
    )

    invalid = _duration_event("bad")
    assert sanity_check(invalid, ctx) == []
    assert ctx.warnings["invalid_duration"].to_finding_args()["count"] == 1


def test_duration_finding_is_exported_to_other_data():
    import json

    from aiu_trace_analyzer.export.exporter import JsonFileTraceExporter

    processor = _processor_with_sanity_stage(PipelineSanityCheckContext())

    event = _duration_event(0.0)
    processed = processor.process(event)
    assert len(processed) == 1

    exporter = JsonFileTraceExporter(
        target_uri="unused.json",
    )
    exporter.export(processor.drain())

    other_data = json.loads(exporter.get_data())["otherData"]
    findings = {
        finding["finding"]: finding
        for finding in other_data["warnings"]
    }

    assert findings["zero_duration"]["count"] == 1
