# Copyright 2024-2026 IBM Corporation

import math

import aiu_trace_analyzer.logger as aiulog

from aiu_trace_analyzer.hw_data import get_hw_ts, has_hw_ts
from aiu_trace_analyzer.pipeline.context import (
    AbstractContext,
    AbstractVerificationContext,
)
from aiu_trace_analyzer.pipeline.tools import event_args_key, event_location
from aiu_trace_analyzer.types import TraceEvent, TraceWarning


_REQUIRED_KEYS = ["ph", "ts", "pid", "name"]
_SILENCEABLE_FINDINGS = {
    "negative_duration",
    "zero_duration",
    "zero_with_hw_time",
}


class _SanityWarnings:
    def __init__(
            self,
            negative_duration_error: bool = False,
            silenced_findings: set[str] = None) -> None:
        self.silenced_findings = (
            set(silenced_findings or set()) & _SILENCEABLE_FINDINGS
        )

        super().__init__(warnings=[
            TraceWarning(
                name="missing_required_key",
                text="SAN: Detected {d[count]} event(s) missing a required key.",
                data={"count": 0},
                is_error=True,
            ),
            TraceWarning(
                name="invalid_duration",
                text="SAN: Detected {d[count]} event(s) with invalid duration.",
                data={"count": 0},
                is_error=True,
            ),
            TraceWarning(
                name="negative_duration",
                text="SAN: Detected {d[count]} event(s) with negative duration.",
                data={"count": 0},
                is_error=negative_duration_error,
            ),
            TraceWarning(
                name="zero_duration",
                text="SAN: Detected {d[count]} event(s) with zero duration.",
                data={"count": 0},
            ),
            TraceWarning(
                name="zero_with_hw_time",
                text=(
                    "SAN: Detected {d[count]} event(s) with zero wallclock "
                    "duration but nonzero HW time."
                ),
                data={"count": 0},
            ),
        ])

    def record_finding(self, name: str, instance: dict) -> None:
        if name not in self.silenced_findings:
            self.warnings[name].record(instance)


class PipelineSanityCheckContext(_SanityWarnings, AbstractContext):
    def __init__(self, silenced_findings: set[str] = None) -> None:
        super().__init__(
            negative_duration_error=False,
            silenced_findings=silenced_findings,
        )


class VerificationSanityCheckContext(
        _SanityWarnings,
        AbstractVerificationContext,
):
    test_name = "Event Sanity Check"

    def __init__(self) -> None:
        super().__init__(negative_duration_error=True)


def sanity_check(event: TraceEvent, context: AbstractContext) -> list[TraceEvent]:
    assert isinstance(context, _SanityWarnings)

    for key in _REQUIRED_KEYS:
        if key not in event:
            aiulog.log(
                aiulog.ERROR,
                "Event failed sanityCheck: ",
                key,
                "is not in",
                event,
            )
            context.record_finding(
                "missing_required_key",
                {"missing_key": key},
            )
            return []

    if "dur" not in event:
        return [event]

    duration = event["dur"]

    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        context.record_finding(
            "invalid_duration",
            event_location(event),
        )
        return []

    if duration < 0.0:
        context.record_finding(
            "negative_duration",
            event_location(event),
        )
        return []

    if not math.isclose(duration, 0.0, abs_tol=1e-9):
        return [event]

    context.record_finding(
        "zero_duration",
        event_location(event),
    )

    args_key = event_args_key(event)
    if has_hw_ts(event, args_key):
        ts1 = get_hw_ts(event, 1, args_key)
        ts5 = get_hw_ts(event, 5, args_key)
        if ts1 != ts5:
            context.record_finding(
                "zero_with_hw_time",
                {
                    **event_location(event),
                    "ts1": ts1,
                    "ts5": ts5,
                },
            )

    return [event]
