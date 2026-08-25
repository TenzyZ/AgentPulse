"""Minimal AgentPulse command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from agentpulse.demo_target import create_demo_target
from agentpulse.env_hygiene import check_env_hygiene
from agentpulse.findings import Finding, Status
from agentpulse.reasoning import ReasoningOutcome
from agentpulse.remediation import Approval, Decision, RemediationPlan
from agentpulse.workflow import run_workflow

Reasoner = Callable[[Finding], ReasoningOutcome]


def _print_finding(finding: Finding) -> None:
    print(json.dumps(finding.to_sanitized_dict(), indent=2, sort_keys=True))


def _status_exit(status: Status) -> int:
    return {Status.PASS: 0, Status.FAIL: 1, Status.ERROR: 2}[status]


def _print_reasoning(outcome: ReasoningOutcome) -> None:
    common = {
        "model": outcome.model,
        "prompt_version": outcome.prompt_version,
        "payload_sha256": outcome.payload_sha256,
    }
    if outcome.result is not None:
        print("Reasoning:")
        print(
            json.dumps(
                {**common, **outcome.result.model_dump(mode="json")},
                indent=2,
                sort_keys=True,
            )
        )
        return
    assert outcome.failure is not None
    print("Reasoning failure:")
    print(
        json.dumps(
            {
                **common,
                "code": outcome.failure.code.value,
                "detail": outcome.failure.detail,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _reason(finding: Finding, reasoner: Reasoner | None) -> ReasoningOutcome:
    if reasoner is None:
        from agentpulse.gemini_bridge import reason_finding

        reasoner = reason_finding
    return reasoner(finding)


def _run_fix(
    target: Path,
    *,
    use_reasoning: bool = False,
    reasoner: Reasoner | None = None,
) -> int:
    finding_presented = False

    def request_approval(
        finding: Finding,
        plan: RemediationPlan,
    ) -> Approval | None:
        nonlocal finding_presented
        _print_finding(finding)
        finding_presented = True
        if use_reasoning:
            reasoning = _reason(finding, reasoner)
            _print_reasoning(reasoning)
            if reasoning.failure is not None:
                print("Authorization was not requested because reasoning failed.")
                return None
        print(plan.unified_diff, end="")
        if not sys.stdin.isatty():
            return Approval(
                decision=Decision.DENIED,
                plan_fingerprint=plan.fingerprint,
            )
        try:
            response = input("Apply this exact remediation? Type yes to approve: ")
        except EOFError:
            response = ""
        decision = Decision.APPROVED if response.strip().lower() == "yes" else Decision.DENIED
        return Approval(decision=decision, plan_fingerprint=plan.fingerprint)

    result = run_workflow(target, request_approval)
    if not finding_presented:
        _print_finding(result.before)
    if result.plan is None:
        print("No remediation plan.")
    else:
        assert result.outcome is not None
        print(f"Applied: {str(result.outcome.applied).lower()}")
        print(f"Outcome: {result.outcome.reason}")
        print(f"Before/after: {result.transition}")
        print(f"Verified: {str(result.verified).lower()}")
        _print_finding(result.after)
    return _status_exit(result.after.status)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentpulse")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "fix"):
        command = commands.add_parser(name)
        command.add_argument("--target", required=True, type=Path)
        command.add_argument("--reason", action="store_true")
    commands.add_parser("demo")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    reasoner: Reasoner | None = None,
) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "check":
        finding = check_env_hygiene(arguments.target)
        _print_finding(finding)
        if arguments.reason and finding.status in (Status.FAIL, Status.ERROR):
            _print_reasoning(_reason(finding, reasoner))
        return _status_exit(finding.status)
    if arguments.command == "fix":
        return _run_fix(
            arguments.target,
            use_reasoning=arguments.reason,
            reasoner=reasoner,
        )

    target = create_demo_target()
    print(f"Persistent demo target: {target}")
    return _run_fix(target)


if __name__ == "__main__":
    raise SystemExit(main())
