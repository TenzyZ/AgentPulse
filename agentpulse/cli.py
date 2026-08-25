"""Minimal Phase 1 command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from agentpulse.demo_target import create_demo_target
from agentpulse.env_hygiene import check_env_hygiene
from agentpulse.findings import Finding, Status
from agentpulse.remediation import Approval, Decision, RemediationPlan
from agentpulse.workflow import run_workflow


def _print_finding(finding: Finding) -> None:
    print(json.dumps(finding.to_sanitized_dict(), indent=2, sort_keys=True))


def _status_exit(status: Status) -> int:
    return {Status.PASS: 0, Status.FAIL: 1, Status.ERROR: 2}[status]


def _run_fix(target: Path) -> int:
    finding_presented = False

    def request_approval(
        finding: Finding,
        plan: RemediationPlan,
    ) -> Approval:
        nonlocal finding_presented
        _print_finding(finding)
        finding_presented = True
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
    commands.add_parser("demo")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "check":
        finding = check_env_hygiene(arguments.target)
        _print_finding(finding)
        return _status_exit(finding.status)
    if arguments.command == "fix":
        return _run_fix(arguments.target)

    target = create_demo_target()
    print(f"Persistent demo target: {target}")
    return _run_fix(target)


if __name__ == "__main__":
    raise SystemExit(main())
