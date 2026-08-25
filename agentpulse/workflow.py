"""Phase 1 detect, authorize, mutate, and verify orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agentpulse.env_hygiene import check_env_hygiene
from agentpulse.findings import Finding, Status
from agentpulse.remediation import (
    Approval,
    RemediationOutcome,
    RemediationPlan,
    apply_plan,
    build_plan,
)

ApprovalProvider = Callable[[Finding, RemediationPlan], Approval | None]


@dataclass(frozen=True)
class WorkflowResult:
    before: Finding
    plan: RemediationPlan | None
    outcome: RemediationOutcome | None
    after: Finding
    transition: str
    verified: bool


def run_workflow(
    target_root: str | Path,
    approval_provider: ApprovalProvider | None = None,
) -> WorkflowResult:
    before = check_env_hygiene(target_root)
    plan = build_plan(before)
    if plan is None:
        return WorkflowResult(
            before=before,
            plan=None,
            outcome=None,
            after=before,
            transition=f"{before.status.value}->{before.status.value}",
            verified=False,
        )

    approval = approval_provider(before, plan) if approval_provider else None
    outcome = apply_plan(plan, approval)
    after = check_env_hygiene(target_root)
    transition = f"{before.status.value}->{after.status.value}"
    verified = (
        before.status is Status.FAIL
        and outcome.applied
        and outcome.integrity_verified
        and after.status is Status.PASS
    )
    return WorkflowResult(
        before=before,
        plan=plan,
        outcome=outcome,
        after=after,
        transition=transition,
        verified=verified,
    )
