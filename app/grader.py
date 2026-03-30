"""
grader.py
Deterministic 0.0–1.0 scorer across 5 dimensions.

Dimensions (from ELARA spec):
  1. Task completion      — did the lead reach the required stage?
  2. Channel correctness  — were the right channels used?
  3. CRM accuracy         — was the CRM updated when required?
  4. Compliance adherence — no consent violations, no spam
  5. Efficiency           — completed in budget steps?

Weights: completion 35%, channel 20%, crm 15%, compliance 20%, efficiency 10%
"""

from typing import Any, Dict, List, Tuple

from models import Action, EpisodeState, LeadProfile


WEIGHTS = {
    "task_completion": 0.35,
    "channel_correctness": 0.20,
    "crm_accuracy": 0.15,
    "compliance": 0.20,
    "efficiency": 0.10,
}

STAGE_ORDER = [
    "new", "contacted", "qualified", "awaiting_docs",
    "proposal_sent", "negotiating", "closed_won", "closed_lost",
]


# ─────────────────────────────────────────────
# Individual dimension scorers
# ─────────────────────────────────────────────

def score_task_completion(
    lead: LeadProfile,
    required_stage: str,
    required_actions: List[str],
    episode_log: List[Dict[str, Any]],
) -> Tuple[float, str]:
    """
    1.0 if lead is at or past required_stage AND all required_actions were used.
    Partial credit for partial progress.
    """
    actions_taken = {e["action_type"] for e in episode_log}
    required_set = set(required_actions)
    actions_done = required_set & actions_taken
    action_ratio = len(actions_done) / max(len(required_set), 1)

    current_idx  = STAGE_ORDER.index(lead.lead_stage) if lead.lead_stage in STAGE_ORDER else 0
    required_idx = STAGE_ORDER.index(required_stage)  if required_stage in STAGE_ORDER else 0

    if current_idx >= required_idx and action_ratio == 1.0:
        return 1.0, f"stage reached ({lead.lead_stage}) + all required actions taken"

    stage_progress = min(current_idx / max(required_idx, 1), 1.0)
    partial = (stage_progress * 0.6) + (action_ratio * 0.4)

    reasons = []
    if current_idx < required_idx:
        reasons.append(f"stage {lead.lead_stage} < required {required_stage}")
    missing = required_set - actions_taken
    if missing:
        reasons.append(f"missing actions: {missing}")

    return round(partial, 4), "; ".join(reasons) or "partial completion"


def score_channel_correctness(
    lead: LeadProfile,
    episode_log: List[Dict[str, Any]],
    must_respect_preference: bool,
) -> Tuple[float, str]:
    """
    Score how well the agent matched the lead's channel preference.
    1.0 = all contact actions matched preference.
    """
    CHANNEL_MAP = {
        "send_email": "email", "make_call": "call",
        "send_message": "message", "request_documents": "email",
        "run_campaign": "email",
    }
    CONTACT_ACTIONS = set(CHANNEL_MAP.keys())

    contact_steps = [e for e in episode_log if e["action_type"] in CONTACT_ACTIONS]
    if not contact_steps:
        return 0.5, "no contact actions taken"

    if not must_respect_preference or lead.preferred_channel == "any":
        return 1.0, "no channel constraint — any channel ok"

    correct = sum(
        1 for e in contact_steps
        if CHANNEL_MAP.get(e["action_type"]) == lead.preferred_channel
    )
    ratio = correct / len(contact_steps)
    return round(ratio, 4), f"{correct}/{len(contact_steps)} contact actions matched preferred channel"


def score_crm_accuracy(
    episode_log: List[Dict[str, Any]],
    must_update_crm: bool,
) -> Tuple[float, str]:
    """
    1.0 if CRM was updated when required (or wasn't required).
    0.0 if required but not done.
    """
    crm_updated = any(e["action_type"] == "update_crm" for e in episode_log)

    if not must_update_crm:
        return 1.0 if crm_updated else 0.7, (
            "CRM updated (bonus)" if crm_updated else "CRM update not required"
        )
    if crm_updated:
        return 1.0, "CRM updated as required"
    return 0.0, "CRM update required but not done"


def score_compliance(
    lead: LeadProfile,
    episode_log: List[Dict[str, Any]],
) -> Tuple[float, str]:
    """
    Starts at 1.0. Deduct for:
      - consent violation: -1.0 (hard fail)
      - duplicate channel in consecutive steps: -0.2 each
    """
    CONTACT_ACTIONS = {
        "send_email", "make_call", "send_message", "request_documents", "run_campaign"
    }
    CHANNEL_MAP = {
        "send_email": "email", "make_call": "call",
        "send_message": "message", "request_documents": "email",
        "run_campaign": "email",
    }

    score = 1.0
    reasons = []

    # Consent violations
    if not lead.consent:
        contact_steps = [e for e in episode_log if e["action_type"] in CONTACT_ACTIONS]
        if contact_steps:
            return 0.0, f"CONSENT VIOLATION: {len(contact_steps)} contact(s) on opted-out lead"

    # Duplicate consecutive channel
    channels = [
        CHANNEL_MAP.get(e["action_type"])
        for e in episode_log if e["action_type"] in CONTACT_ACTIONS
    ]
    for i in range(1, len(channels)):
        if channels[i] and channels[i] == channels[i - 1]:
            score -= 0.2
            reasons.append(f"duplicate {channels[i]} at step {i+1}")

    score = max(0.0, score)
    return round(score, 4), (", ".join(reasons) or "fully compliant")


def score_efficiency(
    step_count: int,
    max_steps: int,
) -> Tuple[float, str]:
    """
    1.0 if completed in ≤ 60% of budget. Scales down linearly after that.
    """
    if max_steps == 0:
        return 1.0, "no steps"
    ratio = step_count / max_steps
    if ratio <= 0.6:
        return 1.0, f"efficient: {step_count}/{max_steps} steps"
    score = max(0.0, 1.0 - (ratio - 0.6) / 0.4)
    return round(score, 4), f"{step_count}/{max_steps} steps used"


# ─────────────────────────────────────────────
# Master grader
# ─────────────────────────────────────────────

def grade(state: EpisodeState) -> Dict[str, Any]:
    """
    Grade a completed episode. Returns:
    {
        "score": 0.0–1.0,
        "dimensions": { ... per-dimension scores and reasons },
        "pass": bool,
    }
    """
    from crm_simulator import SEED_LEADS  # avoid circular at module level

    s = state
    lead = s.leads[s.current_lead_id]
    log  = s.episode_log

    # Load task config for success criteria
    import json
    from pathlib import Path
    task_path = Path(__file__).parent.parent / "tasks" / f"{s.task_id}.json"
    task_cfg = json.loads(task_path.read_text()) if task_path.exists() else {}
    criteria = task_cfg.get("success_criteria", {})

    required_stage   = criteria.get("lead_must_reach_stage", "contacted")
    required_actions = criteria.get("required_actions", [])
    must_update_crm  = criteria.get("must_update_crm", False)
    must_respect_ch  = criteria.get("must_respect_channel_preference", False)

    dims: Dict[str, Dict] = {}

    sc, sr = score_task_completion(lead, required_stage, required_actions, log)
    dims["task_completion"] = {"score": sc, "reason": sr, "weight": WEIGHTS["task_completion"]}

    cc, cr = score_channel_correctness(lead, log, must_respect_ch)
    dims["channel_correctness"] = {"score": cc, "reason": cr, "weight": WEIGHTS["channel_correctness"]}

    mc, mr = score_crm_accuracy(log, must_update_crm)
    dims["crm_accuracy"] = {"score": mc, "reason": mr, "weight": WEIGHTS["crm_accuracy"]}

    pc, pr = score_compliance(lead, log)
    dims["compliance"] = {"score": pc, "reason": pr, "weight": WEIGHTS["compliance"]}

    ec, er = score_efficiency(s.step_count, s.max_steps)
    dims["efficiency"] = {"score": ec, "reason": er, "weight": WEIGHTS["efficiency"]}

    total = sum(
        dims[k]["score"] * WEIGHTS[k] for k in WEIGHTS
    )
    total = round(total, 4)

    return {
        "score": total,
        "pass": total >= 0.6,
        "task_id": s.task_id,
        "steps_used": s.step_count,
        "max_steps": s.max_steps,
        "total_reward": s.total_reward,
        "final_stage": lead.lead_stage,
        "dimensions": dims,
    }