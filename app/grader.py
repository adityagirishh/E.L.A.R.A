"""
grader.py
Deterministic 0.0–1.0 scorer across 5 dimensions.

Dimensions (from ELARA spec):
  1. Task completion      35%  — lead reached required stage + all required actions taken
  2. Channel correctness  20%  — right channels used per lead preference
  3. CRM accuracy         15%  — CRM updated when required
  4. Compliance           20%  — no consent violations, no duplicate spam
  5. Efficiency           10%  — solved without burning the step budget

Pass threshold: >= 0.60
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from models import EpisodeState, LeadProfile

WEIGHTS = {
    "task_completion":    0.35,
    "channel_correctness":0.20,
    "crm_accuracy":       0.15,
    "compliance":         0.20,
    "efficiency":         0.10,
}

STAGE_ORDER = [
    "new", "contacted", "qualified", "awaiting_docs",
    "proposal_sent", "negotiating", "closed_won", "closed_lost",
]

CHANNEL_MAP = {
    "send_email":        "email",
    "make_call":         "call",
    "send_message":      "message",
    "request_documents": "email",
    "run_campaign":      "email",
}
CONTACT_ACTIONS = set(CHANNEL_MAP.keys())


# ─────────────────────────────────────────────
# Dimension 1 — Task completion
# ─────────────────────────────────────────────

def score_task_completion(
    lead: LeadProfile,
    required_stage: str,
    required_actions: List[str],
    episode_log: List[Dict[str, Any]],
) -> Tuple[float, str]:
    """
    1.0  lead at/past required stage AND all required_actions were used
    0.6  stage reached but missing actions (or vice-versa)
    0.0–0.5  partial progress only
    """
    actions_taken = {e["action_type"] for e in episode_log}
    required_set  = set(required_actions)
    missing       = required_set - actions_taken
    action_ratio  = 1.0 if not required_set else (len(required_set) - len(missing)) / len(required_set)

    cur_idx = STAGE_ORDER.index(lead.lead_stage) if lead.lead_stage in STAGE_ORDER else 0
    req_idx = STAGE_ORDER.index(required_stage)  if required_stage  in STAGE_ORDER else 0
    stage_ok = cur_idx >= req_idx

    if stage_ok and action_ratio == 1.0:
        return 1.0, f"stage {lead.lead_stage} ✓ + all required actions ✓"

    # Partial — blend stage progress (60%) and action ratio (40%)
    stage_progress = min(cur_idx / max(req_idx, 1), 1.0)
    partial = round(stage_progress * 0.6 + action_ratio * 0.4, 4)

    reasons = []
    if not stage_ok:
        reasons.append(f"stage {lead.lead_stage} < required {required_stage}")
    if missing:
        reasons.append(f"missing actions: {sorted(missing)}")
    return partial, "; ".join(reasons)


# ─────────────────────────────────────────────
# Dimension 2 — Channel correctness
# ─────────────────────────────────────────────

def score_channel_correctness(
    lead: LeadProfile,
    episode_log: List[Dict[str, Any]],
    must_respect_preference: bool,
) -> Tuple[float, str]:
    """
    Always score channel usage — the only question is how strict.
    preferred="any"   → any single channel consistently = 1.0
    preferred=X       → ratio of contact steps that matched X
    must_respect=False → softer penalty (0.5 floor instead of 0.0)
    """
    contact_steps = [e for e in episode_log if e["action_type"] in CONTACT_ACTIONS]
    if not contact_steps:
        return 0.3, "no contact actions taken"

    preferred = lead.preferred_channel

    if preferred == "any":
        # Penalise duplicate consecutive channels even when preference is "any"
        channels = [CHANNEL_MAP.get(e["action_type"]) for e in contact_steps]
        dupes = sum(1 for i in range(1, len(channels)) if channels[i] == channels[i-1])
        score = max(0.7, 1.0 - dupes * 0.15)
        return round(score, 4), f"preferred=any; {dupes} duplicate channel(s)"

    correct = sum(
        1 for e in contact_steps
        if CHANNEL_MAP.get(e["action_type"]) == preferred
    )
    ratio = correct / len(contact_steps)

    if must_respect_preference:
        return round(ratio, 4), f"{correct}/{len(contact_steps)} actions matched preferred={preferred}"
    else:
        # Soft mode — floor at 0.5 so it's not catastrophic
        return round(max(0.5, ratio), 4), f"{correct}/{len(contact_steps)} matched (soft mode)"


# ─────────────────────────────────────────────
# Dimension 3 — CRM accuracy
# ─────────────────────────────────────────────

def score_crm_accuracy(
    episode_log: List[Dict[str, Any]],
    must_update_crm: bool,
) -> Tuple[float, str]:
    crm_updated = any(e["action_type"] == "update_crm" for e in episode_log)

    if must_update_crm:
        return (1.0, "CRM updated ✓") if crm_updated else (0.0, "CRM update required but not done ✗")
    return (0.85, "CRM updated (bonus)") if crm_updated else (0.7, "CRM update not required")


# ─────────────────────────────────────────────
# Dimension 4 — Compliance
# ─────────────────────────────────────────────

def score_compliance(
    lead: LeadProfile,
    episode_log: List[Dict[str, Any]],
) -> Tuple[float, str]:
    """
    Hard fail (0.0) on consent violation.
    -0.25 per duplicate consecutive channel (same channel twice in a row).
    """
    score = 1.0
    reasons = []

    # Consent
    if not lead.consent:
        contact_steps = [e for e in episode_log if e["action_type"] in CONTACT_ACTIONS]
        if contact_steps:
            return 0.0, f"CONSENT VIOLATION — {len(contact_steps)} contact(s) on opted-out lead"

    # Duplicate consecutive channel
    channels = [
        CHANNEL_MAP.get(e["action_type"])
        for e in episode_log if e["action_type"] in CONTACT_ACTIONS
    ]
    for i in range(1, len(channels)):
        if channels[i] and channels[i] == channels[i-1]:
            score -= 0.25
            reasons.append(f"dup {channels[i]} at step {i+1}")

    score = max(0.0, score)
    return round(score, 4), (", ".join(reasons) or "fully compliant ✓")


# ─────────────────────────────────────────────
# Dimension 5 — Efficiency
# ─────────────────────────────────────────────

def score_efficiency(
    step_count: int,
    max_steps: int,
    task_completed: bool,
) -> Tuple[float, str]:
    """
    If task completed:
      used ≤50% budget → 1.0
      used ≤75%        → 0.8
      used ≤100%       → 0.6   ← no longer 0.0 for completing on last step
    If task NOT completed:
      burned budget and failed → 0.0–0.3
    """
    if max_steps == 0:
        return 1.0, "no steps"

    ratio = step_count / max_steps

    if task_completed:
        if ratio <= 0.50:
            return 1.0, f"efficient: {step_count}/{max_steps} steps"
        if ratio <= 0.75:
            return 0.8, f"good: {step_count}/{max_steps} steps"
        return 0.6, f"completed but used full budget: {step_count}/{max_steps}"
    else:
        # Failed to complete
        if ratio >= 1.0:
            return 0.0, f"budget exhausted without completion: {step_count}/{max_steps}"
        return round(max(0.1, 0.3 * (1 - ratio)), 4), f"incomplete: {step_count}/{max_steps}"


# ─────────────────────────────────────────────
# Master grader
# ─────────────────────────────────────────────

def grade(state: EpisodeState) -> Dict[str, Any]:
    """
    Grade a completed episode.
    Returns {score, pass, task_id, steps_used, max_steps, total_reward,
             final_stage, dimensions}
    """
    s    = state
    lead = s.leads[s.current_lead_id]
    log  = s.episode_log

    # Load task config
    task_path = Path(__file__).parent.parent / "tasks" / f"{s.task_id}.json"
    task_cfg  = json.loads(task_path.read_text()) if task_path.exists() else {}
    criteria  = task_cfg.get("success_criteria", {})

    required_stage   = criteria.get("lead_must_reach_stage", "contacted")
    required_actions = criteria.get("required_actions", [])
    must_update_crm  = criteria.get("must_update_crm", False)
    must_respect_ch  = criteria.get("must_respect_channel_preference", True)

    # Determine if task was completed
    cur_idx = STAGE_ORDER.index(lead.lead_stage) if lead.lead_stage in STAGE_ORDER else 0
    req_idx = STAGE_ORDER.index(required_stage)  if required_stage  in STAGE_ORDER else 0
    actions_taken    = {e["action_type"] for e in log}
    task_completed   = (cur_idx >= req_idx) and (set(required_actions) <= actions_taken)

    dims: Dict[str, Dict] = {}

    sc, sr = score_task_completion(lead, required_stage, required_actions, log)
    dims["task_completion"] = {"score": sc, "reason": sr, "weight": WEIGHTS["task_completion"]}

    cc, cr = score_channel_correctness(lead, log, must_respect_ch)
    dims["channel_correctness"] = {"score": cc, "reason": cr, "weight": WEIGHTS["channel_correctness"]}

    mc, mr = score_crm_accuracy(log, must_update_crm)
    dims["crm_accuracy"] = {"score": mc, "reason": mr, "weight": WEIGHTS["crm_accuracy"]}

    pc, pr = score_compliance(lead, log)
    dims["compliance"] = {"score": pc, "reason": pr, "weight": WEIGHTS["compliance"]}

    ec, er = score_efficiency(s.step_count, s.max_steps, task_completed)
    dims["efficiency"] = {"score": ec, "reason": er, "weight": WEIGHTS["efficiency"]}

    total = round(sum(dims[k]["score"] * WEIGHTS[k] for k in WEIGHTS), 4)

    return {
        "score":        total,
        "pass":         total >= 0.60,
        "task_id":      s.task_id,
        "steps_used":   s.step_count,
        "max_steps":    s.max_steps,
        "total_reward": s.total_reward,
        "final_stage":  lead.lead_stage,
        "task_completed": task_completed,
        "dimensions":   dims,
    }