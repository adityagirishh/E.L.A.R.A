"""
grader.py
Deterministic 0.0–1.0 scorer across 5 dimensions.

Dimensions:
  1. Task completion      35%  — leads reached required stages + all required actions taken
  2. Channel correctness  20%  — right channels used per lead preference
  3. CRM accuracy         15%  — CRM updated when required
  4. Compliance           20%  — no consent violations, no duplicate spam
  5. Efficiency           10%  — solved without burning the step budget

v3: supports multi-lead tasks (grade each lead, blend scores).
Pass threshold: >= 0.60
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from models import EpisodeState, LeadProfile

WEIGHTS = {
    "task_completion":     0.35,
    "channel_correctness": 0.20,
    "crm_accuracy":        0.15,
    "compliance":          0.20,
    "efficiency":          0.10,
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
    leads: Dict[str, LeadProfile],
    lead_criteria: Dict[str, Dict[str, Any]],
    episode_log: List[Dict[str, Any]],
) -> Tuple[float, str]:
    """
    For each lead in lead_criteria, check:
      - Lead at/past required stage
      - All required actions taken (targeting that lead)
    Multi-lead: average across all leads.
    """
    if not lead_criteria:
        return 0.0, "no criteria defined"

    scores = []
    reasons = []

    for lid, criteria in lead_criteria.items():
        lead = leads.get(lid)
        if not lead:
            scores.append(0.0)
            reasons.append(f"{lid}: lead not found")
            continue

        required_stage   = criteria.get("lead_must_reach_stage", "contacted")
        required_actions = criteria.get("required_actions", [])

        # Actions targeting this specific lead
        lead_log = [e for e in episode_log if e.get("target_lead") == lid]
        actions_taken = {e["action_type"] for e in lead_log}
        # Also check global actions (update_crm, escalate might not have specific lead filter)
        all_actions = {e["action_type"] for e in episode_log if e.get("target_lead") == lid}

        required_set = set(required_actions)
        missing = required_set - all_actions
        action_ratio = 1.0 if not required_set else (len(required_set) - len(missing)) / len(required_set)

        cur_idx = STAGE_ORDER.index(lead.lead_stage) if lead.lead_stage in STAGE_ORDER else 0
        req_idx = STAGE_ORDER.index(required_stage)  if required_stage  in STAGE_ORDER else 0
        stage_ok = cur_idx >= req_idx

        if stage_ok and action_ratio == 1.0:
            scores.append(1.0)
            reasons.append(f"{lid}: stage {lead.lead_stage} ✓ + all actions ✓")
        else:
            stage_progress = min(cur_idx / max(req_idx, 1), 1.0)
            partial = round(stage_progress * 0.6 + action_ratio * 0.4, 4)
            r = []
            if not stage_ok:
                r.append(f"stage {lead.lead_stage} < required {required_stage}")
            if missing:
                r.append(f"missing: {sorted(missing)}")
            scores.append(partial)
            reasons.append(f"{lid}: {'; '.join(r)}")

    avg = sum(scores) / len(scores)
    return round(avg, 4), " | ".join(reasons)


# ─────────────────────────────────────────────
# Dimension 2 — Channel correctness
# ─────────────────────────────────────────────

def score_channel_correctness(
    leads: Dict[str, LeadProfile],
    active_lead_ids: List[str],
    episode_log: List[Dict[str, Any]],
) -> Tuple[float, str]:
    contact_steps = [e for e in episode_log if e["action_type"] in CONTACT_ACTIONS]
    if not contact_steps:
        return 0.3, "no contact actions taken"

    correct = 0
    total = 0
    reasons = []

    for step in contact_steps:
        lid = step.get("target_lead", "")
        lead = leads.get(lid)
        if not lead:
            continue

        preferred = lead.preferred_channel
        action_channel = CHANNEL_MAP.get(step["action_type"])
        total += 1

        # Email-only actions are always exempt
        if step["action_type"] in {"request_documents", "run_campaign"}:
            correct += 1
            continue

        if preferred == "any" or action_channel == preferred:
            correct += 1
        else:
            reasons.append(f"step {step.get('step')}: {action_channel} != {preferred} for {lid}")

    # Duplicate channel penalty — only when same channel AND same lead back-to-back
    channel_lead = [(CHANNEL_MAP.get(e["action_type"]), e.get("target_lead")) for e in contact_steps]
    dupes = sum(
        1 for i in range(1, len(channel_lead))
        if channel_lead[i][0] == channel_lead[i-1][0] and channel_lead[i][1] == channel_lead[i-1][1]
    )

    ratio = correct / max(total, 1)
    dupe_penalty = dupes * 0.10
    score = max(0.0, round(ratio - dupe_penalty, 4))

    reason = f"{correct}/{total} correct"
    if dupes:
        reason += f", {dupes} consecutive duplicates"
    if reasons:
        reason += f" — {'; '.join(reasons[:3])}"
    return score, reason


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
    leads: Dict[str, LeadProfile],
    episode_log: List[Dict[str, Any]],
    lead_responses: List[Dict[str, Any]],
) -> Tuple[float, str]:
    """
    Hard fail (0.0) on consent violation.
    -0.25 per duplicate consecutive channel.
    -0.30 for contacting a lead AFTER they revoked consent.
    """
    score = 1.0
    reasons = []

    # Check for consent violations — including mid-episode revocations
    # Build a timeline: which leads revoked consent and at which step
    revoked_leads = set()
    for resp in lead_responses:
        mutations = resp.get("mutations", {})
        if mutations.get("consent") is False:
            revoked_leads.add(resp.get("lead_id", ""))

    # Check for contact with originally opted-out leads
    for step in episode_log:
        if step["action_type"] not in CONTACT_ACTIONS:
            continue
        lid = step.get("target_lead", "")

        # Check if this lead had consent=False from the start
        original_lead = SEED_LEADS_CACHE.get(lid)
        if original_lead and not original_lead.consent:
            score = 0.0
            reasons.append(f"CONSENT VIOLATION: contacted {lid} (originally opted out)")
            break

    # Check for post-revocation contact
    if revoked_leads:
        # Find the step where revocation happened
        revocation_steps = {}
        for i, entry in enumerate(episode_log):
            if entry.get("lead_response"):
                lid = entry.get("target_lead", "")
                if lid in revoked_leads:
                    revocation_steps[lid] = entry.get("step", i)

        for step in episode_log:
            if step["action_type"] not in CONTACT_ACTIONS:
                continue
            lid = step.get("target_lead", "")
            if lid in revocation_steps and step.get("step", 0) > revocation_steps[lid]:
                score -= 0.30
                reasons.append(f"contacted {lid} AFTER consent revocation at step {revocation_steps[lid]}")

    # Duplicate consecutive channel — only when same channel AND same lead back-to-back
    contact_entries = [
        (CHANNEL_MAP.get(e["action_type"]), e.get("target_lead"))
        for e in episode_log if e["action_type"] in CONTACT_ACTIONS
    ]
    for i in range(1, len(contact_entries)):
        ch, lid = contact_entries[i]
        prev_ch, prev_lid = contact_entries[i - 1]
        if ch and ch == prev_ch and lid == prev_lid:
            score -= 0.25
            reasons.append(f"dup {ch} to {lid} at step {i+1}")

    score = max(0.0, round(score, 4))
    return score, (", ".join(reasons) or "fully compliant ✓")


# ─────────────────────────────────────────────
# Dimension 5 — Efficiency
# ─────────────────────────────────────────────

def score_efficiency(
    step_count: int,
    max_steps: int,
    task_completed: bool,
) -> Tuple[float, str]:
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
        if ratio >= 1.0:
            return 0.0, f"budget exhausted without completion: {step_count}/{max_steps}"
        return round(max(0.1, 0.3 * (1 - ratio)), 4), f"incomplete: {step_count}/{max_steps}"


# ─────────────────────────────────────────────
# Cache seed leads for consent checking
# ─────────────────────────────────────────────

from crm_simulator import SEED_LEADS as _SEED_LEADS
SEED_LEADS_CACHE = {k: v.model_copy() for k, v in _SEED_LEADS.items()}


# ─────────────────────────────────────────────
# Master grader
# ─────────────────────────────────────────────

def grade(state: EpisodeState) -> Dict[str, Any]:
    s    = state
    log  = s.episode_log

    # Load task config
    task_path = Path(__file__).parent.parent / "tasks" / f"{s.task_id}.json"
    task_cfg  = json.loads(task_path.read_text()) if task_path.exists() else {}
    criteria  = task_cfg.get("success_criteria", {})

    # Build per-lead criteria
    # Support both single-lead ("lead_must_reach_stage") and multi-lead ("per_lead") formats
    lead_ids = s.active_lead_ids or [s.current_lead_id]

    if "per_lead" in criteria:
        lead_criteria = criteria["per_lead"]
    else:
        # Single-lead backward-compatible format
        lead_criteria = {
            lead_ids[0]: {
                "lead_must_reach_stage": criteria.get("lead_must_reach_stage", "contacted"),
                "required_actions": criteria.get("required_actions", []),
            }
        }

    must_update_crm = criteria.get("must_update_crm", False)

    # Determine if task was completed (all leads meet their criteria)
    task_completed = True
    for lid, lc in lead_criteria.items():
        lead = s.leads.get(lid)
        if not lead:
            task_completed = False
            continue
        req_stage = lc.get("lead_must_reach_stage", "contacted")
        req_actions = set(lc.get("required_actions", []))
        cur_idx = STAGE_ORDER.index(lead.lead_stage) if lead.lead_stage in STAGE_ORDER else 0
        req_idx = STAGE_ORDER.index(req_stage) if req_stage in STAGE_ORDER else 0
        lead_actions = {e["action_type"] for e in log if e.get("target_lead") == lid}
        if cur_idx < req_idx or not (req_actions <= lead_actions):
            task_completed = False

    dims: Dict[str, Dict] = {}

    sc, sr = score_task_completion(s.leads, lead_criteria, log)
    dims["task_completion"] = {"score": sc, "reason": sr, "weight": WEIGHTS["task_completion"]}

    cc, cr = score_channel_correctness(s.leads, lead_ids, log)
    dims["channel_correctness"] = {"score": cc, "reason": cr, "weight": WEIGHTS["channel_correctness"]}

    mc, mr = score_crm_accuracy(log, must_update_crm)
    dims["crm_accuracy"] = {"score": mc, "reason": mr, "weight": WEIGHTS["crm_accuracy"]}

    pc, pr = score_compliance(s.leads, log, s.lead_responses)
    dims["compliance"] = {"score": pc, "reason": pr, "weight": WEIGHTS["compliance"]}

    ec, er = score_efficiency(s.step_count, s.max_steps, task_completed)
    dims["efficiency"] = {"score": ec, "reason": er, "weight": WEIGHTS["efficiency"]}

    total = round(sum(dims[k]["score"] * WEIGHTS[k] for k in WEIGHTS), 4)

    return {
        "score":          total,
        "pass":           total >= 0.60,
        "task_id":        s.task_id,
        "steps_used":     s.step_count,
        "max_steps":      s.max_steps,
        "total_reward":   s.total_reward,
        "final_stage":    s.leads[s.current_lead_id].lead_stage,
        "task_completed": task_completed,
        "dimensions":     dims,
    }
