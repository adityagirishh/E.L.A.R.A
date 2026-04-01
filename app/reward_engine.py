"""
reward_engine.py  —  ELARA dense reward system

Reward table:
  Correct channel       +0.20
  Correct timing        +0.10
  Good message          +0.20   (body present + personalised + objection addressed)
  CRM update            +0.10
  Lead progression      +0.30   (+0.50 for closed_won)
  Policy compliance     +0.10

Penalties:
  Wrong channel         -0.20
  Duplicate outreach    -0.20   (same channel AND same goal back-to-back)
  Consent violation     -0.50
  Looping               -0.10

v3 improvements:
  - Objection addressing check: agent must use solution keywords, not just mention the objection
  - request_documents / run_campaign exempt from channel preference
  - Cross-channel same-day followup exempt from timing penalty
  - Duplicate penalty requires same channel + same goal
"""

from typing import Any, Dict, List, Tuple
from models import Action, LeadProfile, ProductProfile
from crm_simulator import OBJECTION_SOLUTIONS

CONTACT_ACTIONS = {
    "send_email", "make_call", "send_message",
    "request_documents", "run_campaign",
}

EMAIL_ONLY_ACTIONS = {"request_documents", "run_campaign"}

NON_CONTACT_ACTIONS = {
    "update_crm", "schedule_followup", "escalate", "wait",
}

ACTION_CHANNEL_MAP = {
    "send_email":        "email",
    "make_call":         "call",
    "send_message":      "message",
    "request_documents": "email",
    "run_campaign":      "email",
}


def _recent_history(lead: LeadProfile, n: int = 1) -> List[Dict[str, Any]]:
    return lead.conversation_history[-n:] if lead.conversation_history else []


# ── 1. Channel ────────────────────────────────────────────────────────────────

def reward_channel_choice(action: Action, lead: LeadProfile) -> Tuple[float, str]:
    at = action.action_type
    if at not in CONTACT_ACTIONS:
        return 0.0, "non-contact action"

    if at in EMAIL_ONLY_ACTIONS:
        return 0.20, f"{at} always uses email — channel exempt from preference"

    preferred = lead.preferred_channel
    channel   = ACTION_CHANNEL_MAP.get(at, "")

    if preferred == "any":
        return 0.20, f"channel {channel} acceptable (no preference set)"
    if channel == preferred:
        return 0.20, f"correct channel: {channel} matches lead preference"
    return -0.20, f"wrong channel: used {channel}, lead prefers {preferred}"


# ── 2. Timing ─────────────────────────────────────────────────────────────────

def reward_timing(
    action: Action,
    lead: LeadProfile,
    prev_action_type: str = "",
) -> Tuple[float, str]:
    if action.action_type not in CONTACT_ACTIONS:
        return 0.0, "non-contact action"

    days = lead.days_since_last_contact
    due  = lead.next_followup_due

    if lead.last_contact_channel == "none":
        return 0.10, "first contact — timing valid"

    prev_channel = ACTION_CHANNEL_MAP.get(prev_action_type, "")
    this_channel = ACTION_CHANNEL_MAP.get(action.action_type, "")

    # Cross-channel same-day followup is valid sales behaviour
    if days == 0 and prev_channel and this_channel and prev_channel != this_channel:
        return 0.10, f"immediate {this_channel} followup after {prev_channel} — valid cross-channel"

    if days < 1:
        return -0.10, f"contacted too soon ({days}d since last contact, same channel)"

    if days >= due:
        return 0.10, f"follow-up on time ({days}d elapsed, due at {due}d)"

    return 0.05, f"slightly early ({days}d elapsed, due at {due}d)"


# ── 3. Message quality ────────────────────────────────────────────────────────

def reward_message_quality(
    action: Action,
    lead: LeadProfile,
    product: ProductProfile,
) -> Tuple[float, str]:
    """
    Scoring:
      0.00 — empty body
      0.05 — body present but generic
      +0.05 — personalised (lead name or company)
      +0.05 — product-relevant (feature/value prop mentioned)
      +0.05 — objection addressed with solution keywords (not just mentioned)
    Max: +0.20
    """
    at = action.action_type
    if at in NON_CONTACT_ACTIONS:
        return 0.0, "non-message action"

    body = (action.body or "").strip()
    if not body:
        return 0.0, "empty message body"

    score, reasons = 0.05, ["body present"]
    body_lower = body.lower()

    # Personalised
    name_lower    = lead.lead_name.split()[0].lower()
    company_lower = lead.company.lower()
    if name_lower in body_lower or company_lower in body_lower:
        score += 0.05
        reasons.append("personalised")

    # Product-relevant
    product_terms = (
        [f.lower() for f in product.features] +
        [v.lower() for v in product.value_props]
    )
    if any(t[:8] in body_lower for t in product_terms if len(t) >= 5):
        score += 0.05
        reasons.append("product-relevant")

    # Objection addressed with solution keywords
    if lead.objections:
        addressed = False
        for obj in lead.objections:
            solutions = OBJECTION_SOLUTIONS.get(obj, [])
            if any(sol in body_lower for sol in solutions):
                addressed = True
                break
        if addressed:
            score += 0.05
            reasons.append("objection addressed")
        else:
            reasons.append("objection NOT addressed")

    return min(score, 0.20), ", ".join(reasons)


# ── 4. CRM update ─────────────────────────────────────────────────────────────

def reward_crm_update(action: Action) -> Tuple[float, str]:
    if action.action_type == "update_crm":
        return 0.10, "CRM updated"
    return 0.0, "no CRM update this step"


# ── 5. Lead progression ───────────────────────────────────────────────────────

def reward_lead_progression(
    action: Action,
    lead: LeadProfile,
    new_stage: str,
) -> Tuple[float, str]:
    STAGE_ORDER = [
        "new", "contacted", "qualified", "awaiting_docs",
        "proposal_sent", "negotiating", "closed_won",
    ]
    old_idx = STAGE_ORDER.index(lead.lead_stage) if lead.lead_stage in STAGE_ORDER else 0
    new_idx = STAGE_ORDER.index(new_stage)        if new_stage        in STAGE_ORDER else 0

    if new_idx > old_idx:
        bonus = 0.50 if new_stage == "closed_won" else 0.30
        return bonus, f"lead progressed: {lead.lead_stage} → {new_stage}"
    if new_idx < old_idx:
        return -0.10, f"lead regressed: {lead.lead_stage} → {new_stage}"
    return 0.0, "stage unchanged"


# ── 6. Compliance ─────────────────────────────────────────────────────────────

def reward_policy_compliance(
    action: Action,
    lead: LeadProfile,
) -> Tuple[float, str]:
    if not lead.consent and action.action_type in CONTACT_ACTIONS:
        return -0.50, "CONSENT VIOLATION: lead has consent=False"
    return 0.10, "policy compliant"


# ── 7. Duplicate outreach ─────────────────────────────────────────────────────

def reward_duplicate_outreach(
    action: Action,
    lead: LeadProfile,
    prev_action_type: str = "",
    prev_goal: str = "",
) -> Tuple[float, str]:
    """
    -0.20 only when BOTH: same channel as previous + same goal (or both empty).
    """
    at = action.action_type
    if at not in CONTACT_ACTIONS:
        return 0.0, "non-contact action"

    if not prev_action_type or prev_action_type not in CONTACT_ACTIONS:
        return 0.0, "no previous contact action"

    this_channel = ACTION_CHANNEL_MAP.get(at, "")
    prev_channel = ACTION_CHANNEL_MAP.get(prev_action_type, "")

    if this_channel != prev_channel:
        return 0.0, "different channel — not a duplicate"

    this_goal = (action.goal or "").strip()
    if this_goal and prev_goal and this_goal != prev_goal:
        return 0.0, f"different goal ({prev_goal} → {this_goal}) — purposeful sequence"

    return -0.20, f"duplicate: {this_channel} with same goal '{this_goal}' used back-to-back"


# ── 8. Looping ────────────────────────────────────────────────────────────────

def reward_looping(step_count: int, max_steps: int) -> Tuple[float, str]:
    if step_count >= int(max_steps * 0.8):
        return -0.10, f"inefficient: {step_count}/{max_steps} steps used, not done"
    return 0.0, "step budget ok"


# ── Master calculator ─────────────────────────────────────────────────────────

def calculate_reward(
    action: Action,
    lead_before: LeadProfile,
    lead_after_stage: str,
    product: ProductProfile,
    step_count: int,
    max_steps: int,
    done: bool,
    prev_action_type: str = "",
    prev_goal: str = "",
) -> Tuple[float, Dict[str, Any]]:
    components: Dict[str, Tuple[float, str]] = {
        "channel":     reward_channel_choice(action, lead_before),
        "timing":      reward_timing(action, lead_before, prev_action_type),
        "message":     reward_message_quality(action, lead_before, product),
        "crm_update":  reward_crm_update(action),
        "progression": reward_lead_progression(action, lead_before, lead_after_stage),
        "compliance":  reward_policy_compliance(action, lead_before),
        "duplicate":   reward_duplicate_outreach(action, lead_before, prev_action_type, prev_goal),
    }

    if not done:
        components["looping"] = reward_looping(step_count, max_steps)

    total = sum(v for v, _ in components.values())
    total = round(max(-1.5, min(2.0, total)), 4)

    breakdown = {
        k: {"reward": round(v, 4), "reason": r}
        for k, (v, r) in components.items()
    }
    return total, breakdown
