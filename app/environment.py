"""
environment.py
ElaraEnv — the main OpenEnv-compatible environment class.

API:
    env = ElaraEnv()
    obs           = env.reset(task_id="easy")
    obs, r, done, info = env.step(action)
    state_dict    = env.state()
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from models import (
    Action, EpisodeState, LeadProfile, Observation,
    PolicyConstraints, ProductProfile,
)
from crm_simulator import SEED_LEADS, apply_action
from reward_engine import calculate_reward

TASKS_DIR = Path(__file__).parent.parent / "tasks"


# ─────────────────────────────────────────────
# Seed product
# ─────────────────────────────────────────────

SEED_PRODUCT = ProductProfile(
    product_id="P-ELARA-001",
    product_name="E.L.A.R.A.",
    description=(
        "E.L.A.R.A. is a sandboxed multi-channel sales operations platform. "
        "AI agents practice qualifying leads, choosing outreach channels, "
        "handling objections, and managing CRM updates in a safe simulated setting."
    ),
    features=[
        "multi-channel outreach (email / call / message)",
        "lead stage tracking and CRM updates",
        "document request workflows",
        "objection handling playbooks",
        "compliance guardrails",
        "campaign automation",
    ],
    target_users=["sales teams", "revops teams", "founders", "AI sales agents"],
    pricing_summary="Starts at ₹2,999/seat/month. Enterprise plans available. 14-day free trial.",
    objections=[
        "too expensive — offer ROI data and trial",
        "already using a competitor — highlight differentiators",
        "not the right time — understand timeline, schedule follow-up",
        "need to see documents first — send NDA + pricing sheet",
    ],
    compliance_notes=[
        "Never contact leads with consent=False.",
        "Respect preferred_channel when set.",
        "Do not contact same lead twice on same channel same day.",
        "Escalate if documents not received after 2 requests.",
    ],
    value_props=[
        "Cut lead response time by 60%.",
        "Unified inbox across email, calls, and messages.",
        "Auto-CRM updates after every interaction.",
        "Compliance-safe outreach templates.",
    ],
    faqs=[
        {"q": "What CRMs does it integrate with?", "a": "Salesforce, HubSpot, Pipedrive out of the box."},
        {"q": "Is there a free trial?", "a": "Yes — 14 days, no credit card required."},
        {"q": "How is pricing structured?", "a": "Per seat, starting at ₹2,999/month. Volume discounts available."},
    ],
)


# ─────────────────────────────────────────────
# ElaraEnv
# ─────────────────────────────────────────────

class ElaraEnv:

    AVAILABLE_TASKS = ["easy", "medium", "hard"]

    def __init__(self):
        self._state: Optional[EpisodeState] = None

    # ── reset ─────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "easy", seed: Optional[int] = None) -> Observation:
        """Start a fresh episode for the given task. Returns initial Observation."""

        task_cfg = self._load_task(task_id)
        lead_id  = task_cfg["lead_id"]
        max_steps = task_cfg.get("max_steps", 5)

        leads = {k: deepcopy(v) for k, v in SEED_LEADS.items()}
        if lead_id not in leads:
            raise ValueError(f"Lead {lead_id} not found in CRM.")

        self._state = EpisodeState(
            product=deepcopy(SEED_PRODUCT),
            leads=leads,
            current_lead_id=lead_id,
            task_id=task_id,
            step_count=0,
            max_steps=max_steps,
            done=False,
            total_reward=0.0,
            episode_log=[],
            seed=seed,
        )

        return self._build_observation(task_cfg.get("hint", ""))

    # ── step ──────────────────────────────────────────────────────────────

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Take one action. Returns (observation, reward, done, info).
        This is the canonical OpenEnv tuple shape.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            obs = self._build_observation()
            return obs, 0.0, True, {"message": "Episode already done. Call reset()."}

        s = self._state
        lead_before = deepcopy(s.leads[s.current_lead_id])

        # Wrong lead ID
        if action.target_lead_id not in s.leads:
            reward = -0.3
            info = {"error": f"Unknown lead id: {action.target_lead_id}", "reward_breakdown": {}}
            s.step_count += 1
            s.total_reward += reward
            s.done = True
            return self._build_observation(), reward, True, info

        # Apply action → get updated lead + new stage
        updated_lead, new_stage = apply_action(s.leads[action.target_lead_id], action)
        s.leads[action.target_lead_id] = updated_lead

        # Calculate reward
        s.step_count += 1
        done_after = s.step_count >= s.max_steps
        reward, breakdown = calculate_reward(
            action=action,
            lead_before=lead_before,
            lead_after_stage=new_stage,
            product=s.product,
            step_count=s.step_count,
            max_steps=s.max_steps,
            done=done_after,
        )

        s.total_reward = round(s.total_reward + reward, 4)

        # Episode end?
        if done_after:
            s.done = True

        # Log
        log_entry = {
            "step": s.step_count,
            "action_type": action.action_type,
            "target_lead": action.target_lead_id,
            "goal": action.goal,
            "stage_before": lead_before.lead_stage,
            "stage_after": new_stage,
            "reward": reward,
            "breakdown": breakdown,
            "done": s.done,
        }
        s.episode_log.append(log_entry)

        info = {
            "step": s.step_count,
            "stage_before": lead_before.lead_stage,
            "stage_after": new_stage,
            "reward_breakdown": breakdown,
            "total_reward": s.total_reward,
            "done": s.done,
        }

        obs = self._build_observation()
        return obs, reward, s.done, info

    # ── state ─────────────────────────────────────────────────────────────

    def state(self) -> Dict[str, Any]:
        """Return full episode state as a dict."""
        if self._state is None:
            raise RuntimeError("Call reset() before state().")
        return self._state.model_dump()

    # ── helpers ───────────────────────────────────────────────────────────

    def _build_observation(self, hint: str = "") -> Observation:
        s = self._state
        lead = s.leads[s.current_lead_id]
        product = s.product

        policy = PolicyConstraints(
            consent_required=True,
            min_days_since_contact=0,
            max_steps_remaining=s.max_steps - s.step_count,
            allow_campaign=(lead.lead_stage in {"contacted", "qualified"}),
        )

        return Observation(
            task_id=s.task_id,
            step_count=s.step_count,
            lead_id=lead.lead_id,
            lead_name=lead.lead_name,
            company=lead.company,
            role=lead.role,
            lead_stage=lead.lead_stage,
            last_contact_channel=lead.last_contact_channel,
            days_since_last_contact=lead.days_since_last_contact,
            next_followup_due=lead.next_followup_due,
            consent=lead.consent,
            sentiment=lead.sentiment,
            documents_pending=lead.documents_pending,
            preferred_channel=lead.preferred_channel,
            objections=lead.objections,
            product_context={
                "name": product.product_name,
                "description": product.description,
                "features": product.features,
                "value_props": product.value_props,
                "pricing_summary": product.pricing_summary,
                "objection_playbook": product.objections,
                "compliance_notes": product.compliance_notes,
                "faqs": product.faqs,
            },
            recent_history=lead.conversation_history[-5:],
            policy_constraints=policy,
            available_actions=[
                "send_email", "make_call", "send_message",
                "request_documents", "update_crm",
                "schedule_followup", "run_campaign", "escalate", "wait",
            ],
            task_hint=hint or self._task_hint(),
        )

    def _task_hint(self) -> str:
        if self._state is None:
            return ""
        try:
            cfg = self._load_task(self._state.task_id)
            return cfg.get("hint", "")
        except Exception:
            return ""

    @staticmethod
    def _load_task(task_id: str) -> Dict[str, Any]:
        path = TASKS_DIR / f"{task_id}.json"
        if not path.exists():
            raise ValueError(f"Task '{task_id}' not found at {path}")
        return json.loads(path.read_text())
