<<<<<<< HEAD
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
=======
from copy import deepcopy
from typing import Any, Dict, Tuple

from models import Action, LeadProfile, Observation, ProductProfile, State


class ElaraEnv:
    def __init__(self):
        self._seed_product = ProductProfile(
            product_id="P-ELARA-001",
            product_name="E.L.A.R.A.",
            website_url="local://hardcoded",
            description="A sandboxed multi-channel sales operations agent.",
            features=[
                "lead qualification",
                "multi-channel outreach",
                "CRM updates",
                "follow-up workflows",
            ],
            target_users=["sales teams", "revops teams", "founders"],
            pricing_summary="Sandbox demo product with enterprise-style workflows.",
            objections=["too expensive", "need documents", "not now"],
            compliance_notes=[
                "Respect consent",
                "Avoid duplicate outreach",
                "Use the correct channel",
            ],
        )

        self._seed_leads = {
            "L-001": LeadProfile(
                lead_id="L-001",
                lead_name="Arun",
                lead_stage="new",
                last_contact_channel="none",
                days_since_last_contact=0,
                consent=True,
                sentiment="neutral",
                documents_pending=True,
            ),
            "L-002": LeadProfile(
                lead_id="L-002",
                lead_name="Meera",
                lead_stage="contacted",
                last_contact_channel="email",
                days_since_last_contact=3,
                consent=True,
                sentiment="warm",
                documents_pending=False,
                objections=["pricing"],
            ),
        }

        self.state_obj: State | None = None

    def reset(self) -> Observation:
        self.state_obj = State(
            product=deepcopy(self._seed_product),
            leads={k: deepcopy(v) for k, v in self._seed_leads.items()},
            current_lead_id="L-001",
            task_id="easy",
            step_count=0,
            done=False,
        )
        return self._get_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.state_obj is None:
            raise RuntimeError("Call reset() before step().")

        if self.state_obj.done:
            return self._get_observation(), 0.0, True, {"message": "Episode already done"}

        if action.target_lead_id not in self.state_obj.leads:
            self.state_obj.done = True
            return self._get_observation(), -1.0, True, {"error": "Unknown lead id"}

        lead = self.state_obj.leads[action.target_lead_id]
        reward = 0.0
        info: Dict[str, Any] = {"action_type": action.action_type}

        # Base action validation
        if action.action_type not in {"email", "call", "message"}:
            reward -= 0.5
            info["reason"] = "invalid_action"
        else:
            # Record interaction
            lead.conversation_history.append(
                {
                    "step": self.state_obj.step_count + 1,
                    "channel": action.action_type,
                    "content": action.content,
                    "subject": action.subject,
                    "metadata": action.metadata,
                }
            )

            # Small positive reward for any valid interaction
            reward += 0.2

            # Channel-specific shaping
            if action.action_type == "email":
                reward += 0.1
                lead.last_contact_channel = "email"
                if lead.documents_pending:
                    reward += 0.4
                    lead.documents_pending = False
                    lead.notes.append("Requested documents via email")
                    lead.lead_stage = "awaiting_docs"
            elif action.action_type == "call":
                reward += 0.1
                lead.last_contact_channel = "call"
                lead.lead_stage = "contacted"
                if lead.days_since_last_contact >= 3:
                    reward += 0.3
                    lead.notes.append("Appropriate follow-up call after delay")
                else:
                    reward -= 0.1
            elif action.action_type == "message":
                reward += 0.1
                lead.last_contact_channel = "message"
                lead.lead_stage = "contacted"

            # CRM-style state update
            if action.content.strip():
                lead.notes.append(f"Action logged: {action.action_type}")
                reward += 0.1

            # Reduce waiting time after contact
            lead.days_since_last_contact = 0

        self.state_obj.episode_log.append(
            {
                "step": self.state_obj.step_count + 1,
                "lead_id": action.target_lead_id,
                "action": action.model_dump(),
                "reward": reward,
            }
        )

        self.state_obj.step_count += 1

        # Simple episode end for Day 1
        if self.state_obj.step_count >= 3:
            self.state_obj.done = True
            info["episode_complete"] = True

        return self._get_observation(), reward, self.state_obj.done, info

    def state(self) -> Dict[str, Any]:
        if self.state_obj is None:
            raise RuntimeError("Call reset() before state().")

        return self.state_obj.model_dump()

    def _get_observation(self) -> Observation:
        if self.state_obj is None:
            raise RuntimeError("Call reset() before observation.")

        lead = self.state_obj.leads[self.state_obj.current_lead_id]
        product = self.state_obj.product

        return Observation(
            task_id=self.state_obj.task_id,
            product_name=product.product_name,
            lead_id=lead.lead_id,
            lead_name=lead.lead_name,
            lead_stage=lead.lead_stage,
            last_contact_channel=lead.last_contact_channel,
            days_since_last_contact=lead.days_since_last_contact,
            consent=lead.consent,
            sentiment=lead.sentiment,
            documents_pending=lead.documents_pending,
            product_context={
                "description": product.description,
                "features": product.features,
                "target_users": product.target_users,
                "pricing_summary": product.pricing_summary,
                "objections": product.objections,
                "compliance_notes": product.compliance_notes,
            },
            recent_history=lead.conversation_history[-5:],
            available_actions=["email", "call", "message"],
        )
>>>>>>> 8cffedc4c8dc68dd752a6a32295fc5d3f86bccf5
