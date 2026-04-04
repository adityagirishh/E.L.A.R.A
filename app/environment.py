"""
environment.py  —  ElaraEnv

API (OpenEnv-compatible):
    obs                     = env.reset(task_id="easy")
    obs, reward, done, info = env.step(action)
    state_dict              = env.state()

v3 features:
  - Seeded RNG for stochastic variation
  - Dynamic lead responses after contact actions
  - Multi-lead task support (active_leads in observation)
  - Dynamic events: consent revocation, sentiment shifts
"""

import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from models import (
    Action, EpisodeState, LeadProfile, LeadSummary,
    Observation, PolicyConstraints, ProductProfile,
)
from crm_simulator import (
    SEED_LEADS, apply_action, generate_lead_response,
    apply_response_mutations, CONTACT_ACTIONS,
)
from reward_engine import calculate_reward

TASKS_DIR = Path(__file__).parent.parent / "tasks"

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
        "request_documents always goes via email — exempt from channel preference.",
        "Do not contact same lead twice on same channel with the same goal.",
        "Escalate if documents not received after 2 requests.",
        "If a lead revokes consent mid-conversation, STOP all contact immediately.",
    ],
    value_props=[
        "Cut lead response time by 60%.",
        "Unified inbox across email, calls, and messages.",
        "Auto-CRM updates after every interaction.",
        "Compliance-safe outreach templates.",
    ],
    faqs=[
        {"q": "What CRMs does it integrate with?", "a": "Salesforce, HubSpot, Pipedrive."},
        {"q": "Is there a free trial?", "a": "Yes — 14 days, no credit card required."},
        {"q": "How is pricing structured?", "a": "Per seat, ₹2,999/month. Volume discounts available."},
    ],
)


class ElaraEnv:

    AVAILABLE_TASKS = ["easy", "medium", "hard", "escalation", "consent"]

    def __init__(self):
        self._state: Optional[EpisodeState] = None
        self._rng: random.Random = random.Random()
        self._task_cfg: Dict[str, Any] = {}
        self._dynamic_events: Dict[str, Dict[str, Any]] = {}

    # ── reset ──────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "easy", seed: Optional[int] = None) -> Observation:
        task_cfg  = self._load_task(task_id)
        self._task_cfg = task_cfg

        # Seed the RNG
        self._rng = random.Random(seed)

        # Support single lead_id or multi lead_ids
        lead_ids = task_cfg.get("lead_ids", [task_cfg["lead_id"]])
        primary_lead_id = lead_ids[0]
        max_steps = task_cfg.get("max_steps", 5)

        leads = {k: deepcopy(v) for k, v in SEED_LEADS.items()}
        for lid in lead_ids:
            if lid not in leads:
                raise ValueError(f"Lead {lid} not found.")

        # Apply seed-based variations to active leads
        if seed is not None:
            self._apply_seed_variations(leads, lead_ids)

        # Load dynamic events from task config
        self._dynamic_events = task_cfg.get("dynamic_events", {})

        self._state = EpisodeState(
            product=deepcopy(SEED_PRODUCT),
            leads=leads,
            current_lead_id=primary_lead_id,
            active_lead_ids=lead_ids,
            task_id=task_id,
            step_count=0,
            max_steps=max_steps,
            done=False,
            total_reward=0.0,
            episode_log=[],
            lead_responses=[],
            seed=seed,
        )

        return self._build_observation(task_cfg.get("hint", ""))

    # ── step ───────────────────────────────────────────────────────────────

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            return self._build_observation(), 0.0, True, {"message": "Episode done. Call reset()."}

        s = self._state

        if action.target_lead_id not in s.leads:
            s.step_count += 1
            s.done = True
            return self._build_observation(), -0.30, True, {"error": f"Unknown lead: {action.target_lead_id}"}

        # Update current_lead_id to the targeted lead
        s.current_lead_id = action.target_lead_id
        lead_before = deepcopy(s.leads[action.target_lead_id])

        # Previous action context
        prev_action_type, prev_goal = "", ""
        if s.episode_log:
            last = s.episode_log[-1]
            prev_action_type = last.get("action_type", "")
            prev_goal        = last.get("goal", "")

        # Apply action → get updated lead + new stage
        updated_lead, new_stage = apply_action(s.leads[action.target_lead_id], action)
        s.leads[action.target_lead_id] = updated_lead
        s.step_count += 1

        done_after = s.step_count >= s.max_steps

        # Calculate reward
        reward, breakdown = calculate_reward(
            action=action,
            lead_before=lead_before,
            lead_after_stage=new_stage,
            product=s.product,
            step_count=s.step_count,
            max_steps=s.max_steps,
            done=done_after,
            prev_action_type=prev_action_type,
            prev_goal=prev_goal,
        )

        s.total_reward = round(s.total_reward + reward, 4)
        if done_after:
            s.done = True

        # Generate dynamic lead response
        lead_response = None
        if action.action_type in CONTACT_ACTIONS:
            dynamic_events_for_lead = self._dynamic_events.get(action.target_lead_id, {})
            # Check if this is a "trigger after first contact" event
            contact_count = sum(
                1 for e in s.episode_log
                if e.get("target_lead") == action.target_lead_id
                and e.get("action_type") in CONTACT_ACTIONS
            )
            # Only trigger consent revocation after the configured contact number
            trigger_after = dynamic_events_for_lead.get("trigger_after_contact", 1)
            if dynamic_events_for_lead.get("revoke_consent_after_contact") and contact_count + 1 >= trigger_after:
                events_to_pass = dynamic_events_for_lead
            else:
                events_to_pass = {}

            lead_response = generate_lead_response(
                lead=s.leads[action.target_lead_id],
                action=action,
                product=s.product,
                rng=self._rng,
                dynamic_events=events_to_pass,
            )

        # Apply response mutations
        if lead_response:
            mutations = lead_response.get("mutations", {})
            if mutations:
                s.leads[action.target_lead_id] = apply_response_mutations(
                    s.leads[action.target_lead_id], mutations
                )
            s.lead_responses.append(lead_response)

            # Add response to lead's conversation history
            s.leads[action.target_lead_id].conversation_history.append({
                "from": "lead",
                "channel": lead_response.get("channel", "reply"),
                "text": lead_response.get("text", ""),
                "response_type": lead_response.get("response_type", ""),
            })

        # Log
        log_entry = {
            "step":         s.step_count,
            "action_type":  action.action_type,
            "target_lead":  action.target_lead_id,
            "goal":         action.goal or "",
            "stage_before": lead_before.lead_stage,
            "stage_after":  new_stage,
            "reward":       reward,
            "breakdown":    breakdown,
            "done":         s.done,
        }
        if lead_response:
            log_entry["lead_response"] = lead_response.get("text", "")
        s.episode_log.append(log_entry)

        info = {
            "step":             s.step_count,
            "stage_before":     lead_before.lead_stage,
            "stage_after":      new_stage,
            "reward_breakdown": breakdown,
            "total_reward":     s.total_reward,
            "done":             s.done,
        }
        if lead_response:
            info["lead_response"] = lead_response

        return self._build_observation(), reward, s.done, info

    # ── state ──────────────────────────────────────────────────────────────

    def state(self) -> Dict[str, Any]:
        if self._state is None:
            raise RuntimeError("Call reset() before state().")
        return self._state.model_dump()

    # ── helpers ────────────────────────────────────────────────────────────

    def _build_observation(self, hint: str = "") -> Observation:
        s    = self._state
        lead = s.leads[s.current_lead_id]
        prod = s.product

        policy = PolicyConstraints(
            consent_required=True,
            min_days_since_contact=0,
            max_steps_remaining=s.max_steps - s.step_count,
            allow_campaign=(lead.lead_stage in {"contacted", "qualified"}),
        )

        # Build active lead summaries for multi-lead tasks
        active_leads = []
        for lid in s.active_lead_ids:
            al = s.leads[lid]
            active_leads.append(LeadSummary(
                lead_id=al.lead_id,
                lead_name=al.lead_name,
                company=al.company,
                lead_stage=al.lead_stage,
                sentiment=al.sentiment,
                preferred_channel=al.preferred_channel,
                documents_pending=al.documents_pending,
                consent=al.consent,
                days_since_last_contact=al.days_since_last_contact,
                objections=al.objections,
            ))

        # Collect recent lead responses (last 5)
        recent_responses = s.lead_responses[-5:]

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
            active_leads=active_leads,
            lead_responses=[
                {"from": r.get("from", ""), "text": r.get("text", ""), "lead_id": r.get("lead_id", "")}
                for r in recent_responses
            ],
            product_context={
                "name":              prod.product_name,
                "description":       prod.description,
                "features":          prod.features,
                "value_props":       prod.value_props,
                "pricing_summary":   prod.pricing_summary,
                "objection_playbook":prod.objections,
                "compliance_notes":  prod.compliance_notes,
                "faqs":              prod.faqs,
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

    def _apply_seed_variations(
        self, leads: Dict[str, LeadProfile], active_ids: List[str]
    ) -> None:
        """Apply seeded random variations to active leads."""
        rng = self._rng
        sentiment_options = ["cold", "neutral", "warm", "hot"]
        extra_objections = [
            "pricing", "not the right time", "need to think about it",
            "already using a competitor",
        ]

        for lid in active_ids:
            lead = leads[lid]
            # Small chance of sentiment shift
            if rng.random() < 0.3:
                idx = sentiment_options.index(lead.sentiment) if lead.sentiment in sentiment_options else 1
                delta = rng.choice([-1, 1])
                new_idx = max(0, min(len(sentiment_options) - 1, idx + delta))
                lead.sentiment = sentiment_options[new_idx]

            # Small chance of extra objection — but NOT at stages where objections block progression
            blocking_stages = {"contacted", "proposal_sent"}
            if rng.random() < 0.2 and len(lead.objections) < 2 and lead.lead_stage not in blocking_stages:
                candidates = [o for o in extra_objections if o not in lead.objections]
                if candidates:
                    lead.objections.append(rng.choice(candidates))

            # Small chance of followup timing variation
            if rng.random() < 0.25 and lead.next_followup_due > 0:
                lead.next_followup_due += rng.choice([-1, 0, 1])
                lead.next_followup_due = max(1, lead.next_followup_due)

    @staticmethod
    def _load_task(task_id: str) -> Dict[str, Any]:
        path = TASKS_DIR / f"{task_id}.json"
        if not path.exists():
            raise ValueError(f"Task '{task_id}' not found at {path}")
        return json.loads(path.read_text())
