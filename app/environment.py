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