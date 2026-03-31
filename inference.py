"""
Inference Script — E.L.A.R.A.
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import os
import sys
import json
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI
import requests

# ── Configuration ────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

MAX_STEPS_PER_TASK = {"easy": 3, "medium": 4, "hard": 6}
TEMPERATURE = 0.2
MAX_TOKENS = 400

DEBUG = True


# ── OpenAI client ────────────────────────────────────────────────────────────

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)


# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are an AI sales operations agent interacting with the E.L.A.R.A. environment.
You manage leads by choosing the right outreach actions.

You will receive an observation containing:
- Lead info (name, company, stage, sentiment, preferred channel, objections, docs pending)
- Product context (features, value props, pricing, compliance notes)
- Policy constraints (steps remaining, consent status)
- Task hint

You must respond with a JSON object representing your next action. The JSON must have these fields:
- "action_type": one of ["send_email", "make_call", "send_message", "request_documents", "update_crm", "schedule_followup", "run_campaign", "escalate", "wait"]
- "target_lead_id": the lead ID from the observation
- "subject": (optional) email subject line
- "body": message content — personalise with lead name, company, and product features
- "goal": what you're trying to achieve (e.g. "intro", "handle_objection_and_request_docs", "proposal_followup", "close_deal", "log_interaction")
- "priority": "low", "medium", or "high"
- "metadata": (optional) extra data like {"note": "..."} for CRM updates or {"docs_received": true}

Rules:
1. ALWAYS respect the lead's preferred_channel for contact actions.
2. NEVER contact a lead with consent=False.
3. NEVER use the same channel two steps in a row.
4. Personalise every message — mention the lead's name, company, and a relevant product feature.
5. If documents are pending, use "request_documents" at some point.
6. Always "update_crm" before the episode ends.
7. Read the task_hint carefully — it tells you exactly what to do.

Respond with ONLY the JSON object, no explanation or markdown.
""").strip()


# ── Environment API helpers ──────────────────────────────────────────────────

def env_reset(task_id: str) -> Dict[str, Any]:
    """POST /reset and return the observation dict."""
    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    return resp.json()["observation"]


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    """POST /step with the action and return the full response."""
    resp = requests.post(f"{ENV_URL}/step", json={"action": action})
    resp.raise_for_status()
    return resp.json()


def env_grader() -> Dict[str, Any]:
    """POST /grader and return the grading result."""
    resp = requests.post(f"{ENV_URL}/grader")
    resp.raise_for_status()
    return resp.json()


# ── LLM interaction ──────────────────────────────────────────────────────────

def build_user_message(observation: Dict[str, Any], step_num: int, max_steps: int) -> str:
    """Build a user message from the current observation."""
    return textwrap.dedent(f"""\
Step {step_num}/{max_steps}

Current observation:
{json.dumps(observation, indent=2, default=str)}

Based on this observation, choose your next action. Respond with a single JSON object.
""").strip()


def extract_action_json(text: str) -> Dict[str, Any]:
    """Extract a JSON action from the LLM response text."""
    text = text.strip()

    # Try to find JSON in code blocks
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}")


def get_llm_action(
    observation: Dict[str, Any],
    step_num: int,
    max_steps: int,
    history: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Call the LLM to get the next action."""
    user_msg = build_user_message(observation, step_num, max_steps)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_msg},
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    assistant_text = response.choices[0].message.content.strip()
    if DEBUG:
        print(f"  LLM response: {assistant_text[:200]}")

    action = extract_action_json(assistant_text)

    # Add to history for context
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_text})

    return action


# ── Fallback agent (no LLM needed) ──────────────────────────────────────────

def fallback_action(observation: Dict[str, Any], step_num: int, max_steps: int, actions_taken: List[str]) -> Dict[str, Any]:
    """
    Rule-based fallback agent when no LLM is available.
    Follows the task_hint and makes sensible decisions.
    """
    lead_id = observation["lead_id"]
    preferred = observation.get("preferred_channel", "email")
    lead_name = observation.get("lead_name", "there")
    company = observation.get("company", "your company")
    stage = observation.get("lead_stage", "new")
    docs_pending = observation.get("documents_pending", False)
    objections = observation.get("objections", [])
    task_id = observation.get("task_id", "easy")

    channel_map = {"email": "send_email", "call": "make_call", "message": "send_message"}
    preferred_action = channel_map.get(preferred, "send_email")

    # Avoid duplicate channels
    last_action = actions_taken[-1] if actions_taken else None
    last_channel = {
        "send_email": "email", "make_call": "call", "send_message": "message",
        "request_documents": "email",
    }.get(last_action, None)

    def alt_contact():
        """Pick a contact action that doesn't duplicate the last channel."""
        options = ["send_email", "make_call", "send_message"]
        for opt in [preferred_action] + options:
            ch = {"send_email": "email", "make_call": "call", "send_message": "message"}.get(opt)
            if ch != last_channel:
                return opt
        return "send_email"

    # Easy task logic
    if task_id == "easy":
        if step_num == 1:
            return {
                "action_type": preferred_action,
                "target_lead_id": lead_id,
                "subject": f"Intro — E.L.A.R.A. for {company}",
                "body": (
                    f"Hi {lead_name.split()[0]}, reaching out about E.L.A.R.A. — "
                    f"our platform cuts lead response time by 60% and unifies email, calls and messages. "
                    f"Given your role at {company}, this could save real hours weekly. "
                    f"Happy to set up a quick demo — does this week work?"
                ),
                "goal": "intro",
                "priority": "high",
            }
        return {"action_type": "wait", "target_lead_id": lead_id, "goal": "done"}

    # Medium task logic
    if task_id == "medium":
        if step_num == 1:
            objection_text = objections[0] if objections else "concerns"
            return {
                "action_type": alt_contact(),
                "target_lead_id": lead_id,
                "subject": f"RE: {objection_text.title()} — next steps for {company}",
                "body": (
                    f"Hi {lead_name.split()[0]}, thanks for raising the {objection_text} concern. "
                    f"Our ROI calculator shows teams like {company} typically save 8h/week per rep. "
                    f"At our ₹2,999/seat plan that's a 10x return in 3 months. "
                    f"Happy to walk through the numbers on a call."
                ),
                "goal": "handle_objection_and_request_docs",
                "priority": "high",
            }
        elif "request_documents" not in actions_taken:
            return {
                "action_type": "request_documents",
                "target_lead_id": lead_id,
                "subject": f"Documents for {company} proposal",
                "body": (
                    f"Hi {lead_name.split()[0]}, could you share the signed NDA and tooling overview? "
                    f"Once I have those I can get the proposal ready within 24 hours."
                ),
                "goal": "get_documents",
            }
        elif "update_crm" not in actions_taken:
            return {
                "action_type": "update_crm",
                "target_lead_id": lead_id,
                "goal": "log_interaction",
                "metadata": {"note": f"Handled {objections[0] if objections else 'objection'}. Doc request sent. Lead in awaiting_docs."},
            }
        return {"action_type": "wait", "target_lead_id": lead_id, "goal": "done"}

    # Hard task logic
    if task_id == "hard":
        if step_num == 1:
            return {
                "action_type": "make_call" if last_channel != "call" else alt_contact(),
                "target_lead_id": lead_id,
                "body": (
                    f"Hi {lead_name.split()[0]}, following up on the proposal I sent last week. "
                    f"Wanted to check if {company} had a chance to review it. "
                    f"We're excited about the potential fit."
                ),
                "goal": "proposal_followup",
                "priority": "high",
            }
        elif "request_documents" not in actions_taken:
            return {
                "action_type": "request_documents",
                "target_lead_id": lead_id,
                "subject": f"Documents needed — {company}",
                "body": (
                    f"Hi {lead_name.split()[0]}, great speaking just now. "
                    f"To finalise the agreement I'll need the signed NDA and procurement form. "
                    f"I can have the final contract ready within 24 hours of receiving them."
                ),
                "goal": "get_documents",
            }
        elif "send_message" not in actions_taken:
            return {
                "action_type": "send_message",
                "target_lead_id": lead_id,
                "body": (
                    f"Hi {lead_name.split()[0]} — just sent the docs request via email. "
                    f"Let me know if you need any changes to the proposal terms."
                ),
                "goal": "keep_warm",
            }
        elif "update_crm" not in actions_taken:
            return {
                "action_type": "update_crm",
                "target_lead_id": lead_id,
                "goal": "log_interaction",
                "metadata": {"note": "Follow-up call done. Doc request emailed. Message sent. Targeting close."},
            }
        elif "close_deal" not in [a for a in actions_taken]:
            return {
                "action_type": "make_call" if last_channel != "call" else alt_contact(),
                "target_lead_id": lead_id,
                "body": (
                    f"Hi {lead_name.split()[0]}, received your documents — everything looks good. "
                    f"Ready to move forward with {company}?"
                ),
                "goal": "close_deal",
                "metadata": {"docs_received": True},
            }
        return {"action_type": "wait", "target_lead_id": lead_id, "goal": "done"}

    return {"action_type": "wait", "target_lead_id": lead_id}


# ── Main runner ──────────────────────────────────────────────────────────────

def run_task(task_id: str, use_llm: bool = True) -> Dict[str, Any]:
    """Run one task end-to-end. Returns the grader result."""
    max_steps = MAX_STEPS_PER_TASK.get(task_id, 5)

    print(f"\n{'='*60}")
    print(f"  Task: {task_id.upper()} (max {max_steps} steps)")
    print(f"{'='*60}")

    # Reset
    observation = env_reset(task_id)
    if DEBUG:
        print(f"  Lead: {observation['lead_name']} ({observation['lead_id']})")
        print(f"  Stage: {observation['lead_stage']} | Channel: {observation['preferred_channel']}")
        print(f"  Hint: {observation.get('task_hint', 'N/A')}")

    history: List[Dict[str, str]] = []
    actions_taken: List[str] = []

    for step_num in range(1, max_steps + 1):
        print(f"\n  --- Step {step_num}/{max_steps} ---")

        try:
            if use_llm:
                action = get_llm_action(observation, step_num, max_steps, history)
            else:
                action = fallback_action(observation, step_num, max_steps, actions_taken)
        except Exception as e:
            print(f"  ⚠ LLM failed ({e}), using fallback agent")
            action = fallback_action(observation, step_num, max_steps, actions_taken)

        if DEBUG:
            print(f"  Action: {action.get('action_type')} → {action.get('target_lead_id')}")

        actions_taken.append(action.get("action_type", "wait"))

        # Step
        result = env_step(action)
        observation = result["observation"]
        reward = result["reward"]
        done = result["done"]

        print(f"  Reward: {reward:+.3f} | Done: {done} | Stage: {observation['lead_stage']}")

        if done:
            break

    # Grade
    grade_result = env_grader()
    score = grade_result["score"]
    passed = grade_result["pass"]
    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'} — Score: {score:.4f}")
    if DEBUG and "dimensions" in grade_result:
        for dim, data in grade_result["dimensions"].items():
            print(f"    {dim:<24} {data['score']:.2f} × {data['weight']:.0%} = {data['score']*data['weight']:.4f}")

    return grade_result


def main():
    """Run inference on all 3 tasks."""
    use_llm = bool(API_BASE_URL and API_KEY and MODEL_NAME)

    if use_llm:
        print(f"Using LLM: {MODEL_NAME} @ {API_BASE_URL}")
    else:
        print("⚠ No LLM configured (API_BASE_URL / MODEL_NAME / HF_TOKEN missing)")
        print("  Using rule-based fallback agent for reproducible baseline scores")

    results = {}
    for task_id in ["easy", "medium", "hard"]:
        results[task_id] = run_task(task_id, use_llm=use_llm)

    # Summary
    print(f"\n{'='*60}")
    print(f"  INFERENCE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Task':<10} {'Score':>7}  {'Pass':>5}")
    print(f"  {'-'*30}")
    for task_id, r in results.items():
        pf = "✓" if r["pass"] else "✗"
        print(f"  {task_id:<10} {r['score']:>7.4f}  {pf:>5}")
    print()


if __name__ == "__main__":
    main()
