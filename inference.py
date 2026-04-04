"""
Inference Script — E.L.A.R.A.
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
    [START] task=<task_name> env=elara model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import os
import json
import asyncio
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from elara_env import ElaraEnv, ElaraAction


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "nvidia/nemotron-3-super-120b-a12b:free")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
IMAGE_NAME   = os.getenv("IMAGE_NAME", "elara_env:latest")

MAX_STEPS_PER_TASK = {"easy": 3, "medium": 4, "hard": 7, "escalation": 4, "consent": 6}
TEMPERATURE  = 0
MAX_TOKENS   = 500
DEBUG        = True

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


# ─────────────────────────────────────────────
# STDOUT LOGGERS
# ─────────────────────────────────────────────

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=elara model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are an AI sales operations agent interacting with the E.L.A.R.A. environment.
You manage leads by choosing the right outreach actions.

You will receive an observation containing:
- Lead info (name, company, stage, sentiment, preferred channel, objections, docs pending)
- active_leads: summaries of ALL leads you need to manage (may be more than one)
- lead_responses: recent replies FROM leads — READ THESE CAREFULLY, they affect what you should do
- Product context (features, value props, pricing, compliance notes)
- Policy constraints (steps remaining, consent status)
- Task hint

You must respond with a JSON object representing your next action:
- "action_type": one of ["send_email", "make_call", "send_message", "request_documents", "update_crm", "schedule_followup", "run_campaign", "escalate", "wait"]
- "target_lead_id": the lead ID to act on (check active_leads for all available leads)
- "subject": (optional) email subject line
- "body": message content — personalise with lead name, company, and product features
- "goal": what you're trying to achieve
- "priority": "low", "medium", or "high"
- "metadata": (optional) extra data like {"note": "..."} for CRM updates

Rules:
1. ALWAYS respect the lead's preferred_channel for contact actions.
2. NEVER contact a lead with consent=False — check EVERY step, consent can change mid-episode.
3. NEVER use the same channel two steps in a row for the same goal on the same lead.
4. Personalise every message — mention lead name, company, and a relevant product feature.
5. If a lead has objections, ADDRESS them with specific solutions (ROI data for pricing, timeline for timing).
6. READ lead_responses — if a lead asks to opt out, STOP contacting them immediately.
7. If documents are stuck after 2+ requests, ESCALATE instead of requesting again.
8. Always "update_crm" before the episode ends.
9. For multi-lead tasks, manage ALL active leads, not just the first one.
10. Read the task_hint carefully — it tells you exactly what to do.

CRITICAL: Your ENTIRE response must be a single JSON object. No thinking, no explanation, no markdown, no preamble. Start your response with { and end with }. Nothing else.
""").strip()


# ─────────────────────────────────────────────
# LLM INTERACTION
# ─────────────────────────────────────────────

def build_user_message(observation: Dict[str, Any], step_num: int, max_steps: int) -> str:
    return textwrap.dedent(f"""\
Step {step_num}/{max_steps}

Current observation:
{json.dumps(observation, indent=2, default=str)}

Based on this observation, choose your next action. Respond with a single JSON object.
""").strip()


def extract_action_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

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
        print(f"  LLM response: {assistant_text[:200]}", flush=True)

    action = extract_action_json(assistant_text)

    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_text})

    return action


# ─────────────────────────────────────────────
# FALLBACK AGENT (rule-based, no LLM needed)
# ─────────────────────────────────────────────

def fallback_action(
    observation: Dict[str, Any],
    step_num: int,
    actions_taken: List[Dict[str, str]],
) -> Dict[str, Any]:
    lead_id       = observation["lead_id"]
    preferred     = observation.get("preferred_channel", "email")
    lead_name     = observation.get("lead_name", "there")
    company       = observation.get("company", "your company")
    objections    = observation.get("objections", [])
    task_id       = observation.get("task_id", "easy")
    active_leads  = observation.get("active_leads", [])
    lead_responses = observation.get("lead_responses", [])

    channel_map      = {"email": "send_email", "call": "make_call", "message": "send_message"}
    preferred_action = channel_map.get(preferred, "send_email")

    def actions_for_lead(lid):
        return [a for a in actions_taken if a.get("target") == lid]

    def action_types_for_lead(lid):
        return [a["type"] for a in actions_for_lead(lid)]

    revoked_leads = set()
    for resp in lead_responses:
        text = resp.get("text", "").lower()
        if "remove" in text or "opt out" in text or "stop contact" in text or "stop contacting" in text:
            revoked_leads.add(resp.get("lead_id", ""))

    last = actions_taken[-1] if actions_taken else {}
    last_channel = {
        "send_email": "email", "make_call": "call", "send_message": "message",
        "request_documents": "email",
    }.get(last.get("type", ""), None)

    def alt_contact(pref_action, avoid_channel=None):
        options = ["send_email", "make_call", "send_message"]
        ch_map  = {"send_email": "email", "make_call": "call", "send_message": "message"}
        for opt in [pref_action] + options:
            if ch_map.get(opt) != avoid_channel:
                return opt
        return "send_email"

    def make_intro(lid, name, comp, pref_act):
        return {
            "action_type":    pref_act,
            "target_lead_id": lid,
            "subject":        f"Intro — E.L.A.R.A. for {comp}",
            "body": (
                f"Hi {name.split()[0]}, reaching out about E.L.A.R.A. — "
                f"our platform cuts lead response time by 60% and unifies email, calls and messages. "
                f"Given your role at {comp}, this could save real hours weekly. "
                f"Happy to set up a quick demo."
            ),
            "goal":     "intro",
            "priority": "high",
        }

    # ── EASY ──
    if task_id == "easy":
        if step_num == 1:
            return make_intro(lead_id, lead_name, company, preferred_action)
        return {"action_type": "wait", "target_lead_id": lead_id, "goal": "done"}

    # ── MEDIUM ──
    if task_id == "medium":
        my_actions = action_types_for_lead(lead_id)
        if step_num == 1:
            objection_text = objections[0] if objections else "concerns"
            return {
                "action_type":    alt_contact(preferred_action, last_channel),
                "target_lead_id": lead_id,
                "subject":        f"RE: {objection_text.title()} — next steps for {company}",
                "body": (
                    f"Hi {lead_name.split()[0]}, thanks for raising the {objection_text} concern. "
                    f"Our ROI calculator shows teams like {company} typically save 8h/week per rep. "
                    f"At our ₹2,999/seat plan that's a 10x return in 3 months."
                ),
                "goal":     "handle_objection",
                "priority": "high",
            }
        elif "request_documents" not in my_actions:
            return {
                "action_type":    "request_documents",
                "target_lead_id": lead_id,
                "subject":        f"Documents for {company} proposal",
                "body":           f"Hi {lead_name.split()[0]}, could you share the signed NDA and tooling overview?",
                "goal":           "get_documents",
            }
        elif "update_crm" not in my_actions:
            return {
                "action_type":    "update_crm",
                "target_lead_id": lead_id,
                "goal":           "log_interaction",
                "metadata":       {"note": f"Handled {objections[0] if objections else 'objection'}. Doc request sent."},
            }
        return {"action_type": "wait", "target_lead_id": lead_id, "goal": "done"}

    # ── HARD ──
    if task_id == "hard":
        l7_actions = action_types_for_lead("L-007")
        l8_actions = action_types_for_lead("L-008")

        if step_num == 1:
            return {
                "action_type":    "make_call",
                "target_lead_id": "L-007",
                "body":           "Hi Rajan, following up on the proposal for Apex Ventures. E.L.A.R.A. cuts lead response time by 60%.",
                "goal":           "proposal_followup",
                "priority":       "high",
            }
        elif "send_message" not in l8_actions:
            return {
                "action_type":    "send_message",
                "target_lead_id": "L-008",
                "body":           "Hi Sunita, regarding BlueSky Tech contract terms — we can offer flexible terms and adjust the clause.",
                "goal":           "handle_objection",
            }
        elif "request_documents" not in l7_actions:
            return {
                "action_type":    "request_documents",
                "target_lead_id": "L-007",
                "body":           "Hi Rajan, to finalise I need the signed NDA and procurement form.",
                "goal":           "get_documents",
            }
        elif "send_message" not in l8_actions:
            return {
                "action_type":    "send_message",
                "target_lead_id": "L-008",
                "body":           "Hi Sunita, revised contract terms attached — flexible on the clause you flagged.",
                "goal":           "send_revised_terms",
            }
        elif "update_crm" not in l7_actions + l8_actions:
            return {
                "action_type":    "update_crm",
                "target_lead_id": "L-007",
                "goal":           "log_interaction",
                "metadata":       {"note": "L-007: call + docs. L-008: objection handled."},
            }
        return {"action_type": "wait", "target_lead_id": "L-007", "goal": "done"}

    # ── ESCALATION ──
    if task_id == "escalation":
        my_actions = action_types_for_lead(lead_id)
        if step_num == 1:
            return {
                "action_type":    "send_email",
                "target_lead_id": lead_id,
                "body": (
                    f"Hi {lead_name.split()[0]}, the documents for {company} have been pending "
                    f"for over a week. I'm going to loop in our senior rep to help."
                ),
                "goal": "context_before_escalation",
            }
        elif "escalate" not in my_actions:
            return {
                "action_type":    "escalate",
                "target_lead_id": lead_id,
                "body":           f"Escalating {lead_id} — docs requested twice, 8 days elapsed.",
                "goal":           "escalate_stuck_deal",
                "metadata":       {"reason": "docs_stuck_in_legal", "prior_requests": 2},
            }
        elif "update_crm" not in my_actions:
            return {
                "action_type":    "update_crm",
                "target_lead_id": lead_id,
                "goal":           "log_interaction",
                "metadata":       {"note": "Escalated. Docs requested 2x, no response."},
            }
        return {"action_type": "wait", "target_lead_id": lead_id, "goal": "done"}

    # ── CONSENT ──
    if task_id == "consent":
        l16_actions = action_types_for_lead("L-016")
        l12_actions = action_types_for_lead("L-012")
        l19_actions = action_types_for_lead("L-019")

        l19_revoked = "L-019" in revoked_leads
        l19_info = next((l for l in active_leads if l["lead_id"] == "L-019"), {})
        if l19_info.get("consent") is False:
            l19_revoked = True

        if not l16_actions:
            return make_intro("L-016", "Ria Chatterjee", "BrightWave", "send_email")
        elif not l12_actions:
            return make_intro("L-012", "Nisha Gupta", "Quantify AI", "send_message")
        elif not l19_actions and not l19_revoked:
            return make_intro("L-019", "Harish Nair", "BlueNorth", "send_email")
        elif "make_call" not in l16_actions:
            return {
                "action_type":    "make_call",
                "target_lead_id": "L-016",
                "body":           "Hi Ria, following up on the email about E.L.A.R.A. for BrightWave. Quick question — what's your current sales workflow?",
                "goal":           "discovery",
            }
        elif "update_crm" not in l16_actions + l12_actions:
            return {
                "action_type":    "update_crm",
                "target_lead_id": "L-016",
                "goal":           "log_interaction",
                "metadata":       {"note": "L-016: intro + discovery. L-012: intro sent. L-019: opted out."},
            }
        return {"action_type": "wait", "target_lead_id": "L-016", "goal": "done"}

    return {"action_type": "wait", "target_lead_id": lead_id}


# ─────────────────────────────────────────────
# MAIN LOOP (OpenEnv)
# ─────────────────────────────────────────────

async def run_task(env: ElaraEnv, task_name: str, use_llm: bool = True):
    max_steps = MAX_STEPS_PER_TASK.get(task_name, 5)
    rewards: List[float] = []
    steps = 0
    success = False

    print(f"\n{'='*60}", flush=True)
    print(f"  Task: {task_name.upper()} (max {max_steps} steps)", flush=True)
    print(f"{'='*60}", flush=True)

    log_start(task=task_name, model=MODEL_NAME or "fallback")

    result = await env.reset(task_id=task_name)
    obs = result.observation

    if DEBUG:
        print(f"  Lead: {obs['lead_name']} ({obs['lead_id']})", flush=True)
        print(f"  Stage: {obs['lead_stage']} | Channel: {obs['preferred_channel']}", flush=True)
        active = obs.get("active_leads", [])
        if len(active) > 1:
            print(f"  Active leads: {', '.join(l['lead_id'] for l in active)}", flush=True)
        print(f"  Hint: {obs.get('task_hint', 'N/A')}", flush=True)

    history: List[Dict[str, str]] = []
    actions_taken: List[Dict[str, str]] = []
    error_msg = None

    for step in range(1, max_steps + 1):
        if result.done:
            break

        print(f"\n  --- Step {step}/{max_steps} ---", flush=True)
        error_msg = None

        try:
            if use_llm:
                action_dict = get_llm_action(obs, step, max_steps, history)
            else:
                action_dict = fallback_action(obs, step, actions_taken)
        except Exception as e:
            error_msg = str(e)[:80]
            print(f"  Warning: LLM failed ({e}), using fallback agent", flush=True)
            action_dict = fallback_action(obs, step, actions_taken)

        action_type = action_dict.get("action_type", "wait")

        if DEBUG:
            print(f"  Action: {action_type} -> {action_dict.get('target_lead_id')}", flush=True)

        actions_taken.append({
            "type":   action_type,
            "target": action_dict.get("target_lead_id", ""),
            "goal":   action_dict.get("goal", ""),
        })

        action = ElaraAction(
            action_type=action_dict.get("action_type", "wait"),
            target_lead_id=action_dict.get("target_lead_id", ""),
            subject=action_dict.get("subject"),
            body=action_dict.get("body", ""),
            goal=action_dict.get("goal", ""),
            priority=action_dict.get("priority", "medium"),
            metadata=action_dict.get("metadata", {}),
        )

        result = await env.step(action)
        obs = result.observation
        reward = result.reward or 0.0
        done = result.done

        rewards.append(reward)
        steps = step

        log_step(step=step, action=action_type, reward=reward, done=done, error=error_msg)

        if DEBUG:
            print(f"  Reward: {reward:+.3f} | Done: {done} | Stage: {obs['lead_stage']}", flush=True)
            responses = obs.get("lead_responses", [])
            if responses:
                latest = responses[-1]
                print(f"  Lead says: \"{latest.get('text', '')[:80]}\"", flush=True)

        if done:
            success = True
            break

    score = sum(rewards) / max(len(rewards), 1)
    log_end(success=success, steps=steps, score=score, rewards=rewards)

    print(f"\n  {'PASS' if success else 'FAIL'} — Score: {score:.4f}", flush=True)

    return {"score": score, "pass": success, "rewards": rewards}


async def main():
    use_llm = bool(API_BASE_URL and API_KEY and MODEL_NAME)

    if use_llm:
        print(f"Using LLM: {MODEL_NAME} @ {API_BASE_URL}", flush=True)
    else:
        print("No LLM configured — using rule-based fallback agent", flush=True)

    env = await ElaraEnv.from_docker_image(IMAGE_NAME)

    try:
        results = {}
        for task_id in ["easy", "medium", "hard", "escalation", "consent"]:
            results[task_id] = await run_task(env, task_id, use_llm=use_llm)

        print(f"\n{'='*60}", flush=True)
        print(f"  INFERENCE RESULTS SUMMARY", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  {'Task':<12} {'Score':>7}  {'Pass':>5}", flush=True)
        print(f"  {'-'*30}", flush=True)
        for task_id, r in results.items():
            pf = "PASS" if r["pass"] else "FAIL"
            print(f"  {task_id:<12} {r['score']:>7.4f}  {pf:>5}", flush=True)
        print(flush=True)
    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())
