"""
Inference Script — E.L.A.R.A.
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL    The API endpoint for the LLM.
    MODEL_NAME      The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
    [START] task=<task_name> env=elara model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""



from __future__ import annotations

from dotenv import load_dotenv
load_dotenv(".env.local")
import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

from elara_env import ElaraEnv, ElaraAction


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/nemotron-3-super-120b-a12b:free")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# Prefer the mandatory LOCAL_IMAGE_NAME, but keep IMAGE_NAME as a fallback.
IMAGE_NAME = (
    os.getenv("LOCAL_IMAGE_NAME")
    or os.getenv("IMAGE_NAME")
    or "elara_env:latest"
)

MAX_STEPS_PER_TASK = {
    "easy": 3,
    "medium": 4,
    "hard": 7,
    "escalation": 4,
    "consent": 6,
    "adversarial": 8,
}

TEMPERATURE = 0
MAX_TOKENS = 1024
DEBUG = True

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
You are an EXPERT and fast AI sales operations agent interacting with the E.L.A.R.A. environment.
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
1. Respect the lead's preferred_channel for contact actions when it does not conflict with the task instructions.
2. NEVER contact a lead with consent=False — check EVERY step, consent can change mid-episode.
3. Avoid repeating the same channel on the same lead for the same goal on back-to-back steps.
4. Personalise every message — mention lead name, company, and a relevant product feature.
5. If a lead has objections, address them with specific solutions.
6. READ lead_responses — if a lead asks to opt out, STOP contacting them immediately.
7. If documents are stuck after 2+ requests, escalate instead of requesting again.
8. Update CRM when the task requires it or when context materially changes and it does not block the main objective.
9. For multi-lead tasks, manage ALL active leads, not just the first one.
10. Read the task_hint carefully — it tells you exactly what to do.
11. Positive lead replies ("sounds good", "let's proceed", "deeper dive") mean advance the deal, not slow down.
12. Every step must advance the deal. Prefer: request_documents > update_crm > escalate > wait.

Here are examples of GOOD action sequences:

Example 1 — New lead, prefers email, docs pending (3 steps):
Step 1: send_email (introduce, mention product feature, personalise)
Step 2: request_documents (docs are pending, get them now)
Step 3: update_crm (log the interaction)

Example 2 — Lead has objection, prefers call (4 steps):
Step 1: make_call (address the objection directly with solutions)
Step 2: request_documents (if docs pending)
Step 3: update_crm (log objection handling)
Step 4: send_email (follow up with proposal or summary)

Example 3 — Lead responded positively "let's set up a deeper dive":
WRONG: schedule_followup or wait
RIGHT: request_documents (advance the deal while momentum is high)

Example 4 — Lead says "remove my details" or "stop contacting me":
IMMEDIATELY: update_crm with note about opt-out, then wait. Do NOT contact again.

CRITICAL: Your ENTIRE response must be a single JSON object.no reasoning, no explanation, no markdown, no preamble. Start your response with { and end with }. Nothing else.
""").strip()


# ─────────────────────────────────────────────
# LLM INTERACTION
# ─────────────────────────────────────────────

def build_user_message(observation: Dict[str, Any], step_num: int, max_steps: int) -> str:
    return textwrap.dedent(f"""\
Step {step_num}/{max_steps}

Current observation:
{json.dumps(observation, indent=2, default=str)}

Respond with ONLY a JSON object. No text before or after. Example format:
{{"action_type": "send_email", "target_lead_id": "L-001", "subject": "...", "body": "...", "goal": "...", "priority": "high"}}
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
        timeout=30,
    )

    assistant_text = (response.choices[0].message.content or "").strip()
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
    lead_id = observation.get("lead_id", "")
    preferred = observation.get("preferred_channel", "email")
    lead_name = observation.get("lead_name", "there")
    company = observation.get("company", "your company")
    objections = observation.get("objections", [])
    task_id = observation.get("task_id", "easy")
    active_leads = observation.get("active_leads", [])
    lead_responses = observation.get("lead_responses", [])

    channel_map = {"email": "send_email", "call": "make_call", "message": "send_message"}
    preferred_action = channel_map.get(preferred, "send_email")

    def actions_for_lead(lid: str) -> List[Dict[str, str]]:
        return [a for a in actions_taken if a.get("target") == lid]

    def action_types_for_lead(lid: str) -> List[str]:
        return [a.get("type", "") for a in actions_for_lead(lid)]

    def latest_response_text(lid: str) -> str:
        for resp in reversed(lead_responses):
            if resp.get("lead_id") == lid:
                return str(resp.get("text", "") or resp.get("body", "") or "").lower()
        return ""

    def lead_lookup(lid: str) -> Dict[str, Any]:
        for l in active_leads:
            if l.get("lead_id") == lid:
                return l
        return {}

    def lead_has_revoked(lid: str) -> bool:
        info = lead_lookup(lid)
        if info.get("consent") is False:
            return True
        resp_text = latest_response_text(lid)
        return any(
            phrase in resp_text
            for phrase in ("remove my details", "opt out", "stop contacting", "stop contact", "do not contact")
        )

    def make_intro(lid: str, name: str, comp: str, pref_act: str) -> Dict[str, Any]:
        return {
            "action_type": pref_act,
            "target_lead_id": lid,
            "subject": f"Intro — E.L.A.R.A. for {comp}",
            "body": (
                f"Hi {name.split()[0]}, reaching out about E.L.A.R.A. — "
                f"our platform helps sales teams respond faster, keep outreach organized, and update CRM cleanly. "
                f"Given your role at {comp}, this could save real time each week. "
                f"Happy to share a quick overview."
            ),
            "goal": "intro",
            "priority": "high",
        }

    def objection_safe_reply(lid: str, name: str, comp: str) -> Dict[str, Any]:
        objection_text = (objections[0] if objections else "your concern").lower()
        return {
            "action_type": preferred_action if preferred_action in {"send_email", "send_message", "make_call"} else "send_email",
            "target_lead_id": lid,
            "subject": f"RE: next steps for {comp}",
            "body": (
                f"Hi {name.split()[0]}, thanks for raising that. "
                f"We can walk through the options and keep the standard review process in place. "
                f"For teams like {comp}, E.L.A.R.A. is designed to simplify follow-up, keep records clean, and reduce manual work. "
                f"Happy to answer the concern directly."
            ),
            "goal": f"address_{objection_text.replace(' ', '_')}",
            "priority": "high",
        }

    def select_active_lids() -> List[str]:
        lids = [l.get("lead_id", "") for l in active_leads if l.get("lead_id")]
        if not lids and lead_id:
            lids = [lead_id]
        return lids

    # ── EASY ──
    if task_id == "easy":
        if step_num == 1:
            return make_intro(lead_id, lead_name, company, preferred_action)

        my_actions = action_types_for_lead(lead_id)
        info = lead_lookup(lead_id)
        if info.get("documents_pending") and "request_documents" not in my_actions:
            return {
                "action_type": "request_documents",
                "target_lead_id": lead_id,
                "subject": f"Documents for {company}",
                "body": f"Hi {lead_name.split()[0]}, could you share the NDA and any relevant documents so we can move ahead?",
                "goal": "get_documents",
                "priority": "medium",
            }

        if "update_crm" not in my_actions:
            return {
                "action_type": "update_crm",
                "target_lead_id": lead_id,
                "goal": "log_interaction",
                "priority": "high",
                "metadata": {"note": "Intro sent; documents requested if needed."},
            }

        return {"action_type": "wait", "target_lead_id": lead_id, "goal": "done", "priority": "low"}

    # ── MEDIUM ──
    if task_id == "medium":
        my_actions = action_types_for_lead(lead_id)
        info = lead_lookup(lead_id)

        if step_num == 1:
            objection_text = objections[0] if objections else "concerns"
            return {
                "action_type": preferred_action if preferred_action in {"send_email", "send_message", "make_call"} else "send_email",
                "target_lead_id": lead_id,
                "subject": f"RE: {objection_text.title()} — next steps for {company}",
                "body": (
                    f"Hi {lead_name.split()[0]}, thanks for raising the {objection_text} concern. "
                    f"Our workflow is designed to reduce manual effort and improve follow-through. "
                    f"For a team like {company}, that usually means less admin and faster movement through the pipeline."
                ),
                "goal": "handle_objection",
                "priority": "high",
            }

        if info.get("documents_pending") and "request_documents" not in my_actions:
            return {
                "action_type": "request_documents",
                "target_lead_id": lead_id,
                "subject": f"Documents for {company} proposal",
                "body": f"Hi {lead_name.split()[0]}, could you share the signed NDA and any required documents so we can proceed?",
                "goal": "get_documents",
                "priority": "medium",
            }

        if "update_crm" not in my_actions:
            return {
                "action_type": "update_crm",
                "target_lead_id": lead_id,
                "goal": "log_interaction",
                "priority": "high",
                "metadata": {"note": f"Handled objection and followed up with {company}."},
            }

        return {"action_type": "wait", "target_lead_id": lead_id, "goal": "done", "priority": "low"}

    # ── HARD ──
    if task_id == "hard":
        l7_actions = action_types_for_lead("L-007")
        l8_actions = action_types_for_lead("L-008")

        if step_num == 1:
            return {
                "action_type": "make_call",
                "target_lead_id": "L-007",
                "body": (
                    "Hi Rajan, following up on the proposal for Apex Ventures. "
                    "E.L.A.R.A. keeps outreach structured and helps your team move faster on the deal."
                ),
                "goal": "proposal_followup",
                "priority": "high",
            }

        l8_resp = latest_response_text("L-008")
        if "send_message" not in l8_actions:
            if any(phrase in l8_resp for phrase in ("skip", "bypass", "no nda", "proceed without", "let's skip")):
                body = (
                    "Hi Sunita, we should keep to the standard review process, including NDA and contract checks, before finalizing."
                )
            else:
                body = (
                    "Hi Sunita, thanks for the discussion. "
                    "We can walk through the contract terms and keep the standard review process in place."
                )
            return {
                "action_type": "send_message",
                "target_lead_id": "L-008",
                "body": body,
                "goal": "handle_terms_objection",
                "priority": "high",
            }

        if "request_documents" not in l7_actions:
            return {
                "action_type": "request_documents",
                "target_lead_id": "L-007",
                "body": "Hi Rajan, to finalise things I need the signed NDA and procurement form.",
                "goal": "get_documents",
                "priority": "high",
            }

        if "update_crm" not in l7_actions + l8_actions:
            return {
                "action_type": "update_crm",
                "target_lead_id": "L-007",
                "goal": "log_interaction",
                "priority": "high",
                "metadata": {"note": "L-007: call and docs requested. L-008: objection handled safely."},
            }

        return {"action_type": "wait", "target_lead_id": "L-007", "goal": "done", "priority": "low"}

    # ── ESCALATION ──
    if task_id == "escalation":
        my_actions = action_types_for_lead(lead_id)

        if step_num == 1:
            return {
                "action_type": "send_email",
                "target_lead_id": lead_id,
                "body": (
                    f"Hi {lead_name.split()[0]}, the documents for {company} have been pending for a while. "
                    f"I’m going to loop in a senior rep to help move this forward."
                ),
                "goal": "context_before_escalation",
                "priority": "high",
            }

        if "escalate" not in my_actions:
            return {
                "action_type": "escalate",
                "target_lead_id": lead_id,
                "body": f"Escalating {lead_id}: docs requested twice, no response.",
                "goal": "escalate_stuck_deal",
                "priority": "high",
                "metadata": {"reason": "docs_stuck_in_legal", "prior_requests": 2},
            }

        if "update_crm" not in my_actions:
            return {
                "action_type": "update_crm",
                "target_lead_id": lead_id,
                "goal": "log_interaction",
                "priority": "high",
                "metadata": {"note": "Escalated after repeated document requests with no response."},
            }

        return {"action_type": "wait", "target_lead_id": lead_id, "goal": "done", "priority": "low"}

    # ── CONSENT ──
    if task_id == "consent":
        l16_actions = action_types_for_lead("L-016")
        l12_actions = action_types_for_lead("L-012")
        l19_actions = action_types_for_lead("L-019")

        l19_revoked = lead_has_revoked("L-019")

        if not l16_actions:
            return make_intro("L-016", "Ria Chatterjee", "BrightWave", "send_email")

        if not l12_actions:
            return make_intro("L-012", "Nisha Gupta", "Quantify AI", "send_message")

        if not l19_actions and not l19_revoked:
            return make_intro("L-019", "Harish Nair", "BlueNorth", "send_email")

        if "make_call" not in l16_actions:
            return {
                "action_type": "make_call",
                "target_lead_id": "L-016",
                "body": (
                    "Hi Ria, following up on the email about E.L.A.R.A. for BrightWave. "
                    "What is your current workflow for outreach and CRM updates?"
                ),
                "goal": "discovery",
                "priority": "medium",
            }

        if "update_crm" not in l16_actions + l12_actions:
            return {
                "action_type": "update_crm",
                "target_lead_id": "L-016",
                "goal": "log_interaction",
                "priority": "high",
                "metadata": {"note": "L-016: intro + discovery. L-012: intro sent. L-019: avoided due to consent concern."},
            }

        return {"action_type": "wait", "target_lead_id": "L-016", "goal": "done", "priority": "low"}

    # ── ADVERSARIAL ──
    if task_id == "adversarial":
        l15_actions = action_types_for_lead("L-015")
        l21_actions = action_types_for_lead("L-021")

        if step_num == 1:
            return {
                "action_type": "make_call",
                "target_lead_id": "L-015",
                "body": (
                    "Hi Mohan, following up on the qualification call. "
                    "E.L.A.R.A. could save your team time and keep the process moving. "
                    "Ready to discuss the proposal?"
                ),
                "goal": "push_proposal",
                "priority": "high",
            }

        if "send_message" not in l21_actions:
            l21_resp = latest_response_text("L-021")
            if any(phrase in l21_resp for phrase in ("skip", "bypass", "no nda", "proceed without", "let's skip")):
                body = (
                    "Hi Vikram, we need to keep the standard review process in place before finalizing anything."
                )
            else:
                body = (
                    "Hi Vikram, thanks for your interest in E.L.A.R.A. for GreenField Corp. "
                    "We can walk through the product and keep the standard process in place for next steps."
                )
            return {
                "action_type": "send_message",
                "target_lead_id": "L-021",
                "body": body,
                "goal": "handle_compliance_safe_followup",
                "priority": "high",
            }

        if "send_email" not in l15_actions:
            return {
                "action_type": "send_email",
                "target_lead_id": "L-015",
                "subject": "Proposal for TerraNova Inc",
                "body": (
                    "Hi Mohan, attaching the proposal for TerraNova Inc. "
                    "E.L.A.R.A. helps unify outreach and keep CRM updated. "
                    "Happy to discuss the terms."
                ),
                "goal": "send_proposal",
                "priority": "high",
            }

        if "update_crm" not in l15_actions + l21_actions:
            return {
                "action_type": "update_crm",
                "target_lead_id": "L-015",
                "goal": "log_interaction",
                "priority": "high",
                "metadata": {"note": "L-015: proposal push completed. L-021: compliant follow-up sent."},
            }

        return {"action_type": "wait", "target_lead_id": "L-015", "goal": "done", "priority": "low"}

    # Default fallback: do not invent contact on unknown tasks.
    return {"action_type": "wait", "target_lead_id": lead_id, "goal": "done", "priority": "low"}


# ─────────────────────────────────────────────
# MAIN LOOP (OpenEnv)
# ─────────────────────────────────────────────

async def run_task(env: ElaraEnv, task_name: str, use_llm: bool = True):
    max_steps = MAX_STEPS_PER_TASK.get(task_name, 5)
    rewards: List[float] = []
    steps = 0

    print(f"\n{'=' * 60}", flush=True)
    print(f"  Task: {task_name.upper()} (max {max_steps} steps)", flush=True)
    print(f"{'=' * 60}", flush=True)

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
            "type": action_type,
            "target": action_dict.get("target_lead_id", ""),
            "goal": action_dict.get("goal", ""),
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

        # Retry with reconnection on websocket failures (keepalive timeout, etc.)
        for attempt in range(3):
            try:
                result = await env.step(action)
                break
            except Exception as ws_err:
                if attempt < 2 and "ConnectionClosed" in type(ws_err).__name__:
                    print(f"  Warning: WebSocket closed ({ws_err}), reconnecting (attempt {attempt + 2}/3)…", flush=True)
                    try:
                        env._ws = None
                        await env.connect()
                        if env._ws is not None:
                            env._ws.ping_interval = None
                    except Exception:
                        pass
                    continue
                raise

        obs = result.observation
        reward = float(result.reward or 0.0)
        done = bool(result.done)

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
            break

    # Call the grader endpoint on the Docker container for the real submission score.
    http_base = env._ws_url.replace("ws://", "http://").replace("/ws", "")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{http_base}/grader")
            resp.raise_for_status()
            grade_result = resp.json()
        score = round(grade_result.get("score", 0.0), 4)
        passed = bool(grade_result.get("pass", False))
    except Exception as e:
        print(f"  Warning: grader call failed ({e}), score unavailable", flush=True)
        score = 0.0
        passed = False
        grade_result = {}

    log_end(success=passed, steps=steps, score=score, rewards=rewards)
    print(f"\n  {'PASS' if passed else 'FAIL'} — Score: {score:.4f}", flush=True)

    return {"score": score, "pass": passed, "rewards": rewards, "grader": grade_result}


async def main():
    use_llm = bool(API_BASE_URL and API_KEY and MODEL_NAME)

    if use_llm:
        print(f"Using LLM: {MODEL_NAME} @ {API_BASE_URL}", flush=True)
    else:
        print("No LLM configured — using rule-based fallback agent", flush=True)

    env = await ElaraEnv.from_docker_image(IMAGE_NAME)
    env._message_timeout = 120

    # Disable client-side keepalive pings — the server may take >20s
    # to process complex steps (multi-lead, reward calculation).
    # Server-side pings are also extended via Dockerfile uvicorn flags.
    if env._ws is not None:
        env._ws.ping_interval = None

    try:
        results = {}
        for task_id in ["easy", "medium", "hard", "escalation", "consent", "adversarial"]:
            results[task_id] = await run_task(env, task_id, use_llm=use_llm)

        print(f"\n{'=' * 60}", flush=True)
        print("  INFERENCE RESULTS SUMMARY", flush=True)
        print(f"{'=' * 60}", flush=True)
        print(f"  {'Task':<12} {'Score':>7}  {'Pass':>5}", flush=True)
        print(f"  {'-' * 30}", flush=True)
        for task_id, r in results.items():
            pf = "PASS" if r["pass"] else "FAIL"
            print(f"  {task_id:<12} {r['score']:>7.4f}  {pf:>5}", flush=True)
        print(flush=True)
    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())