---
title: E.L.A.R.A.
emoji: 📞
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
---

# E.L.A.R.A.
## Extensive LLM Applied Reasoning Agent

A sandboxed multi-channel sales operations environment for training and evaluating AI agents.

The agent reads a product, manages leads in a simulated CRM, picks the right communication channel, executes compliant outreach, and updates state — all scored deterministically.

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd elara

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the smoke test (no server needed)
python run.py

# 4. Start the API server
uvicorn server:app --host 0.0.0.0 --port 7860 --reload
```

---

## How to Run — 3 Ways

### Way 1 — Python smoke test (fastest, no server)

Tests all 4 tasks with good and bad agents. Prints every reward, stage transition, and final grade.

```bash
python run.py
```

Output shows:
- Initial lead observation (stage, sentiment, preferred channel, docs pending)
- Each action with reward breakdown (+/- per component)
- Grader scorecard across 5 dimensions
- Good vs bad agent comparison table


### Way 2 — API server (interact manually with curl or Swagger)

```bash
uvicorn server:app --host 0.0.0.0 --port 7860 --reload
```

Server is now live at `http://localhost:7860`

**Interactive docs (Swagger UI):** open `http://localhost:7860/docs` in your browser — you can call every endpoint from the browser, no curl needed.

**Endpoints:**

| Method | Endpoint | What it does |
|--------|----------|--------------|
| GET    | `/health` | Ping — returns `{"status": "ok"}` |
| GET    | `/tasks`  | List all 4 tasks with their configs |
| POST   | `/reset`  | Start a new episode |
| POST   | `/step`   | Take one action |
| GET    | `/state`  | Full episode state |
| POST   | `/grader` | Score the current episode |


**Full curl walkthrough — Easy task:**

```bash
# Step 1: reset
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'

# Step 2: send an email (correct channel for L-001)
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "send_email",
      "target_lead_id": "L-001",
      "subject": "Intro — E.L.A.R.A. for NovaTech",
      "body": "Hi Arun, reaching out about E.L.A.R.A. — cuts lead response time by 60%. Happy to demo!",
      "goal": "intro",
      "priority": "high"
    }
  }'

# Step 3: grade it
curl -s -X POST http://localhost:7860/grader
```

**Medium task:**

```bash
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "medium"}'
```

**Hard task:**

```bash
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "hard"}'
```


### Way 3 — Docker

```bash
# Build
docker build -t elara .

# Run
docker run -p 7860:7860 elara

# Server is now live at http://localhost:7860
```

---

## Project Structure

```
elara/
├── server.py          ← FastAPI app (all endpoints)
├── inference.py       ← LLM inference script (mandatory for submission)
├── run.py             ← Smoke test — runs all tasks, no server needed
├── requirements.txt
├── Dockerfile
├── openenv.yaml
│
├── app/
│   ├── environment.py   ← ElaraEnv: reset() / step() / state()
│   ├── models.py        ← Pydantic models: Action, Observation, LeadProfile...
│   ├── reward_engine.py ← 8 reward components + objection addressing
│   ├── crm_simulator.py ← 20 leads + stage transitions + dynamic responses
│   └── grader.py        ← 5-dimension scorer with multi-lead support
│
└── tasks/
    ├── easy.json        ← First-Touch Outreach (L-001, 3 steps)
    ├── medium.json      ← Follow-Up Handling (L-004, 4 steps)
    ├── hard.json        ← Multi-Lead Orchestration (L-007 + L-008, 7 steps)
    └── consent.json     ← Compliance Navigation (L-016 + L-012 + L-019, 6 steps)
```

---

## Environment API

```python
import sys
sys.path.insert(0, "app")

from environment import ElaraEnv
from models import Action

env = ElaraEnv()

# Reset — pick a task
obs = env.reset(task_id="easy")   # or "medium" or "hard"

print(obs.lead_name)        # Arun Sharma
print(obs.preferred_channel)  # email
print(obs.task_hint)        # guidance for the agent

# Step — take an action
obs, reward, done, info = env.step(Action(
    action_type="send_email",
    target_lead_id="L-001",
    subject="Intro",
    body="Hi Arun, reaching out about E.L.A.R.A. ...",
    goal="intro",
))

print(reward)                      # 0.85
print(info["stage_after"])         # contacted
print(info["reward_breakdown"])    # per-component breakdown

# Grade the episode
from grader import grade
from models import EpisodeState
result = grade(EpisodeState(**env.state()))
print(result["score"])   # 0.955
print(result["pass"])    # True
```

---

## Action Space (9 types)

| action_type | What it does | Channel |
|---|---|---|
| `send_email` | Send an email to the lead | email |
| `make_call` | Make a phone call | call |
| `send_message` | Send a chat/WhatsApp message | message |
| `request_documents` | Request NDA / docs via email | email |
| `update_crm` | Log note to CRM | — |
| `schedule_followup` | Set a follow-up timer | — |
| `run_campaign` | Trigger a bulk email campaign | email |
| `escalate` | Escalate to senior rep | — |
| `wait` | Do nothing this step | — |

Action fields: `action_type`, `target_lead_id`, `subject`, `body`, `goal`, `priority`, `metadata`

---

## Observation Space

Each `obs` contains:

```python
obs.lead_name              # "Arun Sharma"
obs.company                # "NovaTech Solutions"
obs.lead_stage             # "new" | "contacted" | "qualified" | ...
obs.sentiment              # "cold" | "neutral" | "warm" | "hot"
obs.preferred_channel      # "email" | "call" | "message" | "any"
obs.days_since_last_contact
obs.next_followup_due      # days until follow-up is due
obs.documents_pending      # bool
obs.objections             # ["pricing", "not the right time", ...]
obs.consent                # bool — False = do not contact (can change mid-episode!)
obs.active_leads           # summaries of ALL active leads in multi-lead tasks
obs.lead_responses         # recent replies FROM leads — critical for decision-making
obs.product_context        # features, value props, pricing, FAQs
obs.recent_history         # last 5 interactions
obs.policy_constraints     # consent_required, max_steps_remaining, ...
obs.task_hint              # plain-English guidance
obs.available_actions      # all 9 action types
```

---

## Reward Design

| Signal | Value | Trigger |
|---|---|---|
| Correct channel | +0.20 | Action channel matches lead preference |
| Correct timing | +0.10 | Follow-up within due window |
| Good message | +0.20 | Body present, personalised, product-relevant |
| CRM update | +0.10 | `update_crm` called |
| Lead progression | +0.30 | Stage advanced |
| Policy compliance | +0.10 | No violations |
| Wrong channel | −0.20 | Ignored lead preference |
| Duplicate outreach | −0.20 | Same channel + same goal back-to-back |
| Consent violation | −0.50 | Contacted lead with consent=False |
| Looping | −0.10 | >80% of step budget used, not done |

---

## Grader — 5 Dimensions

| Dimension | Weight | What it measures |
|---|---|---|
| Task completion | 35% | Lead at required stage + all required actions taken |
| Channel correctness | 20% | Ratio of steps matching lead's preferred channel |
| CRM accuracy | 15% | CRM updated when required |
| Compliance | 20% | No consent violations, no duplicate spam |
| Efficiency | 10% | Solved without burning the step budget |

**Pass threshold: ≥ 0.60**

---

## Tasks (4 total, easy → hard)

### Easy — First-Touch Outreach
- Lead: Arun Sharma (L-001) — new, prefers email, docs pending
- Budget: 3 steps
- Goal: contact lead via correct channel
- Required actions: `send_email`

### Medium — Follow-Up Handling
- Lead: Priya Iyer (L-004) — contacted, pricing objection, docs pending, prefers email
- Budget: 4 steps
- Goal: handle objection, request documents, update CRM
- Required actions: `request_documents`, `update_crm`

### Hard — Multi-Lead Orchestration
- **Two leads**: Rajan Mehta (L-007, proposal_sent, call preferred) + Sunita Kapoor (L-008, negotiating, message preferred, contract objections)
- Budget: 7 steps
- Goal: advance both leads across channels, update CRM
- Required: `make_call` + `request_documents` (L-007), `send_message` (L-008), `update_crm`

### Consent — Compliance Navigation
- **Three leads**: L-016 (warm, email), L-012 (neutral, message), L-019 (cold, email — **revokes consent after first contact**)
- Budget: 6 steps
- Goal: contact willing leads, STOP contacting L-019 after opt-out
- Required: `send_email` (L-016), `send_message` (L-012), `update_crm`

---

## Key Features

- **Dynamic lead responses**: leads reply after contact actions with template-based responses that vary by sentiment, message quality, and RNG seed
- **State mutations**: sentiment shifts, consent revocation, and objection changes happen dynamically based on agent behaviour
- **Multi-lead tasks**: hard and consent tasks require managing multiple leads simultaneously with a shared step budget
- **Seeded stochastic variation**: `seed` parameter randomizes objections, sentiment, and timing — same seed = reproducible, different seed = variety
- **Compliance traps**: leads can revoke consent mid-episode — agents must read responses and adapt

---

## Baseline Scores (good agent, seed=0)

| Task | Score | Pass |
|---|---|---|
| Easy       | 0.955 | ✓ |
| Medium     | 0.910 | ✓ |
| Hard       | 0.860 | ✓ |
| Consent    | 0.910 | ✓ |

---

## Lead Scenarios (20 leads)

All 20 leads are in `app/crm_simulator.py`. Key scenarios covered:

- `L-001` — New lead, email preferred, docs pending (Easy task)
- `L-004` — Pricing objection, follow-up overdue (Medium task)
- `L-007` — Hot lead, proposal 7d old, call preferred (Hard task — multi-lead)
- `L-008` — Negotiating, contract objections, message preferred (Hard task — multi-lead)
- `L-009` — **consent=False** — contacting triggers −0.5 penalty
- `L-012` — Message preferred, part of consent task
- `L-014` — Docs requested twice, needs escalation (Escalation task)
- `L-016` — Warm webinar attendee, email preferred (Consent task)
- `L-019` — Cold lead, **revokes consent mid-episode** (Consent task trap)
- `L-020` — Closed won — upsell candidate