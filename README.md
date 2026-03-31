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

Tests all 3 tasks with good and bad agents. Prints every reward, stage transition, and final grade.

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
| GET    | `/tasks`  | List all 3 tasks with their configs |
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
├── run.py             ← Smoke test — runs all tasks, no server needed
├── requirements.txt
├── Dockerfile
├── openenv.yaml
│
├── app/
│   ├── environment.py   ← ElaraEnv: reset() / step() / state()
│   ├── models.py        ← Pydantic models: Action, Observation, LeadProfile...
│   ├── reward_engine.py ← 8 reward components, exact values from spec
│   ├── crm_simulator.py ← 20 leads + stage transition logic
│   └── grader.py        ← 5-dimension deterministic scorer (0.0–1.0)
│
└── tasks/
    ├── easy.json        ← First-Touch Outreach (L-001, 3 steps)
    ├── medium.json      ← Follow-Up Handling (L-004, 4 steps)
    └── hard.json        ← Multi-Channel Orchestration (L-007, 6 steps)
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
obs.consent                # bool — False = do not contact
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
| Duplicate outreach | −0.20 | Same channel two steps in a row |
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

## Tasks

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

### Hard — Multi-Channel Orchestration
- Lead: Rajan Mehta (L-007) — proposal sent 7 days ago, prefers calls, docs pending, hot
- Budget: 6 steps
- Goal: call to follow up, email for docs, close deal
- Required actions: `make_call`, `request_documents`, `update_crm`

---

## Baseline Scores (good agent)

| Task | Score | Pass |
|---|---|---|
| Easy   | 0.955 | ✓ |
| Medium | 0.930 | ✓ |
| Hard   | 0.670 | ✓ |

---

## Lead Scenarios (20 leads)

All 20 leads are in `app/crm_simulator.py`. Key scenarios covered:

- `L-001` — New lead, email preferred, docs pending (Easy task)
- `L-004` — Pricing objection, follow-up overdue (Medium task)
- `L-007` — Hot lead, proposal 7d old, call preferred (Hard task)
- `L-009` — **consent=False** — contacting triggers −0.5 penalty
- `L-010` — Contacted today — contacting again triggers timing penalty
- `L-014` — Docs requested twice, needs escalation
- `L-020` — Closed won — upsell candidate