<<<<<<< HEAD
#!/usr/bin/env python3
"""
run.py — E.L.A.R.A. Easy Task: End-to-End Smoke Test
Run from repo root: python run.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from environment import ElaraEnv
from models import Action, EpisodeState
from grader import grade

G  = "\033[92m"; Y  = "\033[93m"; R  = "\033[91m"
B  = "\033[94m"; C  = "\033[96m"; W  = "\033[97m"
DIM= "\033[2m";  RST= "\033[0m";  BOLD="\033[1m"

def sep(c="─", n=62): print(DIM + c*n + RST)
def hdr(t): sep("═"); print(f"{BOLD}{C}  {t}{RST}"); sep("═")
def info(m): print(f"  {B}→{RST}  {m}")

def print_obs(obs):
    d = obs.model_dump()
    sep()
    print(f"  {BOLD}Lead:{RST}  {W}{d['lead_name']}{RST}  |  {d['company']}  |  {d['role']}")
    print(f"  {BOLD}Stage:{RST} {Y}{d['lead_stage']}{RST}  |  sentiment: {d['sentiment']}  |  consent: {G if d['consent'] else R}{d['consent']}{RST}")
    print(f"  {BOLD}Channel:{RST} last={d['last_contact_channel']}  preferred={C}{d['preferred_channel']}{RST}  days_since={d['days_since_last_contact']}")
    print(f"  {BOLD}Docs pending:{RST} {R if d['documents_pending'] else G}{d['documents_pending']}{RST}  |  objections: {d['objections'] or 'none'}")
    print(f"  {BOLD}Steps left:{RST} {d['policy_constraints']['max_steps_remaining']}")
    print(f"  {DIM}Hint: {d['task_hint']}{RST}")
    sep()

def print_step(n, action, reward, done, info_d):
    print(f"\n{BOLD}  Step {n} — {C}{action.action_type.upper()}{RST}{BOLD} → {action.target_lead_id}{RST}")
    if action.body:
        preview = action.body[:80] + ("..." if len(action.body)>80 else "")
        print(f"  {DIM}Body: {preview}{RST}")
    print(f"  Reward: {G if reward>0 else R}{reward:+.4f}{RST}  |  Stage: {Y}{info_d['stage_before']}{RST} → {G}{info_d['stage_after']}{RST}  |  Done: {done}")
    bd = {k:v for k,v in info_d.get("reward_breakdown",{}).items() if v["reward"]!=0}
    if bd:
        print(f"  {DIM}Breakdown:{RST}")
        for k,v in bd.items():
            col = G if v["reward"]>0 else R
            print(f"    {col}{v['reward']:+.2f}{RST}  {k:<20}  {DIM}{v['reason']}{RST}")
    print(f"  Running total: {BOLD}{info_d['total_reward']:+.4f}{RST}")

def print_grade(result):
    hdr("GRADER RESULTS")
    sc = result["score"]
    col = G if sc>=0.8 else Y if sc>=0.6 else R
    print(f"\n  {BOLD}Final Score:{RST}  {col}{BOLD}{sc:.4f} / 1.0{RST}  {'✓ PASS' if result['pass'] else '✗ FAIL'}")
    print(f"  Task: {result['task_id']}  Steps: {result['steps_used']}/{result['max_steps']}  Stage: {Y}{result['final_stage']}{RST}\n")
    sep()
    print(f"  {'Dimension':<22} {'Score':>6}  {'Weight':>6}  {'Weighted':>8}  Reason")
    sep()
    for dim, data in result["dimensions"].items():
        s = data["score"]; w = data["weight"]
        c2 = G if s>=0.8 else Y if s>=0.5 else R
        print(f"  {dim:<22} {c2}{s:>6.2f}{RST}  {w:>6.0%}  {s*w:>8.4f}  {DIM}{data['reason'][:50]}{RST}")
    sep()
    print(f"  {'TOTAL':<22} {col}{sc:>6.4f}{RST}  {'100%':>6}  {sc:>8.4f}\n")

def run_task(label, task_id, actions):
    hdr(f"E.L.A.R.A. — {label}")
    env = ElaraEnv()
    obs = env.reset(task_id=task_id)
    print(f"\n{BOLD}  INITIAL OBSERVATION{RST}")
    print_obs(obs)
    print(f"\n{BOLD}  EXECUTING {len(actions)} ACTIONS{RST}")
    for i, action in enumerate(actions, 1):
        info(f"Step {i}: {action.action_type} → {action.target_lead_id}" + (f"  goal={action.goal}" if action.goal else ""))
    print()
    done = False
    for i, action in enumerate(actions, 1):
        obs, reward, done, info_d = env.step(action)
        print_step(i, action, reward, done, info_d)
        if done and i < len(actions):
            print(f"  {Y}Episode ended early at step {i}{RST}")
            break
    result = grade(EpisodeState(**env.state()))
    print_grade(result)
    return result

# ─────────────────────────────────────────────────────────────────────────────

# ── GOOD agent: does everything right ────────────────────────────────────────
good_actions = [
    Action(
        action_type="send_email",
        target_lead_id="L-001",
        subject="Quick intro — E.L.A.R.A. for NovaTech",
        body=(
            "Hi Arun, hope things are going well at NovaTech Solutions. "
            "Reaching out about E.L.A.R.A. — our sales ops platform that cuts "
            "lead response time by 60% and unifies email, calls and messages in one place. "
            "Given your Head of Sales role, this could save your team real hours each week. "
            "Happy to set up a quick demo — does this week work?"
        ),
        goal="intro",
        priority="high",
    ),
    Action(
        action_type="request_documents",
        target_lead_id="L-001",
        subject="Documents to proceed",
        body=(
            "Hi Arun, to tailor the proposal for NovaTech I'll need the signed NDA "
            "and a brief overview of your current sales tooling. "
            "Could you share these when convenient? Won't take long."
        ),
        goal="get_documents",
    ),
    Action(
        action_type="update_crm",
        target_lead_id="L-001",
        goal="log_interaction",
        metadata={"note": "Intro email sent. Doc request sent. Lead is new, email preferred. Awaiting reply."},
    ),
]

good_result = run_task("EASY TASK · Good Agent", "easy", good_actions)

# ── BAD agent: wrong channel, empty body, no CRM ─────────────────────────────
bad_actions = [
    Action(
        action_type="make_call",    # WRONG — Arun prefers email
        target_lead_id="L-001",
        body="",                    # empty body
        goal="",
    ),
    Action(
        action_type="send_message", # still wrong channel
        target_lead_id="L-001",
        body="Hey",                 # no personalisation
        goal="",
    ),
    Action(
        action_type="wait",         # useless
        target_lead_id="L-001",
    ),
]

bad_result = run_task("EASY TASK · Bad Agent", "easy", bad_actions)

# ── Summary ───────────────────────────────────────────────────────────────────
hdr("SUMMARY")
print(f"  Good agent score: {G}{good_result['score']:.4f}{RST}  {'✓' if good_result['pass'] else '✗'}")
print(f"  Bad agent score:  {R}{bad_result['score']:.4f}{RST}  {'✓' if bad_result['pass'] else '✗'}")
diff = good_result['score'] - bad_result['score']
print(f"  Δ difference:     {BOLD}{diff:+.4f}{RST}")
print(f"\n  {DIM}Reward engine is working — good agent scores {diff:.0%} higher.{RST}\n")
=======
from environment import ElaraEnv
from models import Action

env = ElaraEnv()
obs = env.reset()
print(obs.model_dump())

obs, reward, done, info = env.step(
    Action(
        action_type="email",
        target_lead_id="L-001",
        subject="Requesting your documents",
        content="Hi Arun, sharing the documents request as discussed.",
    )
)

print(obs.model_dump())
print(reward, done, info)
>>>>>>> 8cffedc4c8dc68dd752a6a32295fc5d3f86bccf5
