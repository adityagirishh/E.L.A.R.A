#!/usr/bin/env python3
"""
run.py — E.L.A.R.A. Full Smoke Test
Runs all 3 tasks (easy / medium / hard) with good + bad agents.
Usage:  python run.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from environment import ElaraEnv
from models import Action, EpisodeState
from grader import grade

# ── terminal colours ─────────────────────────────────────────────────────────
G="\033[92m"; Y="\033[93m"; R="\033[91m"; B="\033[94m"; C="\033[96m"
W="\033[97m"; DIM="\033[2m"; RST="\033[0m"; BOLD="\033[1m"

def sep(c="─",n=64): print(DIM+c*n+RST)
def hdr(t): print(); sep("═"); print(f"{BOLD}{C}  {t}{RST}"); sep("═")
def sub(t): print(f"\n{BOLD}{W}  ▸ {t}{RST}")

def col_score(s):
    return (G if s>=0.80 else Y if s>=0.60 else R)

def print_obs(obs):
    d = obs.model_dump()
    sep()
    print(f"  {BOLD}Lead:{RST}   {W}{d['lead_name']}{RST} · {d['company']} · {d['role']}")
    stg_col = G if d['lead_stage'] not in ('new','closed_lost') else Y
    print(f"  {BOLD}Stage:{RST}  {stg_col}{d['lead_stage']}{RST}  ·  sentiment={d['sentiment']}  ·  consent={G if d['consent'] else R}{d['consent']}{RST}")
    print(f"  {BOLD}Channel:{RST} last={d['last_contact_channel']}  preferred={C}{d['preferred_channel']}{RST}  days_since={d['days_since_last_contact']}  followup_due={d['next_followup_due']}d")
    print(f"  {BOLD}Docs:{RST}   {''+R+'PENDING'+RST if d['documents_pending'] else G+'ok'+RST}  ·  objections: {d['objections'] or 'none'}")
    print(f"  {BOLD}Budget:{RST} {d['policy_constraints']['max_steps_remaining']} steps left")
    if d['task_hint']:
        print(f"  {DIM}Hint: {d['task_hint']}{RST}")
    sep()

def print_step(n, action, reward, info_d):
    act_col = G if reward > 0 else R
    print(f"\n  {BOLD}Step {n}{RST}  {act_col}{action.action_type.upper()}{RST} → {action.target_lead_id}" +
          (f"  [{DIM}{action.goal}{RST}]" if action.goal else ""))
    if action.body:
        preview = action.body[:90]+("…" if len(action.body)>90 else "")
        print(f"  {DIM}  \"{preview}\"{RST}")
    stage_arrow = f"{Y}{info_d['stage_before']}{RST} → {G}{info_d['stage_after']}{RST}"
    rwd_col = G if reward>0 else R
    print(f"  Reward: {rwd_col}{BOLD}{reward:+.3f}{RST}  ·  {stage_arrow}  ·  running={BOLD}{info_d['total_reward']:+.3f}{RST}")
    bd = {k:v for k,v in info_d.get("reward_breakdown",{}).items() if v["reward"]!=0}
    if bd:
        for k,v in bd.items():
            c2 = G if v["reward"]>0 else R
            print(f"    {c2}{v['reward']:+.2f}{RST}  {k:<20} {DIM}{v['reason']}{RST}")

def print_grade(result):
    sc   = result["score"]
    col  = col_score(sc)
    flag = "✓ PASS" if result["pass"] else "✗ FAIL"
    print(f"\n  {BOLD}Score: {col}{sc:.4f}{RST}{BOLD} / 1.0  {col}{flag}{RST}")
    print(f"  Task completed: {G if result['task_completed'] else R}{result['task_completed']}{RST}  ·  "
          f"Stage: {Y}{result['final_stage']}{RST}  ·  Steps: {result['steps_used']}/{result['max_steps']}\n")
    sep()
    print(f"  {'Dimension':<24} {'Score':>6}  {'Wt':>4}  {'Contribution':>12}  Reason")
    sep()
    for dim, data in result["dimensions"].items():
        s=data["score"]; w=data["weight"]
        c2=col_score(s)
        print(f"  {dim:<24} {c2}{s:>6.2f}{RST}  {w:>4.0%}  {s*w:>12.4f}  {DIM}{data['reason'][:48]}{RST}")
    sep()
    print(f"  {'TOTAL':<24} {col}{sc:>6.4f}{RST}  100%  {sc:>12.4f}")

def run(label, task_id, actions):
    hdr(label)
    env = ElaraEnv()
    obs = env.reset(task_id=task_id)
    sub("Initial observation")
    print_obs(obs)
    sub("Executing actions")
    for i, action in enumerate(actions, 1):
        obs, reward, done, info_d = env.step(action)
        print_step(i, action, reward, info_d)
        if done and i < len(actions):
            print(f"\n  {Y}Episode ended at step {i} — remaining actions skipped{RST}")
            break
    sub("Grader")
    result = grade(EpisodeState(**env.state()))
    print_grade(result)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# EASY TASK
# ═════════════════════════════════════════════════════════════════════════════

easy_good = run("EASY · Good Agent — First-Touch Outreach", "easy", [
    Action(
        action_type="send_email",
        target_lead_id="L-001",
        subject="Intro — E.L.A.R.A. for NovaTech",
        body=(
            "Hi Arun, hope things are going well at NovaTech Solutions. "
            "I wanted to reach out about E.L.A.R.A. — our platform that cuts "
            "lead response time by 60% and unifies email, calls and messages. "
            "Given your role in sales, this could save your team real hours weekly. "
            "Happy to set up a quick 20-min demo — does this week work?"
        ),
        goal="intro",
        priority="high",
    ),
])

easy_bad = run("EASY · Bad Agent — Wrong channel, no personalisation", "easy", [
    Action(
        action_type="make_call",          # wrong: Arun prefers email
        target_lead_id="L-001",
        body="",                          # empty — no personalisation
        goal="",
    ),
    Action(
        action_type="make_call",          # duplicate channel
        target_lead_id="L-001",
        body="just following up",
        goal="",
    ),
    Action(
        action_type="wait",               # useless final action
        target_lead_id="L-001",
    ),
])


# ═════════════════════════════════════════════════════════════════════════════
# MEDIUM TASK
# ═════════════════════════════════════════════════════════════════════════════

medium_good = run("MEDIUM · Good Agent — Follow-Up & Doc Request", "medium", [
    Action(
        action_type="send_email",
        target_lead_id="L-004",
        subject="RE: Pricing + next steps for SwiftOps",
        body=(
            "Hi Priya, thanks for your reply about pricing. "
            "Our ROI calculator shows teams like SwiftOps typically save 8h/week per rep — "
            "at our ₹2,999/seat plan that's a 10x return in 3 months. "
            "Happy to walk through the numbers on a call. "
            "In the meantime, could you share the signed NDA and your current tooling overview? "
            "That'll help me tailor the proposal specifically for your team."
        ),
        goal="handle_objection_and_request_docs",
        priority="high",
    ),
    Action(
        action_type="request_documents",
        target_lead_id="L-004",
        subject="Documents for SwiftOps proposal",
        body=(
            "Hi Priya, just circling back on the documents — "
            "NDA and tooling overview when you get a chance. "
            "Once I have those I can get the proposal to you within 24 hours."
        ),
        goal="get_documents",
    ),
    Action(
        action_type="update_crm",
        target_lead_id="L-004",
        goal="log_interaction",
        metadata={"note": "Handled pricing objection with ROI data. Doc request sent x2. Lead in awaiting_docs stage."},
    ),
])

medium_bad = run("MEDIUM · Bad Agent — Ignores objection, wrong channel, no CRM", "medium", [
    Action(
        action_type="make_call",          # L-004 prefers email
        target_lead_id="L-004",
        body="Hi, just wanted to check in.",
        goal="",
    ),
    Action(
        action_type="wait",
        target_lead_id="L-004",
    ),
    Action(
        action_type="wait",
        target_lead_id="L-004",
    ),
    Action(
        action_type="wait",
        target_lead_id="L-004",
    ),
])


# ═════════════════════════════════════════════════════════════════════════════
# HARD TASK
# ═════════════════════════════════════════════════════════════════════════════

hard_good = run("HARD · Good Agent — Multi-Channel Orchestration to Close", "hard", [
    # Proposal is 7d old, L-007 prefers calls — start with a call nudge
    Action(
        action_type="make_call",
        target_lead_id="L-007",
        body=(
            "Hi Rajan, just following up on the proposal I sent over last week. "
            "Wanted to check if you had any questions and whether the team had a chance to review it. "
            "We're excited about the potential fit with Apex Ventures."
        ),
        goal="proposal_followup",
        priority="high",
    ),
    # Docs still pending — request via email
    Action(
        action_type="request_documents",
        target_lead_id="L-007",
        subject="Documents needed — Apex Ventures",
        body=(
            "Hi Rajan, great speaking just now. "
            "To finalise the agreement I'll need the signed NDA and your procurement form. "
            "Could you share those at your earliest convenience? "
            "I can have the final contract ready within 24 hours of receiving them."
        ),
        goal="get_documents",
    ),
    # Send a message for quick back-and-forth
    Action(
        action_type="send_message",
        target_lead_id="L-007",
        body="Hi Rajan — just sent the docs request via email. Let me know if you need any changes to the proposal terms. Happy to hop on another call if helpful.",
        goal="keep_warm",
    ),
    # Update CRM
    Action(
        action_type="update_crm",
        target_lead_id="L-007",
        goal="log_interaction",
        metadata={"note": "Follow-up call done. Doc request emailed. Message sent. Lead is hot — docs pending. Target close this week."},
    ),
    # Close
    Action(
        action_type="make_call",
        target_lead_id="L-007",
        body="Hi Rajan, received your documents — everything looks good. Ready to move forward?",
        goal="close_deal",
        metadata={"docs_received": True},
    ),
])

hard_bad = run("HARD · Bad Agent — Spams same channel, ignores docs, no close", "hard", [
    Action(action_type="send_email", target_lead_id="L-007",
           body="Hi following up.", goal=""),
    Action(action_type="send_email", target_lead_id="L-007",  # duplicate channel
           body="Hi again.", goal=""),
    Action(action_type="send_email", target_lead_id="L-007",  # triple duplicate
           body="Just checking in.", goal=""),
    Action(action_type="wait", target_lead_id="L-007"),
    Action(action_type="wait", target_lead_id="L-007"),
    Action(action_type="wait", target_lead_id="L-007"),
])


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════════════

hdr("FULL RESULTS SUMMARY")
results = [
    ("Easy   · Good", easy_good),
    ("Easy   · Bad ", easy_bad),
    ("Medium · Good", medium_good),
    ("Medium · Bad ", medium_bad),
    ("Hard   · Good", hard_good),
    ("Hard   · Bad ", hard_bad),
]
sep()
print(f"  {'Agent':<20} {'Score':>7}  {'Pass':>5}  {'Stage':<16}  {'Steps':>6}  {'Completed'}")
sep()
for label, r in results:
    col = col_score(r['score'])
    pf  = f"{G}✓{RST}" if r['pass'] else f"{R}✗{RST}"
    tc  = f"{G}yes{RST}" if r['task_completed'] else f"{R}no{RST}"
    print(f"  {label:<20} {col}{r['score']:>7.4f}{RST}  {pf}       {Y}{r['final_stage']:<16}{RST}  {r['steps_used']:>2}/{r['max_steps']:<3}  {tc}")
sep()

# Δ gaps
gaps = [
    ("Easy",   easy_good['score']  - easy_bad['score']),
    ("Medium", medium_good['score']- medium_bad['score']),
    ("Hard",   hard_good['score']  - hard_bad['score']),
]
print(f"\n  Good vs Bad gaps:")
for name, delta in gaps:
    col = G if delta >= 0.20 else Y if delta >= 0.10 else R
    print(f"  {name:<8}  {col}{delta:+.4f}{RST}  {'✓ signal strong' if delta>=0.20 else '⚠ signal weak — needs tuning'}")
print()