#!/usr/bin/env python3
"""
run.py — E.L.A.R.A. Full Smoke Test
Runs all 5 tasks (easy / medium / hard / escalation / consent) with good + bad agents.
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
    if d.get('active_leads') and len(d['active_leads']) > 1:
        print(f"  {BOLD}Active leads:{RST} {', '.join(l['lead_id']+' ('+l['lead_stage']+')' for l in d['active_leads'])}")
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
    if info_d.get("lead_response"):
        resp = info_d["lead_response"]
        print(f"    {B}💬 Lead says:{RST} \"{resp.get('text', '')[:80]}\"")

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
        print(f"  {dim:<24} {c2}{s:>6.2f}{RST}  {w:>4.0%}  {s*w:>12.4f}  {DIM}{data['reason'][:60]}{RST}")
    sep()
    print(f"  {'TOTAL':<24} {col}{sc:>6.4f}{RST}  100%  {sc:>12.4f}")

def run(label, task_id, actions, seed=0):
    hdr(label)
    env = ElaraEnv()
    obs = env.reset(task_id=task_id, seed=seed)
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
        action_type="make_call",
        target_lead_id="L-001",
        body="",
        goal="",
    ),
    Action(
        action_type="make_call",
        target_lead_id="L-001",
        body="just following up",
        goal="",
    ),
    Action(
        action_type="wait",
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
            "Happy to walk through the numbers on a call."
        ),
        goal="handle_objection",
        priority="high",
    ),
    Action(
        action_type="request_documents",
        target_lead_id="L-004",
        subject="Documents for SwiftOps proposal",
        body=(
            "Hi Priya, circling back on documents — "
            "NDA and tooling overview when you get a chance. "
            "Once I have those I can get the proposal to you within 24 hours."
        ),
        goal="get_documents",
    ),
    Action(
        action_type="update_crm",
        target_lead_id="L-004",
        goal="log_interaction",
        metadata={"note": "Handled pricing objection with ROI data. Doc request sent. Lead in awaiting_docs."},
    ),
])

medium_bad = run("MEDIUM · Bad Agent — Ignores objection, wrong channel, no CRM", "medium", [
    Action(action_type="make_call", target_lead_id="L-004", body="Hi, just wanted to check in.", goal=""),
    Action(action_type="wait", target_lead_id="L-004"),
    Action(action_type="wait", target_lead_id="L-004"),
    Action(action_type="wait", target_lead_id="L-004"),
])


# ═════════════════════════════════════════════════════════════════════════════
# HARD TASK (multi-lead: L-007 + L-008)
# ═════════════════════════════════════════════════════════════════════════════

hard_good = run("HARD · Good Agent — Multi-Lead Orchestration", "hard", [
    # L-007: proposal 7d old, prefers calls — start with a call
    Action(
        action_type="make_call",
        target_lead_id="L-007",
        body=(
            "Hi Rajan, following up on the proposal I sent last week for Apex Ventures. "
            "Wanted to check if your team had a chance to review it. "
            "We're excited about the potential fit — E.L.A.R.A. cuts lead response time by 60%."
        ),
        goal="proposal_followup",
        priority="high",
    ),
    # L-008: negotiating, prefers messages, contract term objections
    Action(
        action_type="send_message",
        target_lead_id="L-008",
        body=(
            "Hi Sunita, following up on our negotiation for BlueSky Tech. "
            "Regarding the contract terms concern — we can offer flexible payment terms "
            "and adjust the clause structure to fit your procurement process. "
            "Happy to discuss specifics."
        ),
        goal="handle_objection",
    ),
    # L-007: request docs via email
    Action(
        action_type="request_documents",
        target_lead_id="L-007",
        subject="Documents needed — Apex Ventures",
        body=(
            "Hi Rajan, great speaking just now. "
            "To finalise the agreement I'll need the signed NDA and procurement form."
        ),
        goal="get_documents",
    ),
    # L-008: follow up via email (different channel from previous message)
    Action(
        action_type="send_email",
        target_lead_id="L-008",
        subject="Updated terms for BlueSky Tech",
        body=(
            "Hi Sunita, as discussed, I've revised the contract terms for BlueSky Tech. "
            "Attached are the updated payment schedule and adjusted clause. "
            "Let me know if this works for your procurement team."
        ),
        goal="send_revised_terms",
    ),
    # CRM update for both
    Action(
        action_type="update_crm",
        target_lead_id="L-007",
        goal="log_interaction",
        metadata={"note": "L-007: Follow-up call done, docs requested. L-008: Handled contract terms objection via message + email."},
    ),
])

hard_bad = run("HARD · Bad Agent — Ignores L-008, spams same channel", "hard", [
    Action(action_type="send_email", target_lead_id="L-007", body="Hi following up.", goal=""),
    Action(action_type="send_email", target_lead_id="L-007", body="Hi again.", goal=""),
    Action(action_type="send_email", target_lead_id="L-007", body="Just checking in.", goal=""),
    Action(action_type="wait", target_lead_id="L-007"),
    Action(action_type="wait", target_lead_id="L-007"),
    Action(action_type="wait", target_lead_id="L-007"),
    Action(action_type="wait", target_lead_id="L-007"),
])


# ═════════════════════════════════════════════════════════════════════════════
# ESCALATION TASK
# ═════════════════════════════════════════════════════════════════════════════

escalation_good = run("ESCALATION · Good Agent — Recognize & Escalate", "escalation", [
    # Recognise docs have been requested twice — escalate instead of requesting again
    Action(
        action_type="send_email",
        target_lead_id="L-014",
        body=(
            "Hi Lavanya, I noticed the documents for ClearPath Solutions have been pending "
            "for over a week. I understand these things can get stuck in legal. "
            "I'm going to loop in our senior rep to help move things along."
        ),
        goal="context_before_escalation",
    ),
    Action(
        action_type="escalate",
        target_lead_id="L-014",
        body="Escalating L-014 — docs requested twice with no response, 8 days elapsed.",
        goal="escalate_stuck_deal",
        metadata={"reason": "docs_stuck_in_legal", "prior_requests": 2},
    ),
    Action(
        action_type="update_crm",
        target_lead_id="L-014",
        goal="log_interaction",
        metadata={"note": "Escalated to senior rep. Docs requested 2x, no response after 8 days. Legal blockers suspected."},
    ),
])

escalation_bad = run("ESCALATION · Bad Agent — Keeps requesting docs", "escalation", [
    Action(
        action_type="request_documents",
        target_lead_id="L-014",
        body="Please send the documents.",
        goal="get_documents",
    ),
    Action(
        action_type="request_documents",
        target_lead_id="L-014",
        body="Reminder: still need those docs.",
        goal="get_documents",
    ),
    Action(action_type="wait", target_lead_id="L-014"),
    Action(action_type="wait", target_lead_id="L-014"),
])


# ═════════════════════════════════════════════════════════════════════════════
# CONSENT TASK (multi-lead: L-016, L-012, L-019 with consent trap)
# ═════════════════════════════════════════════════════════════════════════════

consent_good = run("CONSENT · Good Agent — Compliance Navigation", "consent", [
    # Contact L-016 via email (preferred)
    Action(
        action_type="send_email",
        target_lead_id="L-016",
        body=(
            "Hi Ria, thanks for attending our webinar! "
            "I wanted to reach out about E.L.A.R.A. for BrightWave — "
            "our platform cuts lead response time by 60%. "
            "Would love to set up a quick demo."
        ),
        goal="intro",
        priority="high",
    ),
    # Contact L-012 via message (preferred)
    Action(
        action_type="send_message",
        target_lead_id="L-012",
        body=(
            "Hi Nisha, saw your LinkedIn interest in Quantify AI's sales ops. "
            "E.L.A.R.A. unifies email, calls and messages in one inbox. "
            "Happy to share a quick overview — async works great."
        ),
        goal="intro",
    ),
    # Contact L-019 via email (preferred) — they WILL revoke consent after this
    Action(
        action_type="send_email",
        target_lead_id="L-019",
        body=(
            "Hi Harish, we met at the trade show — wanted to follow up about "
            "E.L.A.R.A. for BlueNorth. Our compliance-safe outreach templates "
            "might be relevant to your engineering team."
        ),
        goal="intro",
    ),
    # After step 3, L-019 revokes consent. Good agent reads the response and STOPS.
    # Instead, follow up with L-016 via call (different channel)
    Action(
        action_type="make_call",
        target_lead_id="L-016",
        body=(
            "Hi Ria, just following up on the email I sent about E.L.A.R.A. for BrightWave. "
            "Quick question — what's your current sales outreach workflow like?"
        ),
        goal="discovery",
    ),
    # CRM update
    Action(
        action_type="update_crm",
        target_lead_id="L-016",
        goal="log_interaction",
        metadata={"note": "L-016: Intro email + discovery call. L-012: Intro message sent. L-019: Opted out after initial contact — do not follow up."},
    ),
])

consent_bad = run("CONSENT · Bad Agent — Ignores consent revocation", "consent", [
    Action(action_type="send_email", target_lead_id="L-019", body="Hi there.", goal="intro"),
    # L-019 revokes consent after this step — bad agent ignores it
    Action(action_type="send_email", target_lead_id="L-019", body="Following up.", goal="followup"),
    Action(action_type="send_email", target_lead_id="L-019", body="Any update?", goal="followup"),
    Action(action_type="wait", target_lead_id="L-019"),
    Action(action_type="wait", target_lead_id="L-019"),
    Action(action_type="wait", target_lead_id="L-019"),
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
    ("Escal  · Good", escalation_good),
    ("Escal  · Bad ", escalation_bad),
    ("Consent· Good", consent_good),
    ("Consent· Bad ", consent_bad),
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

# Gaps
pairs = [
    ("Easy",      easy_good,       easy_bad),
    ("Medium",    medium_good,     medium_bad),
    ("Hard",      hard_good,       hard_bad),
    ("Escalation",escalation_good, escalation_bad),
    ("Consent",   consent_good,    consent_bad),
]
print(f"\n  Good vs Bad gaps:")
for name, good, bad in pairs:
    delta = good['score'] - bad['score']
    col = G if delta >= 0.20 else Y if delta >= 0.10 else R
    print(f"  {name:<12}  {col}{delta:+.4f}{RST}  {'✓ signal strong' if delta>=0.20 else '⚠ signal weak'}")
print()
