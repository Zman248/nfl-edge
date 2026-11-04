#!/usr/bin/env python3
"""
Halftime Edge: college basketball halftime alerts (men's + women's D1).

At halftime of every game it:
  1. reads the pregame win chance and ESPN's live win chance (free ESPN feeds)
  2. runs a model trained on ~36,000 past games to get the fair win chance for each team
  3. if the model disagrees with the market by enough, checks real sportsbook prices
     (The Odds API, free tier, only when there's a candidate) and emails an alert
  4. logs every alert as a paper bet and grades it after the game

Backtest summary: when a +5 or bigger pregame underdog leads at halftime, the market
overrates the favorite's comeback. The model generalizes that.

Usage:
  python3 bot.py                 # watch live games until tonight's slate is done (max ~5h45m)
  python3 bot.py --once          # one pass, then exit
  python3 bot.py --grade         # grade past alerts only
  python3 bot.py --selftest 20250115   # replay a past date's halftimes (no emails) to check the feeds
"""
import argparse, csv, json, math, os, smtplib, sys, time, urllib.request, urllib.parse
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "alerts.csv")
RECORD = os.path.join(HERE, "RECORD.md")
MODEL = json.load(open(os.path.join(HERE, "model.json")))
ET = ZoneInfo("America/New_York")

LEAGUES = {
    "mens":   dict(espn="mens-college-basketball",   odds="basketball_ncaab",  half_period=1, womens=0, label="Men's"),
    "womens": dict(espn="womens-college-basketball", odds="basketball_wncaab", half_period=2, womens=1, label="Women's"),
}
# --- rules (from the backtest) ---
CANDIDATE_EDGE = 0.05   # model must beat ESPN's live odds by 5+ points before we spend an odds lookup
MIN_EV = 0.03           # bet only if expected value at the real price is +3% or better
BAND = (0.25, 0.75)     # only realistic prices (about +300 to -300)
KELLY_FRACTION = 0.25   # suggested stake = quarter Kelly
MAX_STAKE = 0.03        # never suggest more than 3% of bankroll on one bet
RUN_SECONDS = 5 * 3600 + 45 * 60

UA = {"User-Agent": "halftime-edge/1.0"}

def get_json(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) as r:
                return json.load(r)
        except Exception as e:
            if i == tries - 1:
                print(f"  fetch failed {url[:90]}: {e}", file=sys.stderr); return None
            time.sleep(2 + 3 * i)

def espn(league, path, **q):
    q = urllib.parse.urlencode(q)
    return get_json(f"https://site.api.espn.com/apis/site/v2/sports/basketball/{LEAGUES[league]['espn']}/{path}?{q}")

# ---------------------------------------------------------------- odds math
def logit(p):
    p = min(max(p, 1e-3), 1 - 1e-3); return math.log(p / (1 - p))

def model_fav_prob(espn_fav_wp, pre_fav, fav_margin, womens):
    x = [1, logit(espn_fav_wp), logit(pre_fav), fav_margin / 10, womens, (1 if fav_margin < 0 else 0) * logit(pre_fav)]
    z = sum(a * b for a, b in zip(MODEL["weights"], x))
    return 1 / (1 + math.exp(-z))

def american_to_decimal(a):
    a = float(a); return 1 + (a / 100 if a > 0 else 100 / -a)

def decimal_to_american(d):
    return f"+{round((d - 1) * 100)}" if d >= 2 else f"{round(-100 / (d - 1))}"

def implied(a): return 1 / american_to_decimal(a)

def min_price(p, ev=MIN_EV):
    """Worst odds you should accept for win chance p and target EV."""
    return decimal_to_american((1 + ev) / p)

def kelly_stake(p, dec):
    f = (p * dec - 1) / (dec - 1)
    return max(0.0, min(MAX_STAKE, f * KELLY_FRACTION))

# ---------------------------------------------------------------- reading a game
def teams(ev):
    c = ev["competitions"][0]
    t = {x["homeAway"]: x for x in c["competitors"]}
    return t["home"], t["away"]

def name(x): return x["team"].get("displayName") or x["team"].get("name")

def game_inputs(league, ev, summary, wp_at_play=None):
    """Pregame win chance, ESPN live win chance at halftime, halftime score."""
    home, away = teams(ev)
    pred = (summary or {}).get("predictor") or {}
    try:
        pre_home = float(pred["homeTeam"]["gameProjection"]) / 100
    except Exception:
        return None, "no pregame projection"
    wps = (summary or {}).get("winprobability") or []
    if not wps: return None, "no live win probability"
    if wp_at_play is not None:
        cand = [w for w in wps if str(w.get("playId")) == str(wp_at_play)]
        if not cand: return None, "halftime play not found"
        w = cand[0]
    else:
        w = wps[-1]
    home_wp = float(w["homeWinPercentage"])
    hs, as_ = float(home.get("score") or 0), float(away.get("score") or 0)
    return dict(home=name(home), away=name(away), home_id=home["team"]["id"], away_id=away["team"]["id"],
                pre_home=pre_home, home_wp=home_wp, home_score=hs, away_score=as_), None

def evaluate(league, gi):
    """Model fair chance for both teams vs ESPN live odds. Returns candidate side or None."""
    home_fav = gi["pre_home"] > 0.5
    pre_fav = gi["pre_home"] if home_fav else 1 - gi["pre_home"]
    espn_fav = gi["home_wp"] if home_fav else 1 - gi["home_wp"]
    margin = (gi["home_score"] - gi["away_score"]) * (1 if home_fav else -1)
    m_fav = model_fav_prob(espn_fav, pre_fav, margin, LEAGUES[league]["womens"])
    sides = [("fav", m_fav, espn_fav), ("dog", 1 - m_fav, 1 - espn_fav)]
    best = None
    for side, mp, ep in sides:
        if not (BAND[0] <= ep <= BAND[1]): continue
        if mp - ep >= CANDIDATE_EDGE and (best is None or mp - ep > best["model_p"] - best["espn_p"]):
            team_is_home = (side == "fav") == home_fav
            best = dict(side=side, team=gi["home"] if team_is_home else gi["away"],
                        opp=gi["away"] if team_is_home else gi["home"], is_home=team_is_home,
                        model_p=mp, espn_p=ep, pre_p=pre_fav if side == "fav" else 1 - pre_fav,
                        lead=margin if side == "fav" else -margin)
    return best

# ---------------------------------------------------------------- real prices
_odds_cache = {}
def odds_api_prices(league, team, opp):
    key = os.environ.get("ODDS_API_KEY")
    if not key: return None
    sk = LEAGUES[league]["odds"]
    now = time.time()
    if sk not in _odds_cache or now - _odds_cache[sk][0] > 90:
        data = get_json(f"https://api.the-odds-api.com/v4/sports/{sk}/odds?apiKey={key}&regions=us&markets=h2h&oddsFormat=american", tries=2)
        _odds_cache[sk] = (now, data or [])
    def norm(s): return "".join(ch for ch in s.lower() if ch.isalnum())
    best = None
    for e in _odds_cache[sk][1]:
        names = [e.get("home_team", ""), e.get("away_team", "")]
        hit = lambda t: any(norm(t).startswith(norm(n)[:8]) or norm(n).startswith(norm(t)[:8]) for n in names)
        if not (hit(team) and hit(opp)): continue
        for bk in e.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m.get("key") != "h2h": continue
                for oc in m.get("outcomes", []):
                    if norm(oc["name"])[:8] in norm(team) or norm(team)[:8] in norm(oc["name"]):
                        if best is None or american_to_decimal(oc["price"]) > american_to_decimal(best["price"]):
                            best = dict(price=oc["price"], book=bk.get("title", bk.get("key")))
    return best

def espn_dk_price(ev, is_home):
    """DraftKings moneyline from ESPN's scoreboard, if ESPN is showing a live one."""
    try:
        o = ev["competitions"][0]["odds"][0]["moneyline"]["home" if is_home else "away"]
        for k in ("current", "live"):
            if k in o and o[k].get("odds") not in (None, "", "OFF"):
                return dict(price=o[k]["odds"], book="DraftKings (via ESPN)")
    except Exception:
        pass
    return None

# ---------------------------------------------------------------- log / email
FIELDS = ["logged_et", "league", "game_id", "date", "team", "opp", "side", "lead", "pre_p", "espn_p", "model_p",
          "price", "book", "ev", "stake_pct", "bet_at_or_better", "status", "final_team", "final_opp", "result", "profit_units"]

def read_log():
    if not os.path.exists(LOG): return []
    return list(csv.DictReader(open(LOG)))

def write_log(rows):
    with open(LOG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); [w.writerow({k: r.get(k, "") for k in FIELDS}) for r in rows]

def send_email(subject, body):
    user, pw, to = os.environ.get("GMAIL_USER"), os.environ.get("GMAIL_APP_PASSWORD"), os.environ.get("ALERT_TO") or os.environ.get("GMAIL_USER")
    if not (user and pw):
        print(f"  (email not configured) {subject}"); return
    msg = MIMEText(body); msg["Subject"] = subject; msg["From"] = user; msg["To"] = to
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as s:
            s.login(user, pw); s.sendmail(user, [to], msg.as_string())
        print(f"  emailed: {subject}")
    except Exception as e:
        print(f"  email failed: {e}", file=sys.stderr)

def alert_text(league, gi, c, price, ev_val, stake):
    lines = [f"{LEAGUES[league]['label']} college hoops · HALFTIME",
             f"{gi['away']} {int(gi['away_score'])} @ {gi['home']} {int(gi['home_score'])}", "",
             f"BET: {c['team']} moneyline (to win the game)",
             f"Model win chance: {c['model_p']*100:.0f}%   ·   ESPN live odds: {c['espn_p']*100:.0f}%   ·   before the game: {c['pre_p']*100:.0f}%"]
    if price:
        lines += [f"Best price found: {price['price']} at {price['book']}  →  edge {ev_val*100:+.1f}%",
                  f"Suggested stake: {stake*100:.1f}% of your bankroll"]
    lines += [f"Only bet at {min_price(c['model_p'])} or better. Worse than that, skip it.", "",
              "Paper-tracking: logged and graded automatically after the game."]
    return "\n".join(lines)

# ---------------------------------------------------------------- one pass over live games
def scan(seen, dry=False):
    rows = read_log(); logged = {r["game_id"] for r in rows}
    n_half = 0; active = False
    for league in LEAGUES:
        today = datetime.now(ET)
        events = []
        for d in {today.strftime("%Y%m%d"), (today - timedelta(hours=6)).strftime("%Y%m%d")}:
            sb = espn(league, "scoreboard", dates=d, groups=50, limit=500)
            events += (sb or {}).get("events", [])
        for ev in events:
            st = ev["status"]["type"]
            if st.get("state") in ("pre", "in"): active = True
            if st.get("name") != "STATUS_HALFTIME": continue
            n_half += 1
            gid = ev["id"]
            if gid in seen or gid in logged: continue
            seen.add(gid)
            summ = espn(league, "summary", event=gid)
            gi, why = game_inputs(league, ev, summ)
            if gi is None: print(f"  skip {gid}: {why}"); continue
            c = evaluate(league, gi)
            if not c: continue
            price = odds_api_prices(league, c["team"], c["opp"]) or espn_dk_price(ev, c["is_home"])
            ev_val = stake = None
            if price:
                dec = american_to_decimal(price["price"]); ev_val = c["model_p"] * dec - 1; stake = kelly_stake(c["model_p"], dec)
                if ev_val < MIN_EV:
                    print(f"  {c['team']}: price {price['price']} too short (EV {ev_val*100:+.1f}%), no alert"); continue
            body = alert_text(league, gi, c, price, ev_val, stake)
            subj = f"HALFTIME: {c['team']} ML" + (f" {price['price']} @ {price['book']}" if price else f" (bet at {min_price(c['model_p'])} or better)")
            print("\n" + body + "\n")
            if not dry:
                send_email(subj, body)
                rows.append(dict(logged_et=datetime.now(ET).strftime("%Y-%m-%d %H:%M"), league=league, game_id=gid,
                                 date=ev["date"][:10], team=c["team"], opp=c["opp"], side=c["side"], lead=int(c["lead"]),
                                 pre_p=round(c["pre_p"], 3), espn_p=round(c["espn_p"], 3), model_p=round(c["model_p"], 3),
                                 price=price["price"] if price else "", book=price["book"] if price else "",
                                 ev=round(ev_val, 4) if ev_val is not None else "", stake_pct=round(stake * 100, 2) if stake else "",
                                 bet_at_or_better=min_price(c["model_p"]), status="open"))
                write_log(rows)
    return n_half, active

# ---------------------------------------------------------------- grading
def grade():
    rows = read_log(); changed = False
    for r in rows:
        if r.get("status") != "open": continue
        summ = espn(r["league"], "summary", event=r["game_id"])
        try:
            comp = summ["header"]["competitions"][0]
            if not comp["status"]["type"]["completed"]: continue
            sc = {c["team"]["displayName"]: float(c["score"]) for c in comp["competitors"]}
        except Exception:
            continue
        mine = next((v for k, v in sc.items() if k == r["team"]), None)
        theirs = next((v for k, v in sc.items() if k != r["team"]), None)
        if mine is None or theirs is None: continue
        won = mine > theirs
        # paper profit: real price if we had one, otherwise the minimum acceptable price we told you to take
        price = r["price"] or r["bet_at_or_better"]
        dec = american_to_decimal(price)
        r.update(status="graded", final_team=int(mine), final_opp=int(theirs), result="W" if won else "L",
                 profit_units=round(dec - 1 if won else -1, 3))
        changed = True
    if changed: write_log(rows)
    g = [r for r in rows if r.get("status") == "graded"]
    w = sum(r["result"] == "W" for r in g); units = sum(float(r["profit_units"]) for r in g)
    real = [r for r in g if r["price"]]
    lines = ["# Halftime Edge: paper record", "",
             f"Updated {datetime.now(ET).strftime('%a %b %d, %I:%M %p ET')}", "",
             f"**Record: {w}–{len(g) - w}**, **{units:+.2f} units** (1 unit per bet), {len(g)} graded, "
             f"{sum(r.get('status') == 'open' for r in rows)} open.", "",
             f"Bets with a real book price: {sum(r['result']=='W' for r in real)}–{len(real)-sum(r['result']=='W' for r in real)}, "
             f"{sum(float(r['profit_units']) for r in real):+.2f} units. "
             "Bets without one are graded at the minimum price the alert said to accept.", "",
             "| Date | League | Bet | Lead at half | Model | Odds | Price | Final | Result | Units |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: x["logged_et"], reverse=True):
        lines.append(f"| {r['date']} | {LEAGUES[r['league']]['label']} | {r['team']} ML vs {r['opp']} | {r['lead']:>3} | "
                     f"{float(r['model_p'])*100:.0f}% | {float(r['espn_p'])*100:.0f}% | {r['price'] or r['bet_at_or_better'] + ' (min)'} | "
                     f"{(str(r['final_team']) + '–' + str(r['final_opp'])) if r.get('final_team') else '—'} | {r.get('result') or 'open'} | {r.get('profit_units') or ''} |")
    open(RECORD, "w").write("\n".join(lines) + "\n")
    print(f"Record: {w}-{len(g)-w}, {units:+.2f} units")

# ---------------------------------------------------------------- selftest on a past date
def selftest(date):
    """Replay halftimes from a finished date to prove the feeds and math work (no emails)."""
    for league in LEAGUES:
        sb = espn(league, "scoreboard", dates=date, groups=50, limit=500)
        evs = (sb or {}).get("events", [])
        print(f"\n== {league}: {len(evs)} games on {date}")
        ok = cands = 0
        if evs:
            s0 = espn(league, "summary", event=evs[0]["id"]) or {}
            pl = s0.get("plays") or []
            print("   DEBUG summary keys:", sorted(s0.keys()))
            print("   DEBUG plays:", len(pl), "| first play:", json.dumps(pl[0])[:400] if pl else None)
            print("   DEBUG winprobability:", len(s0.get("winprobability") or []), "| sample:", json.dumps((s0.get("winprobability") or [None])[-1])[:200])
            print("   DEBUG predictor:", json.dumps(s0.get("predictor"))[:300])
            print("   DEBUG status:", json.dumps(evs[0]["status"])[:200])
        for ev in evs[:40]:
            summ = espn(league, "summary", event=ev["id"])
            plays = (summ or {}).get("plays") or []
            hp = [p for p in plays if (p.get("period") or {}).get("number") == LEAGUES[league]["half_period"]]
            if not hp: continue
            last = hp[-1]
            home, away = teams(ev)
            # rewind the score to halftime for the replay
            home["score"], away["score"] = last.get("homeScore"), last.get("awayScore")
            gi, why = game_inputs(league, ev, summ, wp_at_play=last.get("id"))
            if gi is None: continue
            ok += 1
            c = evaluate(league, gi)
            if c:
                cands += 1
                print(f"   CANDIDATE {gi['away']} {int(gi['away_score'])} @ {gi['home']} {int(gi['home_score'])} -> {c['team']} "
                      f"model {c['model_p']*100:.0f}% vs ESPN {c['espn_p']*100:.0f}% (pregame {c['pre_p']*100:.0f}%), bet at {min_price(c['model_p'])}+")
        print(f"   parsed halftime for {ok} games, {cands} would have alerted")
        if evs:
            print("   ESPN odds block sample:", json.dumps(evs[0]["competitions"][0].get("odds", [{}])[0].get("moneyline", {}))[:300])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true"); ap.add_argument("--grade", action="store_true")
    ap.add_argument("--selftest"); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.selftest: selftest(a.selftest); return
    grade()
    if a.grade: return
    seen, start, idle = set(), time.time(), 0
    while True:
        n_half, active = scan(seen, dry=a.dry)
        print(f"{datetime.now(ET).strftime('%H:%M')} ET: {n_half} games at halftime")
        if a.once or time.time() - start > RUN_SECONDS: break
        idle = 0 if active else idle + 1
        if idle >= 3: print("No games left tonight."); break
        time.sleep(60)
    grade()

if __name__ == "__main__":
    main()
