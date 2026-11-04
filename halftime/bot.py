#!/usr/bin/env python3
"""
Halftime Edge: college basketball halftime alerts (men's + women's D1).

At halftime of every game it:
  1. reads the pregame line (DraftKings via ESPN's free feed) and the halftime score
  2. runs a model trained on ~36,000 past games' final results to get each team's fair win chance
  3. for spots where a pregame underdog leads, compares that to real sportsbook moneylines
     (DraftKings via ESPN, or The Odds API free tier) and emails an alert if the price is good
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
MIN_DOG_PREGAME = 0.62  # watch games where the leading team was a pregame underdog of about +3 or more
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

def model_fav_prob(pre_fav, fav_margin, womens):
    m = fav_margin / 10
    x = [1, logit(pre_fav), m, womens, m * womens, (1 if fav_margin < 0 else 0) * logit(pre_fav)]
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

def norm_cdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def pregame_home_prob(league, ev, summary):
    """Best available pregame strength: the DraftKings spread, then ESPN's projection, then ESPN's tip-off win %."""
    try:
        o = ev["competitions"][0]["odds"][0]
        sp = o.get("spread")
        if sp not in (None, ""):
            return norm_cdf(-float(sp) / MODEL["spread_sd"][league]), f"DK spread {float(sp):+g}"
    except Exception:
        pass
    pred = (summary or {}).get("predictor") or {}
    try: return float(pred["homeTeam"]["gameProjection"]) / 100, "ESPN projection"
    except Exception: pass
    wps = (summary or {}).get("winprobability") or []
    if wps: return float(wps[0]["homeWinPercentage"]), "ESPN tip-off win %"
    return None, None

def game_inputs(league, ev, summary, score=None):
    home, away = teams(ev)
    pre_home, src = pregame_home_prob(league, ev, summary)
    if pre_home is None: return None, "no pregame line"
    hs, as_ = score if score else (float(home.get("score") or 0), float(away.get("score") or 0))
    return dict(home=name(home), away=name(away), pre_home=pre_home, pre_src=src, home_score=hs, away_score=as_), None

def evaluate(league, gi):
    """Spot = a pregame underdog (about +3 or more) leading at halftime. Returns the underdog side with model odds."""
    home_fav = gi["pre_home"] > 0.5
    pre_fav = gi["pre_home"] if home_fav else 1 - gi["pre_home"]
    margin = (gi["home_score"] - gi["away_score"]) * (1 if home_fav else -1)
    if pre_fav < MIN_DOG_PREGAME or margin >= 0: return None
    m_fav = model_fav_prob(pre_fav, margin, LEAGUES[league]["womens"])
    dog_home = not home_fav
    return dict(side="dog", team=gi["home"] if dog_home else gi["away"], opp=gi["away"] if dog_home else gi["home"],
                is_home=dog_home, model_p=1 - m_fav, pre_p=1 - pre_fav, lead=-margin, pre_src=gi["pre_src"])

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
FIELDS = ["logged_et", "league", "game_id", "date", "team", "opp", "side", "lead", "pre_p", "book_p", "model_p",
          "price", "book", "ev", "stake_pct", "bet_at_or_better", "status", "final_team", "final_opp", "result", "profit_units"]

def read_log():
    if not os.path.exists(LOG): return []
    return list(csv.DictReader(open(LOG)))

def write_log(rows):
    with open(LOG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); [w.writerow({k: r.get(k, "") for k in FIELDS}) for r in rows]

def send_ntfy(subject, body):
    """Free alerts via ntfy.sh: push to the ntfy phone app, plus an email copy if ALERT_TO is set."""
    topic, to = os.environ.get("NTFY_TOPIC"), os.environ.get("ALERT_TO")
    if not topic: return False
    headers = {"Title": subject.encode("ascii", "ignore").decode(), "Priority": "high", "Tags": "basketball"}
    if to: headers["Email"] = to
    try:
        req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=body.encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=20) as r:
            print(f"  alert sent via ntfy ({r.status}){' + email' if to else ''}: {subject}")
        return True
    except Exception as e:
        print(f"  ntfy failed: {e}", file=sys.stderr); return False

def send_email(subject, body):
    user, pw, to = os.environ.get("GMAIL_USER"), os.environ.get("GMAIL_APP_PASSWORD"), os.environ.get("ALERT_TO") or os.environ.get("GMAIL_USER")
    if not (user and pw):
        if not send_ntfy(subject, body): print(f"  (alerts not configured) {subject}")
        return
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
             f"Model win chance: {c['model_p']*100:.0f}%   ·   before the game: {c['pre_p']*100:.0f}% ({c['pre_src']})"]
    if price:
        lines += [f"Best price found: {price['price']} at {price['book']} (book says {implied(price['price'])*100:.0f}%)  →  edge {ev_val*100:+.1f}%",
                  f"Suggested stake: {stake*100:.1f}% of your bankroll"]
    else:
        lines += ["No live price found automatically. Check your sportsbook app."]
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
            if not price and not (0.30 <= c["model_p"] <= BAND[1]):
                print(f"  {c['team']}: no price and model {c['model_p']*100:.0f}% is outside the bettable range, skip"); continue
            if price:
                dec = american_to_decimal(price["price"]); ev_val = c["model_p"] * dec - 1; stake = kelly_stake(c["model_p"], dec)
                if not (BAND[0] <= implied(price["price"]) <= BAND[1]):
                    print(f"  {c['team']}: price {price['price']} outside realistic range, skip"); continue
                if ev_val < MIN_EV:
                    print(f"  {c['team']}: price {price['price']} too short (EV {ev_val*100:+.1f}%), no alert"); continue
            body = alert_text(league, gi, c, price, ev_val, stake)
            subj = f"HALFTIME: {c['team']} ML" + (f" {price['price']} @ {price['book']}" if price else f" (bet at {min_price(c['model_p'])} or better)")
            print("\n" + body + "\n")
            if not dry:
                send_email(subj, body)
                rows.append(dict(logged_et=datetime.now(ET).strftime("%Y-%m-%d %H:%M"), league=league, game_id=gid,
                                 date=ev["date"][:10], team=c["team"], opp=c["opp"], side=c["side"], lead=int(c["lead"]),
                                 pre_p=round(c["pre_p"], 3), book_p=round(implied(price["price"]), 3) if price else "", model_p=round(c["model_p"], 3),
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
             "| Date | League | Bet | Lead at half | Model | Book | Price | Final | Result | Units |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: x["logged_et"], reverse=True):
        lines.append(f"| {r['date']} | {LEAGUES[r['league']]['label']} | {r['team']} ML vs {r['opp']} | {r['lead']:>3} | "
                     f"{float(r['model_p'])*100:.0f}% | {(str(round(float(r['book_p'])*100)) + '%') if r.get('book_p') else '—'} | {r['price'] or r['bet_at_or_better'] + ' (min)'} | "
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
            gi, why = game_inputs(league, ev, summ, score=(float(last.get("homeScore") or 0), float(last.get("awayScore") or 0)))
            if gi is None: continue
            ok += 1
            c = evaluate(league, gi)
            if c:
                cands += 1
                print(f"   SPOT {gi['away']} {int(gi['away_score'])} @ {gi['home']} {int(gi['home_score'])} -> {c['team']} "
                      f"model {c['model_p']*100:.0f}% (pregame {c['pre_p']*100:.0f}% via {c['pre_src']}), bet at {min_price(c['model_p'])} or better")
        print(f"   parsed halftime for {ok} games, {cands} spots")
        if evs:
            print("   ESPN odds block sample:", json.dumps(evs[0]["competitions"][0].get("odds", [{}])[0].get("moneyline", {}))[:300])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true"); ap.add_argument("--grade", action="store_true")
    ap.add_argument("--selftest"); ap.add_argument("--dry", action="store_true"); ap.add_argument("--testalert", action="store_true")
    a = ap.parse_args()
    if a.selftest: selftest(a.selftest); return
    if a.testalert:
        print("NTFY_TOPIC set:", bool(os.environ.get("NTFY_TOPIC")), "| ALERT_TO set:", bool(os.environ.get("ALERT_TO")),
              "| GMAIL set:", bool(os.environ.get("GMAIL_USER")))
        send_email("Halftime Edge test alert", "If you got this, alerts are working.\n\nReal alerts look like:\nBET: Iowa moneyline\nModel 58% · best price +120 at FanDuel · edge +28%\nOnly bet at -131 or better.")
        return
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
