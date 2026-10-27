#!/usr/bin/env python3
"""
NFL Edge — free-data prop research tool.

Automates the "notepad" process: team pass volume, injuries + who absorbs the
missing targets, defense weak spots (vs WR/TE/RB, middle vs outside), blitz
rates, then projects player stats and simulates over/under probabilities.

Data: nflverse (free, public). Usage:
    python3 nfl_edge.py --week 8                      # all games that week
    python3 nfl_edge.py --week 8 --game PHI-WAS       # one game
    python3 nfl_edge.py --week 8 --lines lines.csv    # compare to prop lines
    python3 nfl_edge.py --week 8 --odds-key KEY       # pull lines from The Odds API
    python3 nfl_edge.py --backtest 7                  # test accuracy on a past week

lines.csv columns: player,market,line,over_odds,under_odds
  market = rec_yds | receptions | rush_yds | pass_yds | pass_att
"""
import argparse, json, os, sys, time, math, html, urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download/"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nfl_cache")
RNG = np.random.default_rng(7)
SIMS = 20000
# Tunable settings. These are defaults; tune() re-fits them every run from finished weeks.
MATCH_W = 0.25       # how much the defense matchup moves receiving yards (0 = ignore, 1 = full)
MATCH_W_RUSH = 1.0   # same for rushing
LEAN_CONF = 0.10     # lean needs model >= 50% + this on one side
POS_SKILL = ["WR", "TE", "RB"]

# ---------------------------------------------------------------- data
def fetch(rel, max_age_h=6):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, rel.split("/")[-1])
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age_h * 3600:
        return path
    try:
        urllib.request.urlretrieve(BASE + rel, path)
    except Exception as e:
        if not os.path.exists(path):
            raise RuntimeError(f"Couldn't download {rel}: {e}")
        print(f"  (using cached {rel}: {e})", file=sys.stderr)
    return path

def read(rel):
    p = fetch(rel)
    return pd.read_csv(p, low_memory=False) if p.endswith(".csv") else pd.read_parquet(p)

def load(season):
    d = {}
    d["pbp"] = read(f"pbp/play_by_play_{season}.parquet")
    d["games"] = read("schedules/games.csv")
    d["roster"] = read(f"weekly_rosters/roster_weekly_{season}.parquet")
    d["snaps"] = read(f"snap_counts/snap_counts_{season}.parquet")
    for k, rel in [("inj", f"injuries/injuries_{season}.parquet"),
                   ("ftn", f"ftn_charting/ftn_charting_{season}.parquet")]:
        try: d[k] = read(rel)
        except Exception: d[k] = pd.DataFrame()
    # man/zone coverage: newest season that exists (free feed stopped after 2025)
    d["cov"], d["cov_season"] = pd.DataFrame(), None
    for s in range(season, season - 3, -1):
        try:
            d["cov"] = read(f"pbp_participation/pbp_participation_{s}.parquet"); d["cov_season"] = s; break
        except Exception: continue
    return d

# ---------------------------------------------------------------- prep
def prep(d, season, thru_week):
    pbp = d["pbp"]
    pbp = pbp[(pbp.season_type == "REG") & (pbp.week <= thru_week)].copy()
    ros = d["roster"]; ros = ros[ros.week <= thru_week + 1]
    latest = ros.sort_values("week").groupby("gsis_id").tail(1)
    pos = latest.set_index("gsis_id")["position"].replace({"FB": "RB"}).to_dict()
    name = latest.set_index("gsis_id")["full_name"].to_dict()
    team_now = latest.set_index("gsis_id")["team"].to_dict()

    pa = pbp[(pbp.pass_attempt == 1) & (pbp.sack == 0) & (pbp.two_point_attempt != 1)].copy()
    tg = pa[pa.receiver_player_id.notna()].copy()
    tg["pos"] = tg.receiver_player_id.map(pos).fillna("WR")
    tg["yds"] = np.where(tg.complete_pass == 1, tg.yards_gained, 0.0)
    tg["loc"] = np.where(tg.pass_location == "middle", "middle", "outside")
    tg["rz"] = tg.yardline_100 <= 20
    tg["ay"] = tg.air_yards.fillna(0)

    ru = pbp[(pbp.play_type == "run") & (pbp.two_point_attempt != 1) & pbp.rusher_player_id.notna()].copy()
    ru["pos"] = ru.rusher_player_id.map(pos).fillna("RB")
    ru["rz"] = ru.yardline_100 <= 20

    db = pbp[(pbp.qb_dropback == 1) & (pbp.two_point_attempt != 1)].copy()

    # games each player appeared in (snap counts, matched by pfr id then by name+team)
    sn = d["snaps"]; sn = sn[(sn.week <= thru_week) & (sn.game_type == "REG") & (sn.offense_snaps > 0)].copy()
    pfr2g = latest.dropna(subset=["pfr_id"]).set_index("pfr_id")["gsis_id"].to_dict()
    nt2g = {(r.full_name, r.team): r.gsis_id for r in latest.itertuples()}
    sn["gsis_id"] = sn.pfr_player_id.map(pfr2g)
    miss = sn.gsis_id.isna()
    sn.loc[miss, "gsis_id"] = [nt2g.get((a, b)) for a, b in zip(sn.loc[miss, "player"], sn.loc[miss, "team"])]
    sn = sn.dropna(subset=["gsis_id"])
    return dict(pbp=pbp, pa=pa, tg=tg, ru=ru, db=db, sn=sn, pos=pos, name=name,
                team_now=team_now, roster=ros)

# ---------------------------------------------------------------- team volume
def team_volume(P):
    pbp, db, ru = P["pbp"], P["db"], P["ru"]
    g_off = pbp.groupby("posteam").game_id.nunique()
    g_def = pbp.groupby("defteam").game_id.nunique()
    att = P["pa"].groupby("posteam").size() / g_off
    drop = db.groupby("posteam").size() / g_off
    rush = ru[ru.qb_scramble != 1].groupby("posteam").size() / g_off
    neutral = pbp[(pbp.wp.between(.2, .8)) & (pbp.down <= 3) & (pbp.half_seconds_remaining > 120)
                  & pbp.play_type.isin(["pass", "run"])]
    proe = neutral.groupby("posteam").pass_oe.mean()
    npr = neutral.groupby("posteam").apply(lambda x: (x.play_type == "pass").mean(), include_groups=False)
    att_allowed = P["pa"].groupby("defteam").size() / g_def
    drop_allowed = db.groupby("defteam").size() / g_def
    rush_allowed = ru[ru.qb_scramble != 1].groupby("defteam").size() / g_def
    v = pd.DataFrame(dict(games=g_off, att_pg=att, drop_pg=drop, rush_pg=rush, proe=proe, neutral_pass=npr))
    a = pd.DataFrame(dict(att_allowed=att_allowed, drop_allowed=drop_allowed, rush_allowed=rush_allowed))
    # pass efficiency
    v["ypa"] = P["pa"].groupby("posteam").yards_gained.sum() / P["pa"].groupby("posteam").size()
    v["comp"] = P["pa"].groupby("posteam").complete_pass.mean()
    return v, a

# ---------------------------------------------------------------- defense profile (opponent-adjusted)
def opp_adjusted(df, key, val, cnt=None):
    """For each defense: actual sum of `val` allowed vs what those same offenses
    produce against everyone else. key = extra grouping (e.g. pos). Returns ratio table."""
    keys = ["posteam"] + ([key] if key else [])
    tot = df.groupby(keys).agg(v=(val, "sum"), n=(val, "size"))
    grp = df.groupby(keys + ["defteam"]).agg(v=(val, "sum"), n=(val, "size")).reset_index()
    grp = grp.merge(tot, left_on=keys, right_index=True, suffixes=("", "_tot"))
    other_n = (grp.n_tot - grp.n).clip(lower=1)
    grp["exp"] = (grp.v_tot - grp.v) / other_n * grp.n
    out = grp.groupby(["defteam"] + ([key] if key else [])).agg(act=("v", "sum"), exp=("exp", "sum"), n=("n", "sum"))
    return out

def defense_profile(P, cov, ftn):
    tg, ru, db = P["tg"], P["ru"], P["db"]
    lg_ypt = tg.groupby("pos").yds.mean()
    # yards per target allowed by position, opponent-adjusted & shrunk
    o = opp_adjusted(tg, "pos", "yds")
    k = 40 * o.index.get_level_values("pos").map(lg_ypt).values
    o["ypt_factor"] = (o.act + k) / (o.exp + k)
    c = opp_adjusted(tg, "pos", "complete_pass")
    o["catch_factor"] = (c.act + 30 * 0.65) / (c.exp + 30 * 0.65)
    o["tgts"] = o.n
    # share of targets funneled to each position vs those offenses' usual mix
    tg2 = tg.copy()
    for p in POS_SKILL: tg2[f"is_{p}"] = (tg2.pos == p).astype(float)
    funnel = {}
    for p in POS_SKILL:
        f = opp_adjusted(tg2, None, f"is_{p}")
        funnel[p] = (f.act + 15) / (f.exp + 15)
    funnel = pd.DataFrame(funnel)
    # yards per game allowed by position (raw) + adjusted per game
    gdef = P["pbp"].groupby("defteam").game_id.nunique()
    o = o.reset_index()
    o["ypg_allowed"] = o.act / o.defteam.map(gdef)
    o["ypg_vs_exp"] = (o.act - o.exp) / o.defteam.map(gdef)
    pos_tbl = o
    # location (middle vs outside)
    lg_loc = tg.groupby("loc").yds.mean()
    l = opp_adjusted(tg, "loc", "yds")
    kl = 40 * l.index.get_level_values("loc").map(lg_loc).values
    l["factor"] = (l.act + kl) / (l.exp + kl)
    l["ypt"] = l.act / l.n
    loc_tbl = l.reset_index()
    # overall pass D
    D = pd.DataFrame(dict(
        epa_db=db.groupby("defteam").epa.mean(),
        succ_db=db.groupby("defteam").success.mean(),
        sack_rate=db.groupby("defteam").sack.mean(),
        explosive=P["pa"].assign(x=P["pa"].yards_gained >= 20).groupby("defteam").x.mean(),
        ypc_allowed=ru[ru.qb_scramble != 1].groupby("defteam").yards_gained.mean(),
    ))
    rr = opp_adjusted(ru[ru.qb_scramble != 1], None, "yards_gained")
    D["rush_factor"] = (rr.act + 60 * 4.3) / (rr.exp + 60 * 4.3)
    pyd = opp_adjusted(P["pa"], None, "yards_gained")
    D["pass_factor"] = (pyd.act + 80 * 6.5) / (pyd.exp + 80 * 6.5)
    # blitz / rush (FTN, current season)
    if len(ftn):
        f = ftn.merge(P["db"][["game_id", "play_id", "defteam", "posteam", "epa"]],
                      left_on=["nflverse_game_id", "nflverse_play_id"], right_on=["game_id", "play_id"])
        D["blitz_rate"] = f.groupby("defteam").n_blitzers.apply(lambda x: (x > 0).mean())
        D["rushers"] = f.groupby("defteam").n_pass_rushers.mean()
        f["blitz"] = f.n_blitzers > 0
        ob = f.groupby(["posteam", "blitz"]).epa.mean().unstack()
        off_blitz = (ob.get(True) - ob.get(False)).rename("epa_vs_blitz_minus_base")
    else:
        off_blitz = pd.Series(dtype=float)
    # man coverage (last season with free data)
    if len(cov):
        cv = cov[cov.defense_man_zone_type.isin(["MAN_COVERAGE", "ZONE_COVERAGE"])].copy()
        # defteam isn't in participation; derive from game id + possession team
        gm = cv.nflverse_game_id.str.split("_", expand=True)
        cv["defteam"] = np.where(cv.possession_team == gm[2], gm[3], gm[2])
        D["man_rate_prev"] = cv.groupby("defteam").defense_man_zone_type.apply(lambda x: (x == "MAN_COVERAGE").mean())
        D["top_shell_prev"] = cv.groupby("defteam").defense_coverage_type.agg(lambda x: x.mode().iat[0] if len(x.mode()) else None)
    D["rank_pass_d"] = D.epa_db.rank(ascending=False)  # 1 = worst (allows most EPA)
    return dict(pos=pos_tbl, loc=loc_tbl, D=D, funnel=funnel, off_blitz=off_blitz, lg_ypt=lg_ypt)

# ---------------------------------------------------------------- player usage
def player_usage(P, thru_week):
    tg, ru, sn = P["tg"], P["ru"], P["sn"]
    games_played = sn.groupby("gsis_id").week.apply(set).to_dict()
    snap_pct = sn[sn.week > thru_week - 3].groupby("gsis_id").offense_pct.mean()
    team_tg_wk = tg.groupby(["posteam", "week"]).size()
    team_ru_wk = ru[ru.qb_scramble != 1].groupby(["posteam", "week"]).size()
    rows = []
    pt = tg.groupby(["receiver_player_id", "posteam", "week"]).agg(
        t=("yds", "size"), rec=("complete_pass", "sum"), yds=("yds", "sum"),
        ay=("ay", "sum"), rz=("rz", "sum"), mid=("loc", lambda x: (x == "middle").sum())).reset_index()
    pr = ru[ru.qb_scramble != 1].groupby(["rusher_player_id", "posteam", "week"]).agg(
        c=("yards_gained", "size"), ryds=("yards_gained", "sum"), rrz=("rz", "sum")).reset_index()
    pt = pt.rename(columns={"receiver_player_id": "id"}); pr = pr.rename(columns={"rusher_player_id": "id"})
    m = pd.merge(pt, pr, on=["id", "posteam", "week"], how="outer").fillna(0)
    for (pid, team), g in m.groupby(["id", "posteam"]):
        weeks = games_played.get(pid, set()) | set(g.week)
        weeks = sorted(w for w in weeks if (team, w) in team_tg_wk.index or (team, w) in team_ru_wk.index)
        if not weeks: continue
        g = g.set_index("week").reindex(weeks).fillna(0)
        ttm = np.array([team_tg_wk.get((team, w), 0) for w in weeks], float)
        trm = np.array([team_ru_wk.get((team, w), 0) for w in weeks], float)
        last3 = weeks[-3:]; i3 = [weeks.index(w) for w in last3]
        sh = g.t.sum() / max(ttm.sum(), 1); sh3 = g.t.values[i3].sum() / max(ttm[i3].sum(), 1)
        cs = g.c.sum() / max(trm.sum(), 1); cs3 = g.c.values[i3].sum() / max(trm[i3].sum(), 1)
        rows.append(dict(id=pid, team=team, name=P["name"].get(pid, pid), pos=P["pos"].get(pid, "?"),
                         gp=len(weeks), last_week=max(weeks), targets=g.t.sum(), rec=g.rec.sum(), yds=g.yds.sum(),
                         tshare=sh, tshare_l3=sh3, ay=g.ay.sum(), rz_t=g.rz.sum(), mid_share=g.mid.sum() / max(g.t.sum(), 1),
                         carries=g.c.sum(), ryds=g.ryds.sum(), cshare=cs, cshare_l3=cs3, rz_c=g.rrz.sum(),
                         ypg=g.yds.mean(), rypg=g.ryds.mean(), snap_l3=snap_pct.get(pid, np.nan)))
    U = pd.DataFrame(rows)
    U = U[U.team == U.id.map(P["team_now"]).fillna(U.team)]  # drop old teams after trades
    team_ay = tg.groupby("posteam").ay.sum()
    U["ay_share"] = U.ay / U.team.map(team_ay)
    U["adot"] = U.ay / U.targets.replace(0, np.nan)
    U["wopr"] = 1.5 * U.tshare + 0.7 * U.ay_share
    return U

# ---------------------------------------------------------------- injuries
def injury_status(d, P, team, week):
    inj = d["inj"]
    out = {}
    note = None
    if len(inj):
        wk = inj[(inj.team == team) & (inj.week == week)]
        if wk.empty:
            note = f"Week {week} injury report not out yet (usually Wed–Fri). Using last week's report + roster IR."
            wk = inj[(inj.team == team) & (inj.week == week - 1)]
        for r in wk.itertuples():
            out[r.gsis_id] = dict(status=r.report_status if isinstance(r.report_status, str) else "—",
                                  practice=r.practice_status if isinstance(r.practice_status, str) else "—",
                                  injury=r.report_primary_injury if isinstance(r.report_primary_injury, str) else "")
    ros = P["roster"]; rw = ros[(ros.team == team)]
    if len(rw):
        lastw = rw.week.max()
        for r in rw[(rw.week == lastw) & rw.status.isin(["RES", "INA"])].itertuples():
            if r.status == "RES" or r.gsis_id not in out:
                out[r.gsis_id] = dict(status="IR/Reserve" if r.status == "RES" else "Inactive last wk",
                                      practice="—", injury=out.get(r.gsis_id, {}).get("injury", ""))
    return out, note

def unavailable(s):
    return s in ("Out", "Doubtful", "IR/Reserve")

# ---------------------------------------------------------------- projection + sim
def proj_game(off, de, spread_for_off, total, vol, allowed, DP, U, inj, P):
    lg_att = vol.att_pg.mean(); lg_rush = vol.rush_pg.mean()
    att = 0.55 * vol.att_pg.get(off, lg_att) + 0.45 * allowed.att_allowed.get(de, lg_att)
    rush = 0.55 * vol.rush_pg.get(off, lg_rush) + 0.45 * allowed.rush_allowed.get(de, lg_rush)
    # game script: underdogs throw more, favorites run more (~0.35 att per point)
    if spread_for_off is not None and not np.isnan(spread_for_off):
        att -= 0.35 * spread_for_off
        rush += 0.30 * spread_for_off
    # spread_for_off > 0 means off is favored by that many
    if total is not None and not np.isnan(total):
        scale = 1 + (total - 45) * 0.006
        att *= scale; rush *= scale
    team_U = U[U.team == off].copy()
    team_U["status"] = team_U.id.map(lambda x: inj.get(x, {}).get("status", ""))
    team_U["practice"] = team_U.id.map(lambda x: inj.get(x, {}).get("practice", ""))
    team_U["out"] = team_U.status.map(unavailable)
    # blend share: recent role weighted more; recency only counts games played
    team_U["share_raw"] = 0.6 * team_U.tshare_l3 + 0.4 * team_U.tshare
    team_U["cshare_raw"] = 0.6 * team_U.cshare_l3 + 0.4 * team_U.cshare
    # positional funnel from defense
    fun = DP["funnel"]
    team_U["funnel"] = team_U.pos.map(lambda p: fun.loc[de, p] if de in fun.index and p in fun.columns else 1.0)
    team_U["share_adj"] = team_U.share_raw * np.sqrt(team_U.funnel.clip(.6, 1.6))
    act = team_U[~team_U.out & (team_U.last_week >= team_U.last_week.max() - 3)]
    recent_cut = team_U.last_week.max() - 1
    newly_out = team_U[team_U.out & (team_U.last_week >= recent_cut)]
    vacated_t = newly_out.share_raw.sum()
    vacated_c = newly_out.cshare_raw.sum()
    S = 0.92
    tot = act.share_adj.sum()
    scale = np.clip(S / tot, 0.8, 1.35) if tot > 0 else 1
    team_U.loc[act.index, "proj_share"] = act.share_adj * scale
    ctot = act[act.pos.isin(["RB", "QB", "WR"])].cshare_raw.sum()
    cscale = np.clip(0.95 / ctot, 0.8, 1.45) if ctot > 0 else 1
    team_U.loc[act.index, "proj_cshare"] = act.cshare_raw * cscale
    team_U = team_U.loc[act.index]
    team_U["proj_targets"] = team_U.proj_share * att
    team_U["proj_carries"] = team_U.proj_cshare * rush
    pos_t = DP["pos"].set_index(["defteam", "pos"])
    loc_t = DP["loc"].set_index(["defteam", "loc"])
    lg_ypt = DP["lg_ypt"]
    res = []
    tg, ru = P["tg"], P["ru"]
    lg_catch = tg.groupby("pos").complete_pass.mean()
    for r in team_U.itertuples():
        pf = pos_t.ypt_factor.get((de, r.pos), 1.0)
        cf = pos_t.catch_factor.get((de, r.pos), 1.0)
        mf = loc_t.factor.get((de, "middle"), 1.0); of = loc_t.factor.get((de, "outside"), 1.0)
        lf = r.mid_share * mf + (1 - r.mid_share) * of
        mraw = pf * lf
        yf = mraw ** MATCH_W
        # player catch rate shrunk to position average
        pc = (r.rec + 20 * lg_catch.get(r.pos, .65)) / (r.targets + 20)
        pcatch = float(np.clip(pc * cf, .3, .92))
        my_rec = tg[(tg.receiver_player_id == r.id) & (tg.complete_pass == 1)].yds.values
        pool_rec = tg[(tg.pos == r.pos) & (tg.complete_pass == 1)].yds.values
        res.append(dict(id=r.id, name=r.name, pos=r.pos, team=off, opp=de, status=r.status, practice=r.practice, share_raw=r.share_raw,
                        tshare=r.tshare, tshare_l3=r.tshare_l3, proj_share=r.proj_share, proj_targets=r.proj_targets,
                        proj_carries=r.proj_carries, catch=pcatch, yf=yf, funnel=r.funnel,
                        rec_pool=(my_rec, pool_rec), rush_pool=(ru[(ru.rusher_player_id == r.id) & (ru.qb_scramble != 1)].yards_gained.values,
                                                                 ru[(ru.pos == r.pos) & (ru.qb_scramble != 1)].yards_gained.values),
                        mraw=mraw, rraw=DP["D"].rush_factor.get(de, 1.0),
                        rf=DP["D"].rush_factor.get(de, 1.0) ** MATCH_W_RUSH, ypg=r.ypg, adot=r.adot, wopr=r.wopr, snap=r.snap_l3,
                        rz_t=r.rz_t, rz_c=r.rz_c, gp=r.gp))
    out = dict(att=att, rush=rush, vacated_t=vacated_t, vacated_c=vacated_c, players=res,
               out_list=[f"{r.name} ({r.status})" for r in newly_out.itertuples() if r.share_raw > 0.05])
    return out

def bootstrap(own, pool, n, k=25):
    """Sample yards per play: player's own plays mixed with position pool (shrinkage)."""
    w_own = len(own) / (len(own) + k) if len(own) else 0
    use_own = RNG.random(n) < w_own
    a = RNG.choice(own, n) if len(own) else np.zeros(n)
    b = RNG.choice(pool, n) if len(pool) else np.zeros(n)
    return np.where(use_own, a, b)

def negbin(mean, r, n):
    mean = max(mean, 0.01)
    lam = RNG.gamma(r, mean / r, n)
    return RNG.poisson(lam)

def simulate(p):
    n = SIMS
    T = negbin(p["proj_targets"], 6, n)
    out = {}
    if p["proj_targets"] >= 0.8:
        maxT = int(T.max()) if T.max() > 0 else 1
        catches = RNG.random((n, maxT)) < p["catch"]
        mask = np.arange(maxT)[None, :] < T[:, None]
        C = (catches & mask)
        y = bootstrap(*p["rec_pool"], n * maxT).reshape(n, maxT) * p["yf"]
        out["receptions"] = C.sum(1)
        out["rec_yds"] = (y * C).sum(1)
    if p["proj_carries"] >= 2:
        Cn = negbin(p["proj_carries"], 10, n)
        maxC = int(Cn.max()) if Cn.max() > 0 else 1
        y = bootstrap(*p["rush_pool"], n * maxC).reshape(n, maxC) * p["rf"]
        out["rush_yds"] = (y * (np.arange(maxC)[None, :] < Cn[:, None])).sum(1)
    return out

def qb_sim(team, att, vol, DP, de, P):
    pa = P["pa"][P["pa"].posteam == team]
    comp = pa[pa.complete_pass == 1].yards_gained.values
    lg = P["pa"][P["pa"].complete_pass == 1].yards_gained.values
    cp = vol.comp.get(team, .64)
    n = SIMS
    A = np.maximum(0, RNG.normal(att, 5.5, n)).round().astype(int)
    maxA = int(A.max()) or 1
    C = (RNG.random((n, maxA)) < cp) & (np.arange(maxA)[None, :] < A[:, None])
    y = bootstrap(comp, lg, n * maxA, k=60).reshape(n, maxA) * DP["D"].pass_factor.get(de, 1.0)
    return dict(pass_att=A, pass_yds=(y * C).sum(1))

def implied(odds):
    try: o = float(odds)
    except Exception: return 0.5238
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)

def payout(odds):
    o = float(odds) if odds not in (None, "") and not (isinstance(odds, float) and np.isnan(odds)) else -110
    return o / 100 if o > 0 else 100 / -o

# ---------------------------------------------------------------- odds api (optional)
_EVENTS = None
ODDS_MARKETS = ["rec_yds", "rush_yds", "receptions"]  # each market costs 1 credit per game

def odds_api_lines(key, home, away):
    """Pull player props from The Odds API (needs your own free key). Returns DataFrame."""
    mk = {"player_reception_yds": "rec_yds", "player_receptions": "receptions",
          "player_rush_yds": "rush_yds", "player_pass_yds": "pass_yds"}
    mk = {k: v for k, v in mk.items() if v in ODDS_MARKETS}
    base = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
    global _EVENTS
    if _EVENTS is None:
        try:
            _EVENTS = json.load(urllib.request.urlopen(f"{base}/events?apiKey={key}", timeout=20))
        except Exception as e:
            print(f"  Odds API error: {e}", file=sys.stderr); _EVENTS = []
    ev = _EVENTS
    rows = []
    for e in ev:
        if not (TEAMNAME.get(home, "?") in e["home_team"] and TEAMNAME.get(away, "?") in e["away_team"]): continue
        url = f"{base}/events/{e['id']}/odds?apiKey={key}&regions=us&oddsFormat=american&markets={','.join(mk)}"
        try: o = json.load(urllib.request.urlopen(url, timeout=20))
        except Exception as ex: print(f"  Odds API error: {ex}", file=sys.stderr); continue
        for bk in o.get("bookmakers", [])[:1]:
            for m in bk["markets"]:
                tmp = {}
                for oc in m["outcomes"]:
                    k2 = (oc.get("description"), oc.get("point"))
                    tmp.setdefault(k2, {})[oc["name"].lower()] = oc["price"]
                for (pl, pt), pr in tmp.items():
                    rows.append(dict(player=pl, market=mk[m["key"]], line=pt, over_odds=pr.get("over", -110),
                                     under_odds=pr.get("under", -110), book=bk["title"]))
    return pd.DataFrame(rows)

TEAMNAME = {"ARI":"Cardinals","ATL":"Falcons","BAL":"Ravens","BUF":"Bills","CAR":"Panthers","CHI":"Bears","CIN":"Bengals",
 "CLE":"Browns","DAL":"Cowboys","DEN":"Broncos","DET":"Lions","GB":"Packers","HOU":"Texans","IND":"Colts","JAX":"Jaguars",
 "KC":"Chiefs","LV":"Raiders","LAC":"Chargers","LA":"Rams","MIA":"Dolphins","MIN":"Vikings","NE":"Patriots","NO":"Saints",
 "NYG":"Giants","NYJ":"Jets","PHI":"Eagles","PIT":"Steelers","SF":"49ers","SEA":"Seahawks","TB":"Buccaneers","TEN":"Titans","WAS":"Commanders"}

# ---------------------------------------------------------------- angles (the notepad)
def ordinal(n):
    n = int(n); return f"{n}{'th' if 11<=n%100<=13 else {1:'st',2:'nd',3:'rd'}.get(n%10,'th')}"

def angles_for(off, de, DP, vol, allowed, G, inj_off, U, cov_season):
    A = []
    pos = DP["pos"]; D = DP["D"]; loc = DP["loc"]
    for p in POS_SKILL:
        t = pos[pos.pos == p].copy()
        t["rk"] = t.ypg_vs_exp.rank(ascending=False)
        r = t[t.defteam == de]
        if len(r):
            r = r.iloc[0]
            if r.rk <= 8:
                A.append(("good", f"{de} D gives up {r.ypg_vs_exp:+.0f} yds/game to {p}s vs what those offenses normally get ({ordinal(r.rk)} worst). {r.ypg_allowed:.0f} {p} yds/game allowed."))
            elif r.rk >= 25:
                A.append(("bad", f"{de} D shuts down {p}s: {r.ypg_vs_exp:+.0f} yds/game vs expected ({ordinal(33 - r.rk)} best)."))
    lm = loc[loc["loc"] == "middle"].copy(); lm["rk"] = lm.factor.rank(ascending=False)
    lo = loc[loc["loc"] == "outside"].copy(); lo["rk"] = lo.factor.rank(ascending=False)
    for nm, t, who in [("middle of the field", lm, "slot WRs / TEs"), ("outside", lo, "outside WRs")]:
        r = t[t.defteam == de]
        if len(r):
            r = r.iloc[0]
            if r.rk <= 6: A.append(("good", f"{de} is weak {nm}: {r.ypt:.1f} yds/target, {(r.factor-1)*100:+.0f}% vs expected ({ordinal(r.rk)} worst). Favors {who}."))
            if r.rk >= 27: A.append(("bad", f"{de} is stingy {nm} ({(r.factor-1)*100:+.0f}% vs expected). Tough spot for {who}."))
    if "blitz_rate" in D:
        br = D.blitz_rate.rank(ascending=False)
        if de in br.index and br[de] <= 6:
            ob = DP["off_blitz"].get(off, np.nan)
            extra = f" {off} offense is {'better' if ob > 0 else 'worse'} vs blitz ({ob:+.2f} EPA/play)." if not np.isnan(ob) else ""
            A.append(("info", f"{de} blitzes {D.blitz_rate[de]*100:.0f}% of dropbacks ({ordinal(br[de])} most).{extra}"))
    if "man_rate_prev" in D and de in D.index and not np.isnan(D.man_rate_prev.get(de, np.nan)):
        mr = D.man_rate_prev.rank(ascending=False)[de]
        if mr <= 6 or mr >= 27:
            A.append(("info", f"{de} played man {D.man_rate_prev[de]*100:.0f}% in {cov_season} ({ordinal(mr)} most). Last season's data; scheme may have changed."))
    vr = vol.att_pg.rank(ascending=False)
    A.append(("info", f"{off} projects for ~{G['att']:.0f} pass attempts and ~{G['rush']:.0f} RB/WR carries (season avg {vol.att_pg.get(off, np.nan):.1f} att, {ordinal(vr.get(off, 16))} most)."))
    if G["vacated_t"] > 0.06 and G["out_list"]:
        gain = sorted(G["players"], key=lambda x: -(x["proj_share"] - x["share_raw"]))[:3]
        g = ", ".join(f"{x['name']} {x['share_raw']*100:.0f}%→{x['proj_share']*100:.0f}%" for x in gain if x["proj_share"] - x["share_raw"] > 0.01)
        A.append(("good", f"{G['vacated_t']*100:.0f}% of {off} targets open up with {', '.join(G['out_list'])} out." + (f" Biggest bumps: {g}." if g else "")))
    elif G["out_list"]:
        A.append(("info", f"{off} missing {', '.join(G['out_list'])}."))
    for x in G["players"]:
        st = inj_off.get(x["id"], {})
        if st.get("status") == "Questionable" and x["proj_share"] > 0.12:
            A.append(("warn", f"{x['name']} is Questionable ({st.get('practice','')}). If he sits, rerun; his {x['proj_share']*100:.0f}% target share gets spread out."))
    return A

# ---------------------------------------------------------------- run a week
def current_week(games, season):
    """Week of the next game that hasn't been played yet."""
    g = games[(games.season == season) & (games.game_type == "REG") & games.home_score.isna()]
    return int(g.week.min()) if len(g) else int(games[games.season == season].week.max())

def run_week(d, season, week, only_game=None, lines=None, odds_key=None, skip_final=False):
    P = prep(d, season, week - 1)
    vol, allowed = team_volume(P)
    DP = defense_profile(P, d["cov"], d["ftn"][d["ftn"].week < week] if len(d["ftn"]) else d["ftn"])
    U = player_usage(P, week - 1)
    g = d["games"]; g = g[(g.season == season) & (g.week == week) & (g.game_type == "REG")]
    if skip_final: g = g[g.home_score.isna()]
    if only_game:
        a, h = only_game.upper().split("-")
        g = g[((g.away_team == a) & (g.home_team == h)) | ((g.away_team == h) & (g.home_team == a))]
    Ui = U.drop_duplicates("id").set_index("id")
    games = []
    for gm in g.itertuples():
        sp = gm.spread_line if not pd.isna(gm.spread_line) else np.nan  # >0 = home favored
        tot = gm.total_line
        gl = lines.copy() if lines is not None else pd.DataFrame()
        if odds_key:
            ol = odds_api_lines(odds_key, gm.home_team, gm.away_team)
            gl = pd.concat([gl, ol]) if len(ol) else gl
        sides = []
        for off, de, spf in [(gm.away_team, gm.home_team, -sp), (gm.home_team, gm.away_team, sp)]:
            inj, note = injury_status(d, P, off, week)
            G = proj_game(off, de, spf, tot, vol, allowed, DP, U, inj, P)
            props = []
            for p in G["players"]:
                if p["proj_targets"] < 1.5 and p["proj_carries"] < 4: continue
                sim = simulate(p)
                row = dict(name=p["name"], pos=p["pos"], status=p["status"], practice=p["practice"],
                           tshare=p["tshare"], tshare_l3=p["tshare_l3"], proj_share=p["proj_share"],
                           proj_targets=p["proj_targets"], proj_carries=p["proj_carries"], adot=p["adot"],
                           wopr=p["wopr"], snap=p["snap"], yf=p["yf"], funnel=p["funnel"], sims=sim, edges=[],
                           share_raw=p["share_raw"], gp=int(Ui.gp.get(p["id"], 0)), team=off, opp=de,
                           base={"rec_yds": Ui.ypg.get(p["id"], np.nan),
                                 "receptions": Ui.rec.get(p["id"], 0) / max(Ui.gp.get(p["id"], 1), 1),
                                 "rush_yds": Ui.rypg.get(p["id"], np.nan)})
                for mk in ["receptions", "rec_yds", "rush_yds"]:
                    if mk in sim: row[f"med_{mk}"] = float(np.median(sim[mk])); row[f"mean_{mk}"] = float(sim[mk].mean())
                props.append(row)
            qbs = qb_sim(off, G["att"], vol, DP, de, P)
            qb_name = P["db"][(P["db"].posteam == off) & (P["db"].week == P["db"][P["db"].posteam == off].week.max())].passer_player_name.mode()
            qb_name = qb_name.iat[0] if len(qb_name) else f"{off} QB"
            qrow = dict(name=qb_name, med_pass_yds=float(np.median(qbs["pass_yds"])), med_pass_att=float(np.median(qbs["pass_att"])), sims=qbs)
            ang = angles_for(off, de, DP, vol, allowed, G, inj, U, d["cov_season"])
            # compare to lines
            edges = []
            if len(gl):
                allp = props + [qrow]
                for ln in gl.itertuples():
                    pl = next((x for x in allp if name_match(x["name"], ln.player)), None)
                    if pl is None or ln.market not in pl["sims"]: continue
                    s = pl["sims"][ln.market]
                    po = float((s > ln.line).mean()); pu = float((s < ln.line).mean())
                    oo = getattr(ln, "over_odds", -110); uo = getattr(ln, "under_odds", -110)
                    oo = -110 if pd.isna(oo) else oo; uo = -110 if pd.isna(uo) else uo
                    evo = po * payout(oo) - (1 - po); evu = pu * payout(uo) - (1 - pu)
                    side, prob, ev, odds = ("Over", po, evo, oo) if evo > evu else ("Under", pu, evu, uo)
                    edges.append(dict(player=pl["name"], market=ln.market, line=ln.line, side=side, prob=prob,
                                      implied=implied(odds), odds=odds, ev=ev, median=float(np.median(s))))
            sides.append(dict(off=off, de=de, G=G, props=props, qb=qrow, angles=ang, inj_note=note,
                              inj=inj, edges=sorted(edges, key=lambda e: -e["ev"])))
        G_ = dict(game_id=gm.game_id, away=gm.away_team, home=gm.home_team, day=gm.weekday,
                  date=gm.gameday, time=gm.gametime, spread=sp, total=tot, sides=sides)
        G_["picks"] = make_picks(G_)
        games.append(G_)
    return dict(season=season, week=week, games=games, thru=week - 1, cov_season=d["cov_season"], _U=Ui,
                built=datetime.now(ZoneInfo("America/New_York")).strftime("%a %b %d, %I:%M %p ET"))

MK_LABEL = {"rec_yds": "rec yds", "receptions": "catches", "rush_yds": "rush yds", "pass_yds": "pass yds", "pass_att": "pass att"}
# Books set lines near a player's typical game, not his average (big games drag averages up).
# Calibrated on 2026 Wks 4-7: these multiples of season avg give a ~50/50 line.
FAIR_K = {"rec_yds": 0.82, "receptions": 0.90, "rush_yds": 0.98}
MIN_LINE = {"rec_yds": 19.5, "receptions": 2.5, "rush_yds": 24.5}  # skip tiny lines books rarely offer
LEAN_MIN = {"rec_yds": ("proj_targets", 4), "receptions": ("proj_targets", 4), "rush_yds": ("proj_carries", 8)}

def pick_reason(p, mk):
    bits = []
    if mk in ("rec_yds", "receptions"):
        bits.append(f"{p['proj_targets']:.1f} proj targets")
        if p["proj_share"] - p["share_raw"] > 0.02:
            bits.append(f"target share up {p['share_raw']*100:.0f}%→{p['proj_share']*100:.0f}% with teammates out")
        if abs(p["yf"] - 1) >= 0.04:
            bits.append(f"{p['opp']} matchup {(p['yf']-1)*100:+.0f}%")
    else:
        bits.append(f"{p['proj_carries']:.1f} proj carries")
    if p["status"] in ("Questionable",) or p["status"] in ("Limited", "DNP"):
        bits.append(f"{p['status']}: confirm he plays")
    return " · ".join(bits)

def make_picks(G):
    """Best bets for one game. Uses real sportsbook lines when present (ranked by EV);
    otherwise 'leans' where the model disagrees with the player's season average,
    which is roughly where books open their lines."""
    picks, have_line = [], set()
    for s in G["sides"]:
        pl_by_name = {p["name"]: p for p in s["props"]}
        for ed in s["edges"]:
            have_line.add((ed["player"], ed["market"]))
            if ed["ev"] >= 0.04 and ed["prob"] - ed["implied"] >= 0.05:
                p = pl_by_name.get(ed["player"])
                reason = pick_reason(p, ed["market"]) if p else ""
                picks.append(dict(kind="line", player=ed["player"], team=s["off"], market=ed["market"], side=ed["side"],
                                  line=ed["line"], odds=ed["odds"], prob=ed["prob"], implied=ed["implied"], ev=ed["ev"],
                                  median=ed["median"], reason=reason, score=ed["ev"]))
    for s in G["sides"]:
        for p in s["props"]:
            if p["gp"] < 3 or p["status"] in ("Out", "Doubtful"): continue
            for mk, (fld, mn) in LEAN_MIN.items():
                if mk not in p["sims"] or p[fld] < mn or (p["name"], mk) in have_line: continue
                b = p["base"].get(mk)
                if b is None or pd.isna(b) or b <= 0: continue
                line = math.floor(b * FAIR_K[mk]) + 0.5
                if line < MIN_LINE[mk]: continue
                po = float((p["sims"][mk] > line).mean())
                conf = abs(po - 0.5)
                if conf < LEAN_CONF: continue
                side = "Over" if po > 0.5 else "Under"
                picks.append(dict(kind="lean", player=p["name"], team=s["off"], market=mk, side=side, line=line,
                                  odds=None, prob=max(po, 1 - po), implied=None, ev=None,
                                  median=float(np.median(p["sims"][mk])), reason=pick_reason(p, mk), score=conf - 0.5))
    # one pick per player (keep the strongest), real lines first
    picks.sort(key=lambda x: (x["kind"] != "line", -x["score"] if x["kind"] == "line" else -x["prob"]))
    seen, out = set(), []
    for pk in picks:
        if pk["player"] in seen: continue
        seen.add(pk["player"]); out.append(pk)
    return out[:4]

def _norm(x):
    x = str(x).lower().replace(".", " ").replace("'", "")
    return [t for t in x.split() if t not in ("jr", "sr", "ii", "iii", "iv")]

def name_match(a, b):
    pa, pb = _norm(a), _norm(b)
    if pa == pb: return True
    return len(pa) > 1 and len(pb) > 1 and pa[-1] == pb[-1] and pa[0][0] == pb[0][0]

# ---------------------------------------------------------------- backtest
def backtest(d, season, week):
    P_next = prep(d, season, week)
    actual_rec = P_next["tg"][P_next["tg"].week == week].groupby("receiver_player_id").yds.sum()
    actual_n = P_next["tg"][P_next["tg"].week == week].groupby("receiver_player_id").complete_pass.sum()
    actual_t = P_next["tg"][P_next["tg"].week == week].groupby("receiver_player_id").size()
    team_att = P_next["pa"][P_next["pa"].week == week].groupby("posteam").size()
    played = set(P_next["sn"][P_next["sn"].week == week].gsis_id)
    R = run_week(d, season, week)
    P = prep(d, season, week - 1); U = player_usage(P, week - 1).set_index("id")
    rows = []
    for g in R["games"]:
        for s in g["sides"]:
            for p in s["G"]["players"]:
                if p["proj_targets"] < 3 or p["id"] not in played: continue
                sim = simulate(p)
                rows.append(dict(name=p["name"], team=p["team"], proj=float(np.median(sim["rec_yds"])),
                                 proj_mean=float(sim["rec_yds"].mean()), proj_rec=float(sim["receptions"].mean()),
                                 base=U.ypg.get(p["id"], np.nan),
                                 base_rec=U.rec.get(p["id"], 0) / max(U.gp.get(p["id"], 1), 1),
                                 actual=actual_rec.get(p["id"], 0.0), actual_rec=actual_n.get(p["id"], 0),
                                 proj_t=p["proj_targets"], actual_t=actual_t.get(p["id"], 0),
                                 base_t=U.targets.get(p["id"], 0) / max(U.gp.get(p["id"], 1), 1),
                                 proj_att=s["G"]["att"], actual_att=team_att.get(p["team"], np.nan),
                                 yf=p["yf"], pos=p["pos"],
                                 p_over_base=float((sim["rec_yds"] > U.ypg.get(p["id"], 0)).mean()),
                                 p_over_lean=float((sim["rec_yds"] > math.floor(U.ypg.get(p["id"], 0) * FAIR_K["rec_yds"]) + 0.5).mean()),
                                 p_over_lean_rec=float((sim["receptions"] > math.floor(U.rec.get(p["id"], 0) / max(U.gp.get(p["id"], 1), 1) * FAIR_K["receptions"]) + 0.5).mean()),
                                 gp=int(U.gp.get(p["id"], 0))))
    B = pd.DataFrame(rows)
    out = dict(n=len(B),
               mae_model=float((B.proj_mean - B.actual).abs().mean()),
               mae_base=float((B.base - B.actual).abs().mean()),
               mae_model_rec=float((B.proj_rec - B.actual_rec).abs().mean()),
               mae_base_rec=float((B.base_rec - B.actual_rec).abs().mean()),
               over_median=float((B.actual > B.proj).mean()))
    # directional test: when model disagrees with season-average "line" by 10+ yds, how often right?
    B["lean_line"] = np.floor(B.base * FAIR_K["rec_yds"]) + 0.5
    B["lean_line_rec"] = np.floor(B.base_rec * FAIR_K["receptions"]) + 0.5
    dis = B[(B.proj_mean - B.base).abs() >= 10]
    hit = ((dis.proj_mean > dis.base) == (dis.actual > dis.base)).mean() if len(dis) else np.nan
    out.update(n_disagree=len(dis), hit_disagree=float(hit))
    return out, B

# ---------------------------------------------------------------- self-tuning
DEFAULT_K = {"rec_yds": 0.82, "receptions": 0.90, "rush_yds": 0.98}

def week_history(d, season, w):
    """Re-project a finished week blind (data before it only) and pair with what happened."""
    global SIMS
    keep = SIMS; SIMS = 4000
    try:
        Pn = prep(d, season, w)
        tgw = Pn["tg"][Pn["tg"].week == w]; ruw = Pn["ru"][(Pn["ru"].week == w) & (Pn["ru"].qb_scramble != 1)]
        a_ry = tgw.groupby("receiver_player_id").yds.sum(); a_rec = tgw.groupby("receiver_player_id").complete_pass.sum()
        a_ru = ruw.groupby("rusher_player_id").yards_gained.sum()
        played = set(Pn["sn"][Pn["sn"].week == w].gsis_id)
        R = run_week(d, season, w); U = R["_U"]
        rec, rush = [], []
        for g in R["games"]:
            for s_ in g["sides"]:
                for p in s_["G"]["players"]:
                    if p["id"] not in played: continue
                    q = dict(p, yf=1.0, rf=1.0); sim = simulate(q)
                    gp = int(U.gp.get(p["id"], 0)); grow = max(gp, 1)
                    if p["proj_targets"] >= 3 and "rec_yds" in sim:
                        rec.append(dict(week=w, sim_ry=sim["rec_yds"], sim_rec=sim["receptions"], mraw=p["mraw"],
                                        act_ry=float(a_ry.get(p["id"], 0)), act_rec=float(a_rec.get(p["id"], 0)),
                                        base_ry=float(U.ypg.get(p["id"], np.nan)), base_rec=float(U.rec.get(p["id"], 0)) / grow,
                                        gp=gp, proj_t=p["proj_targets"]))
                    if p["proj_carries"] >= 6 and "rush_yds" in sim:
                        rush.append(dict(week=w, sim_ru=sim["rush_yds"], rraw=p["rraw"], act_ru=float(a_ru.get(p["id"], 0)),
                                         base_ru=float(U.rypg.get(p["id"], np.nan)), gp=gp, proj_c=p["proj_carries"]))
        return rec, rush
    finally:
        SIMS = keep

def _lean_rows(rec, rush, mw, mwr, K):
    """Every lean the rules would have made, with model prob and result."""
    out = []
    for r in rec:
        if r["gp"] < 3: continue
        c = r["mraw"] ** mw
        for mk, sim, act, base in [("rec_yds", r["sim_ry"] * c, r["act_ry"], r["base_ry"]),
                                   ("receptions", r["sim_rec"], r["act_rec"], r["base_rec"])]:
            if not base > 0 or r["proj_t"] < LEAN_MIN[mk][1]: continue
            line = math.floor(base * K[mk]) + 0.5
            if line < MIN_LINE[mk]: continue
            po = float((sim > line).mean()); out.append(dict(week=r["week"], mk=mk, po=po, hit_over=act > line))
    for r in rush:
        if r["gp"] < 3 or not r["base_ru"] > 0 or r["proj_c"] < LEAN_MIN["rush_yds"][1]: continue
        line = math.floor(r["base_ru"] * K["rush_yds"]) + 0.5
        if line < MIN_LINE["rush_yds"]: continue
        po = float((r["sim_ru"] * r["rraw"] ** mwr > line).mean())
        out.append(dict(week=r["week"], mk="rush_yds", po=po, hit_over=r["act_ru"] > line))
    return out

def fit(rec, rush):
    """Pick the settings that would have done best on the finished weeks given."""
    W = [0, .1, .2, .3, .4, .5, .75, 1.0]
    if len(rec) >= 60:
        med = np.array([np.median(r["sim_ry"]) for r in rec]); m = np.array([r["mraw"] for r in rec]); a = np.array([r["act_ry"] for r in rec])
        mw = min(W, key=lambda w: np.abs(med * m ** w - a).mean())
    else: mw = MATCH_W
    if len(rush) >= 40:
        med = np.array([np.median(r["sim_ru"]) for r in rush]); m = np.array([r["rraw"] for r in rush]); a = np.array([r["act_ru"] for r in rush])
        mwr = min(W, key=lambda w: np.abs(med * m ** w - a).mean())
    else: mwr = MATCH_W_RUSH
    # fair-line factor: multiple of season avg that went over ~50% of the time (shrunk to default)
    K = {}
    for mk, rows, act, base in [("rec_yds", rec, "act_ry", "base_ry"), ("receptions", rec, "act_rec", "base_rec"), ("rush_yds", rush, "act_ru", "base_ru")]:
        rr = [r for r in rows if r["gp"] >= 3 and r[base] > 0]
        if len(rr) < 30: K[mk] = DEFAULT_K[mk]; continue
        A = np.array([r[act] for r in rr]); B = np.array([r[base] for r in rr])
        grid = np.arange(0.6, 1.15, 0.01)
        k_fit = min(grid, key=lambda k: abs((A > np.floor(B * k) + 0.5).mean() - 0.5))
        wgt = len(rr) / (len(rr) + 150)
        K[mk] = round(float(wgt * k_fit + (1 - wgt) * DEFAULT_K[mk]), 3)
    # lean cutoff: loosest one that still hit 56%+ on 30+ leans
    L = _lean_rows(rec, rush, mw, mwr, K); conf = 0.15
    for t in [.08, .10, .12, .15, .20]:
        sel = [x for x in L if abs(x["po"] - .5) >= t]
        if len(sel) >= 30 and np.mean([(x["po"] > .5) == x["hit_over"] for x in sel]) >= 0.56:
            conf = t; break
    return dict(MATCH_W=mw, MATCH_W_RUSH=mwr, FAIR_K=K, LEAN_CONF=conf)

def tune(d, season, week, first_week=4):
    """Re-test every finished week, re-fit settings, and grade a walk-forward record
    (each week judged with settings learned only from the weeks before it)."""
    global MATCH_W, MATCH_W_RUSH, LEAN_CONF
    weeks = list(range(first_week, week))
    if not weeks: return None
    H = {w: week_history(d, season, w) for w in weeks}
    record, by_week = [], []
    defaults = dict(MATCH_W=MATCH_W, MATCH_W_RUSH=MATCH_W_RUSH, FAIR_K=dict(FAIR_K), LEAN_CONF=LEAN_CONF)
    for w in weeks:
        prior = [x for v in weeks if v < w for x in H[v][0]], [x for v in weeks if v < w for x in H[v][1]]
        prm = fit(*prior) if prior[0] else defaults
        L = [x for x in _lean_rows(*H[w], prm["MATCH_W"], prm["MATCH_W_RUSH"], prm["FAIR_K"]) if abs(x["po"] - .5) >= prm["LEAN_CONF"]]
        wins = sum((x["po"] > .5) == x["hit_over"] for x in L)
        by_week.append(dict(week=w, wins=int(wins), losses=len(L) - int(wins))); record += L
    rec_all = [x for w in weeks for x in H[w][0]]; rush_all = [x for w in weeks for x in H[w][1]]
    P = fit(rec_all, rush_all)
    MATCH_W, MATCH_W_RUSH, LEAN_CONF = P["MATCH_W"], P["MATCH_W_RUSH"], P["LEAN_CONF"]
    FAIR_K.update(P["FAIR_K"])
    med = np.array([np.median(r["sim_ry"]) * r["mraw"] ** MATCH_W for r in rec_all])
    act = np.array([r["act_ry"] for r in rec_all]); base = np.array([r["base_ry"] for r in rec_all])
    ok = ~np.isnan(base)
    W_ = sum(b["wins"] for b in by_week); L_ = sum(b["losses"] for b in by_week)
    strong = [x for x in record if abs(x["po"] - .5) >= 0.15]
    return dict(weeks=f"{weeks[0]}–{weeks[-1]}", n=len(rec_all) + len(rush_all), params=P, defaults=defaults,
                mae_model=float(np.abs(med - act)[ok].mean()), mae_base=float(np.abs(base - act)[ok].mean()),
                wins=W_, losses=L_, by_week=by_week,
                strong_w=int(sum((x["po"] > .5) == x["hit_over"] for x in strong)), strong_n=len(strong))

# ---------------------------------------------------------------- report
CSS = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.css")).read() \
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.css")) else ""

def e(x): return html.escape(str(x))

def pick_html(pk, show_game=None):
    side_cls = "over" if pk["side"] == "Over" else "under"
    head = f'<span class="pside {side_cls}">{pk["side"]} {pk["line"]:g}</span> <b>{e(pk["player"])}</b> {MK_LABEL[pk["market"]]}'
    if pk["kind"] == "line":
        odds = int(pk["odds"])
        nums = (f'<span class="pnum">{pk["prob"]*100:.0f}% model vs {pk["implied"]*100:.0f}% book · {odds:+d} · '
                f'EV {pk["ev"]:+.2f}/$1</span><span class="ptag line">vs book line</span>')
    else:
        strong = '<span class="ptag strong">strong</span>' if pk["prob"] >= 0.55 + LEAN_CONF else ""
        nums = (f'<span class="pnum">{pk["prob"]*100:.0f}% model · median {pk["median"]:.0f}</span>'
                f'<span class="ptag lean">lean · no book line yet</span>{strong}')
    g = f'<span class="pgame">{e(show_game)}</span>' if show_game else ""
    return f'<li class="pick">{g}<div class="phead">{head}</div><div class="pmeta">{nums}</div><div class="preason">{e(pk["reason"])}</div></li>'

def render(R, bt=None):
    out = []
    W = out.append
    W(f"<title>NFL Edge Week {R['week']}</title>")
    W('<meta name="claude:viewport-fit-sides" content="cover">')
    W('<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">')
    W(f"<style>{CSS}</style>")
    W('<div class="pad"><header class="mast">')
    W(f'<div class="eyebrow">{R["season"]} season · data through Week {R["thru"]} · built {e(R["built"])}</div>')
    W(f'<h1>NFL Edge <span>Week {R["week"]}</span></h1>')
    W('<p class="lede">The notepad, automated. Each game shows the matchup angles, projected volume, injuries and who picks up the missing targets, '
      'then a 20,000-game simulation for every player\'s catches and yards. Compare the medians to the sportsbook line. '
      'A gap of 10%+ in hit rate is worth a look; anything smaller is noise.</p>')
    if bt:
        P = bt["params"]; tot = bt["wins"] + bt["losses"]
        pct = bt["wins"] / tot * 100 if tot else float("nan")
        wk = " · ".join(f'Wk {b["week"]} {b["wins"]}–{b["losses"]}' for b in bt["by_week"])
        sp = f' Strong leans (65%+): {bt["strong_w"]}–{bt["strong_n"]-bt["strong_w"]}.' if bt["strong_n"] else ""
        W(f'<div class="bt"><b>Self-check, Weeks {bt["weeks"]}.</b> Every run re-projects each finished week using only data from before it, '
          f'grades the result, and re-tunes the settings below. '
          f'<b>Lean record: {bt["wins"]}–{bt["losses"]} ({pct:.0f}%)</b>. Each week was graded with settings learned only from the weeks before it. {wk}.{sp} '
          f'Break-even at −110 is 52.4%. Receiving-yard misses averaged {bt["mae_model"]:.1f} yds vs {bt["mae_base"]:.1f} for season averages.'
          f'<div class="tune">This run\'s settings: matchup weight {P["MATCH_W"]:g} (receiving) / {P["MATCH_W_RUSH"]:g} (rushing) · '
          f'fair line = {P["FAIR_K"]["rec_yds"]:.2f}× avg rec yds, {P["FAIR_K"]["receptions"]:.2f}× catches, {P["FAIR_K"]["rush_yds"]:.2f}× rush yds · '
          f'lean cutoff {50 + P["LEAN_CONF"]*100:.0f}%</div></div>')
    allpk = [(pk, f'{g["away"]} @ {g["home"]}') for g in R["games"] for pk in g["picks"]]
    allpk.sort(key=lambda x: (x[0]["kind"] != "line", -(x[0]["ev"] or 0), -x[0]["prob"]))
    W('<section class="top"><h2 class="toph">Top picks</h2>')
    if not any(pk["kind"] == "line" for pk, _ in allpk):
        W('<p class="note">No sportsbook lines were loaded for this run, so these are <b>leans</b>. Each one compares the model to the line a book would '
          'normally set from the player\'s season (his typical game, not his average). Before you bet, check the real line: if it\'s already moved past the model\'s median, skip it.</p>')
    W('<ol class="picks">' + "".join(pick_html(pk, gname) for pk, gname in allpk[:9]) + '</ol></section>')
    W('<nav class="toc">' + "".join(f'<a href="#{g["away"]}-{g["home"]}">{g["away"]} @ {g["home"]}</a>' for g in R["games"]) + "</nav>")
    W("</header>")
    for g in R["games"]:
        sp = g["spread"]; fav = g["home"] if sp > 0 else g["away"]
        W(f'<section class="game" id="{g["away"]}-{g["home"]}">')
        W(f'<div class="ghead"><h2>{g["away"]} <i>@</i> {g["home"]}</h2><div class="meta">{e(g["day"])} {e(g["date"])} · {e(g["time"])} ET · '
          f'{fav} −{abs(sp):g} · total {g["total"]:g}</div></div>')
        if g["picks"]:
            W('<div class="gpicks"><h3 class="gph">Best picks</h3><ol class="picks">' + "".join(pick_html(pk) for pk in g["picks"]) + '</ol></div>')
        else:
            W('<p class="note">No strong picks in this game. The model mostly agrees with the averages.</p>')
        for s in g["sides"]:
            G = s["G"]
            W(f'<div class="side"><h3>{s["off"]} offense <span>vs {s["de"]} defense</span></h3>')
            W('<div class="pad-notes">')
            if s["inj_note"]: W(f'<p class="note">{e(s["inj_note"])}</p>')
            W("<ul class='angles'>")
            for kind, txt in s["angles"]:
                W(f'<li class="{kind}">{e(txt)}</li>')
            W("</ul></div>")
            keyinj = [(v, k) for k, v in s["inj"].items() if v["status"] in ("Out", "Doubtful", "Questionable")]
            names = {p["name"] for p in s["props"]}
            W('<div class="tablewrap"><table><thead><tr><th>Player</th><th>Pos</th><th>Status</th><th class="n">Tgt share<br><small>season / L3 / proj</small></th>'
              '<th class="n">Proj tgts</th><th class="n">aDOT</th><th class="n">Matchup</th><th class="n">Catches<br><small>median</small></th>'
              '<th class="n">Rec yds<br><small>median</small></th><th class="n">Rush yds<br><small>median</small></th></tr></thead><tbody>')
            for p in sorted(s["props"], key=lambda x: -(x["proj_targets"] + x["proj_carries"] * 0.5)):
                st = p["status"] if p["status"] not in ("", "—", None) else ""
                if not st and p["practice"] not in ("", "—", None) and "Full" not in str(p["practice"]):
                    st = "Limited" if "Limited" in p["practice"] else "DNP"
                chip = f'<span class="chip {st.lower().split("/")[0]}">{e(st)}</span>' if st else ""
                adot = "" if pd.isna(p["adot"]) else f"{p['adot']:.1f}"
                rush = f"{p['med_rush_yds']:.0f}" if "med_rush_yds" in p else "—"
                mu = p["yf"]; mcls = "up" if mu > 1.05 else "down" if mu < 0.95 else ""
                W(f'<tr><td>{e(p["name"])}</td><td>{p["pos"]}</td><td>{chip}</td>'
                  f'<td class="n">{p["tshare"]*100:.0f} / {p["tshare_l3"]*100:.0f} / <b>{p["proj_share"]*100:.0f}%</b></td>'
                  f'<td class="n">{p["proj_targets"]:.1f}</td><td class="n">{adot}</td>'
                  f'<td class="n {mcls}">{(mu-1)*100:+.0f}%</td>'
                  f'<td class="n">{p.get("med_receptions", float("nan")):.0f}</td>'
                  f'<td class="n">{p.get("med_rec_yds", float("nan")):.0f}</td>'
                  f'<td class="n">{rush}</td></tr>')
            q = s["qb"]
            W(f'<tr class="qb"><td>{e(q["name"])}</td><td>QB</td><td></td><td class="n" colspan="2">{q["med_pass_att"]:.0f} att (median)</td>'
              f'<td></td><td></td><td></td><td class="n" colspan="2">{q["med_pass_yds"]:.0f} pass yds (median)</td></tr>')
            W("</tbody></table></div>")
            if s["edges"]:
                W('<div class="tablewrap"><table class="edges"><thead><tr><th>Bet</th><th class="n">Line</th><th class="n">Model hit %</th><th class="n">Book implied</th><th class="n">EV / $1</th></tr></thead><tbody>')
                for ed in s["edges"][:8]:
                    cls = "pos" if ed["ev"] > 0.05 else ""
                    W(f'<tr class="{cls}"><td>{e(ed["player"])} {ed["side"]} {e(ed["market"])}</td><td class="n">{ed["line"]:g} ({int(ed["odds"]):+d})</td>'
                      f'<td class="n">{ed["prob"]*100:.0f}%</td><td class="n">{ed["implied"]*100:.0f}%</td><td class="n">{ed["ev"]:+.2f}</td></tr>')
                W("</tbody></table></div>")
            W("</div>")
        W("</section>")
    W('<footer><p><b>How to read it.</b> <i>Tgt share</i> is the slice of team targets: season, last 3 games, then projected after injuries and the defense\'s position funnel. '
      '<i>Matchup</i> is how much this defense inflates or deflates yards per target for this player\'s position and field location, '
      'opponent-adjusted (compared to what the same offenses do vs everyone else). Medians come from simulating the game 20,000 times. '
      'If a median is well above the book line, the over hits more than half the time in the sims.</p>'
      f'<p><b>Limits.</b> Free data has no slot/wide alignment, so middle-of-field targets stand in for slot. Man/zone rates are from {R["cov_season"]}, the latest free season. '
      'Rerun after the final injury report (Friday, or Saturday for Monday games). No model is a lock. Small samples early in the season swing a lot.</p>'
      '<p>Data: nflverse (play-by-play, FTN charting, injuries, snap counts). Lines from the schedule feed.</p></footer></div>')
    return "\n".join(out)

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int)
    ap.add_argument("--game", help="e.g. PHI-WAS")
    ap.add_argument("--lines", help="CSV of prop lines")
    ap.add_argument("--odds-key", default=os.environ.get("ODDS_API_KEY"))
    ap.add_argument("--backtest", type=int, help="week to test accuracy on")
    ap.add_argument("--out", default="nfl_edge_report.html")
    ap.add_argument("--quick", action="store_true", help="skip the accuracy check")
    ap.add_argument("--standalone", action="store_true", help="write a full HTML page (for a normal website)")
    ap.add_argument("--odds-markets", default=",".join(ODDS_MARKETS),
                    help="comma list: rec_yds,receptions,rush_yds,pass_yds (fewer = fewer API credits)")
    a = ap.parse_args()
    d = load(a.season)
    if a.backtest:
        bt, B = backtest(d, a.season, a.backtest)
        print(json.dumps(bt, indent=2)); B.to_csv(f"backtest_week{a.backtest}.csv", index=False); return
    week = a.week or current_week(d["games"], a.season)
    ODDS_MARKETS[:] = [m.strip() for m in a.odds_markets.split(",") if m.strip()]
    lines = pd.read_csv(a.lines) if a.lines and os.path.exists(a.lines) and os.path.getsize(a.lines) > 40 else None
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    bt = None
    if not a.quick:
        try:
            bt = tune(d, a.season, week)
            if bt:
                print(f"Tuned on Weeks {bt['weeks']}: {bt['params']}  lean record {bt['wins']}-{bt['losses']}")
                json.dump(dict(bt, by_week=bt["by_week"]), open(os.path.join(os.path.dirname(os.path.abspath(a.out)), "tuning.json"), "w"),
                          indent=1, default=float)
        except Exception as ex: print(f"(self-tuning skipped, using defaults: {ex})", file=sys.stderr)
    R = run_week(d, a.season, week, a.game, lines, a.odds_key, skip_final=True)
    page = render(R, bt)
    if a.standalone:
        title, rest = page.split("\n", 1)
        page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">' + title +
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
                '<style>body{margin:0}img{max-width:100%}</style></head><body>' + rest + '</body></html>')
    open(a.out, "w").write(page)
    pj = [dict(game=f'{g["away"]} @ {g["home"]}', **{k: v for k, v in pk.items() if k != "score"}) for g in R["games"] for pk in g["picks"]]
    json.dump(pj, open(os.path.splitext(a.out)[0] + "_picks.json", "w"), indent=1, default=float)
    print(f"Wrote {a.out}")
    for g in R["games"]:
        for s in g["sides"]:
            for ed in s["edges"][:3]:
                if ed["ev"] > 0.05:
                    print(f"  EDGE: {ed['player']} {ed['side']} {ed['line']} {ed['market']}  model {ed['prob']*100:.0f}% vs book {ed['implied']*100:.0f}%")

if __name__ == "__main__":
    main()
