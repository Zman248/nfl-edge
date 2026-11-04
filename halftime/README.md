# Halftime Edge

Emails you at halftime of men's and women's college basketball games when the live moneyline looks mispriced.

**The edge (backtested on ~36,000 games, 2023-24 to 2025-26):** when a +5 or bigger pregame underdog leads at halftime, live odds overrate the favorite's comeback. The underdog beat its odds by 3–9 points in both men's and women's college, every season. A model trained on those games was then tested on a season it hadn't seen and beat ESPN's live odds. Real sportsbooks are sharper than ESPN, so expect a smaller edge in practice. That's why this starts as **paper betting**.

**Each alert tells you:** the team to bet (moneyline), the model's win chance vs the live odds, the best real price it found (when it can), a suggested stake, and the **worst price you should accept**.

**Record:** every alert is logged in `alerts.csv` and graded after the game. See [RECORD.md](RECORD.md).

## Setup (repo Settings → Secrets and variables → Actions → New repository secret)
- `GMAIL_USER`: your Gmail address
- `GMAIL_APP_PASSWORD`: a Gmail *app password* (Google Account → Security → 2-Step Verification → App passwords). This is not your normal password.
- `ALERT_TO`: optional, where to send alerts (defaults to GMAIL_USER)
- `ODDS_API_KEY`: already set. It's used only when there's a candidate bet, so it stays inside the free 500/month.

## Run it by hand
Actions tab → **Halftime Edge** → Run workflow → mode `selftest` and a past date replays that night's halftimes, which checks that the feeds work.
