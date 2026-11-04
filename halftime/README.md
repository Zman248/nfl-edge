# Halftime Edge

Emails you at halftime of men's and women's college basketball games when the live moneyline looks mispriced.

**The idea:** when a pregame underdog (about +3 or more) leads at halftime, the favorite's comeback may be overrated. In ~36,000 games (men's + women's, 2023-24 to 2025-26), those underdogs won more often than a standard community win-probability model expected, every season.

**What's not proven yet:** that **sportsbooks** misprice it. The backtest compared against a free community model, not real live odds. The model here (pregame line + halftime score) is calibrated on a season it never saw, but the real test is against live book prices. That's why this runs as **paper betting** first.

**Each alert tells you:** the team to bet (moneyline), the model's win chance vs the live odds, the best real price it found (when it can), a suggested stake, and the **worst price you should accept**.

**Record:** every alert is logged in `alerts.csv` and graded after the game. See [RECORD.md](RECORD.md).

## Setup (repo Settings → Secrets and variables → Actions → New repository secret)
- `NTFY_TOPIC`: a private channel name only you know (alerts go through the free ntfy.sh service)
- `ALERT_TO`: your email address, so each alert is also emailed to you
- Optional: install the **ntfy** phone app and subscribe to the same topic for push notifications
- (Alternative: `GMAIL_USER` + `GMAIL_APP_PASSWORD` if your Google account offers app passwords)
- `ODDS_API_KEY`: already set. It's used only when there's a candidate bet, so it stays inside the free 500/month.

## Run it by hand
Actions tab → **Halftime Edge** → Run workflow → mode `selftest` and a past date replays that night's halftimes, which checks that the feeds work.
