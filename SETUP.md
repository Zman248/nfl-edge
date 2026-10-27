# Put NFL Edge on its own website (free, runs itself)

GitHub runs the script on a schedule and hosts the report at
`https://YOUR-USERNAME.github.io/nfl-edge/`. You don't need Claude or your laptop.

## 1. Make the repo (5 min)
1. Make a free account at github.com if you don't have one.
2. Click **+** (top right), then **New repository**. Name it `nfl-edge` and set it to **Public**. Free accounts only get websites from public repos.
3. Click **uploading an existing file**. Drag in everything from this folder, including the `.github` folder, then click **Commit changes**.
   - If the `.github` folder doesn't upload by dragging: click **Add file → Create new file**, type `.github/workflows/update.yml` as the name, paste in the contents of that file, and commit.

## 2. Turn on the website
1. In the repo, go to **Settings → Pages**.
2. Under **Source**, pick **GitHub Actions**.

## 3. Run it the first time
1. Go to the **Actions** tab. If it asks, click **I understand, enable workflows**.
2. Click **Update NFL Edge → Run workflow**.
3. Wait about 3–5 minutes for the green check. Your site is the link under **Settings → Pages**.

From then on it rebuilds by itself Wednesday night, Friday night, and Sunday at noon ET.

## 4. Optional: real sportsbook lines
1. Get a free key at the-odds-api.com.
2. In the repo, go to **Settings → Secrets and variables → Actions → New repository secret**. Name it `ODDS_API_KEY` and paste the key.
3. That's it. Each run now pulls lines, and the picks become "vs book line" with expected value.

Free-tier math: each run pulls 3 prop types for about 14 games, roughly 45 credits. Three runs a week is about 540 credits a month, and the free limit is 500. To stay under, change `--odds-markets` in the workflow to `rec_yds,rush_yds`, or delete the Wednesday line from the schedule.

You can also paste lines by hand. Edit `lines.csv` on GitHub (pencil icon), add rows, commit, then click **Run workflow**.

## Notes
- The schedule is in UTC and set for daylight time. After the clocks change (Nov 1), runs happen an hour earlier ET. That's fine.
- GitHub sometimes runs scheduled jobs 5–20 minutes late.
- If a run fails, open it in the Actions tab to see the error. It's usually a temporary download problem, so just rerun it.
- A public repo means anyone with the link can see the site. Private sites need a paid GitHub plan.
