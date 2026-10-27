# NFL Edge

Free-data version of the notepad prop research from the video. It pulls every stat from nflverse (free, public NFL data) and builds a report for each game:

- **Volume**: projected pass attempts and carries, adjusted for the spread and total
- **Injuries**: who's out/limited, and how their targets get redistributed
- **Defense weak spots**: yards allowed to WRs/TEs/RBs and middle vs outside, compared to what those same offenses do against everyone else (so a defense isn't punished for facing good teams)
- **Blitz rate** (this season) and **man coverage rate** (2025, the latest free season)
- **Projections**: 20,000 simulated games per player for catches, receiving yards, rushing yards, and QB attempts/yards
- **Edges**: when you give it prop lines, it shows the hit % vs the book's implied % and the expected value

## Setup (one time)

You need Python 3.10+.

```
pip install pandas pyarrow numpy
```

## Run it

```
python3 nfl_edge.py --week 8                     # every game that week
python3 nfl_edge.py --week 8 --game PHI-WAS      # one game
python3 nfl_edge.py --week 8 --lines lines.csv   # compare against your prop lines
python3 nfl_edge.py --week 8 --quick             # skip the accuracy check (faster)
python3 nfl_edge.py --backtest 7                 # score the model on a past week
```

Open `nfl_edge_report.html` in your browser. The first run downloads about 20 MB of data into `nfl_cache/`. Later runs reuse it for 6 hours.

## Adding prop lines

Copy lines from your sportsbook app into `lines.csv` (template included):

```
player,market,line,over_odds,under_odds
Puka Nacua,rec_yds,85.5,-115,-105
```

Markets: `rec_yds`, `receptions`, `rush_yds`, `pass_yds`, `pass_att`.

Or get a free key from the-odds-api.com and run with `--odds-key YOUR_KEY` to pull lines automatically. The free tier has a monthly request limit, so pull one game at a time. The automatic pull couldn't be tested from where this tool was built. If it errors, use the CSV.

## When to run it

- **Tuesday/Wednesday**: first look, once last week's data is in
- **Friday** (Saturday for Monday games): rerun after the final injury report. This is the run that matters.

## Self-tuning

Every run, before making picks, the tool:
1. Re-projects every finished week (Week 4 on) using only the data from before that week
2. Grades those projections against what actually happened
3. Re-fits its settings to whatever worked best: how much to trust defense matchups, where books' "fair" lines sit, and how confident a lean needs to be
4. Posts a walk-forward record, where each week is graded with settings learned only from earlier weeks, so the record isn't cheating

The settings and record are shown at the top of the site and saved to `tuning.json`. Runs get about 20 seconds longer per finished week, which tops out around 5 minutes late in the season.

## How good is it?

Tested on Weeks 4–7 of 2026, with each week projected using only earlier data:

- Receiving yards: off by about 25 per player, vs 26.5 using season averages
- When it disagreed with a player's season average by 8+ yards, it picked the right side about 65% of the time

Sportsbook lines are sharper than season averages, so expect a smaller edge against real lines. Only bet the biggest gaps (model hit % at least 8–10 points above the book's implied %), and track your results.

## Limits

- Free data doesn't say who lined up in the slot. Targets over the middle stand in for slot targets.
- Man/zone coverage stopped being free after 2025, so those rates are last season's.
- With only a few games played, defense splits are mostly noise. The model gives them a small weight on purpose.
