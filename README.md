# Keepp Capital — Monte Carlo Allocation Simulator

A Streamlit app that simulates a portfolio's future value using Monte Carlo
methods, built on J.P. Morgan's 2027 Long-Term Capital Market Assumptions
(LTCMA). Enter an allocation, and the app runs thousands of simulated
paths to show a range of likely outcomes over 3, 5, and 10 years.

## What it does

- **Base mode** — start from Keepp's target allocation. Returns, volatilities
  and correlations are locked to the J.P. Morgan figures; you can only change
  the weights.
- **Custom mode** — build your own allocation from any of the 59 LTCMA lines
  across three categories (Fixed Income, Equities, Alternatives), with fully
  editable returns and volatilities. A **Keepp vs provider comparison** panel
  re-runs the same weights with the provider's published figures so you can
  see the exact dollar impact of differing assumptions (common random numbers
  keep simulation noise near zero).
- **Two sets of market assumptions** — switch in the sidebar between J.P.
  Morgan's 2027 LTCMA (59 asset classes, including Digital Assets) and
  BlackRock's Capital Market Assumptions, August 2026 (34 asset classes, with
  a base case and two scenarios). Each keeps its own portfolio; saved runs
  remember which one they used.
- **Optimizer** — given a set of return/risk assumptions, solves for the
  allocation that maximizes Sharpe ratio, maximizes long-run growth, or hits
  a target return/risk with minimum risk/maximum return.
- **Liquidity tiers** — every line is assigned an L1–L5 tier (Cash → Locked)
  with an estimated time-to-cash and a crisis haircut. The optimizer can
  enforce a minimum liquid fraction; the allocation panel shows a stacked bar
  of the portfolio's liquidity profile.
- **Return reconciliation** — an expandable panel breaks down why the
  simulator's compound return differs from the weighted average of the Return
  column, splitting the gap into a dispersion term (spread of sleeve
  volatilities) and a diversification term (correlations below 1).
- **Results** — a percentile table (5th/25th/median/75th/95th) and a wealth
  fan chart for each horizon.
- **Saved runs** — pressing *Save scenario* at the bottom of Your mix saves
  the run (numbered 1, 2, 3, ... in the list) to the signed-in user's
  account in a database, so the history survives a refresh, a restart and a
  change of device. Runs can be restored (puts the holdings, figures,
  horizon, number of futures and seed back into the app), renamed, compared,
  deleted (with one-step undo) and exported to Excel.

Correlations come from J.P. Morgan's published 60×60 matrix
(`keepp_mc/data/jpm_correlations.csv`) and are not editable — they're the one thing in the
model nobody has to guess.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at <http://localhost:8501>. Or run `./scripts/run_local.sh`, which sets up a
virtualenv and does both steps for you.

## Running the tests

```bash
python tests/test_engine.py
python tests/test_optimizer.py
python tests/test_universe.py
python tests/test_run_store.py
python tests/test_db.py
python tests/test_blackrock.py
python tests/test_compare.py
python tests/test_liquidity.py
python tests/test_geography.py
```

Run these from the repo root after any change to the `keepp_mc` package
(`engine.py`, `optimizer.py`, `universe.py`, `jpm_ltcma.py`, ...).

## Project layout

```
app.py                     Streamlit UI (entry point: `streamlit run app.py`)
keepp_mc/                  Core package — pure Python, no Streamlit
  engine.py                Monte Carlo simulation + return_ladder reconciliation
  optimizer.py             Mean-variance optimization
  universe.py              Optimization over the full 59-line LTCMA universe
  run_store.py             Run records, run fingerprint (digest) and Excel export
  db.py                    Accounts, login sessions, saved runs (SQLite or Postgres)
  jpm_ltcma.py             The 59 LTCMA lines and correlation loading
  blackrock_cma.py         BlackRock's assumptions: returns by horizon, scenarios, correlation build
  compare.py               Keepp-vs-provider two-view comparison (common random numbers)
  liquidity.py             L1–L5 liquidity tiers, time-to-cash and crisis haircuts
  geography.py             Geography look-through and region limits
  personal_lines.py        Personal PE / PC lines with proxy correlations
  data/
    jpm_correlations.csv   J.P. Morgan's published 60×60 correlation matrix
    blackrock_cma.csv      The USD rows extracted from BlackRock's workbook
tests/                     One script per module (run with `python tests/test_*.py`)
scripts/
  manage_users.py          Command-line account admin (add, reset password, list)
  run_local.sh             Set up a virtualenv and run without Docker
docs/images/               Screenshots
.streamlit/                Streamlit theme (config.toml) and secrets template
data/                      Local SQLite database, created at runtime (gitignored)
requirements.txt           Pinned dependencies
Dockerfile, docker-compose.yml   Container deployment
```

## Accounts and the database

Everyone signs in with a username and password. Runs belong to the account
that made them; nobody sees anyone else's runs. Passwords are stored as
scrypt hashes, and a sign-in lasts 30 days in that browser (a cookie holding
a random session token) until **Sign out**.

Settings go in `.streamlit/secrets.toml` (gitignored; see
`secrets.toml.example`):

```toml
[database]
url = "postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require"

[auth]
signup_code = "something-long"
```

- **`[database] url`** — leave it out and the app uses a local SQLite file,
  `data/keepp.db`. That is right for your laptop and for Docker (the file
  sits on a volume). **On Streamlit Community Cloud it is not**: the
  container's disk is wiped on every reboot and redeploy, so use a hosted
  Postgres database there (a free Neon or Supabase project works) and paste
  its connection string. The tables are created automatically on first start.
- **`[auth] signup_code`** — the invitation code needed to create an account
  on the sign-in page. With no code set, only the first account can be
  created in the app. A legacy `app_password` is accepted as the code.

Account admin from a terminal (uses the same database as the app):

```bash
python scripts/manage_users.py add <username>
python scripts/manage_users.py reset <username>
python scripts/manage_users.py list
```

## Deploying

- **Streamlit Community Cloud**: push this repo to GitHub, create a new app
  pointing at it with main file `app.py`, and paste the `[database]` and
  `[auth]` settings under **Manage app → Settings → Secrets**.
- **Self-hosted (Docker)**: `docker compose up -d` on any server, then visit
  it at `http://<server-ip>:8501`. Accounts and runs live in SQLite on the
  `keepp-data` volume, so no separate database server is needed — which also
  keeps the zero-external-request property for hosting in mainland China.

The app makes no external network calls at runtime (no CDNs, no web fonts,
no telemetry), so it loads reliably from anywhere, including behind
restrictive firewalls. The one exception is the database connection when you
point `[database] url` at a hosted Postgres; pick a provider region your
users can reach.

## BlackRock assumptions

Taken from BlackRock's Excel workbook
(`blackrock-capital-market-assumptions.xlsx`), US-dollar rows only. To load a
new quarter's edition:

```bash
python -m keepp_mc.blackrock_cma path/to/blackrock-capital-market-assumptions.xlsx
python tests/test_blackrock.py
```

That rewrites `keepp_mc/data/blackrock_cma.csv`; commit it. How it differs from J.P. Morgan:

- **Returns depend on the horizon.** BlackRock publishes 5- to 30-year
  figures; the app uses the one matching the number of years chosen,
  interpolating between published horizons (under 5 years uses the 5-year
  figure).
- **No full correlation matrix is published** — only each asset's
  correlation to government bonds and to equities. Those set the common part
  of every pair exactly; the "family" resemblance two factors cannot capture
  (e.g. levered vs unlevered direct lending) comes from J.P. Morgan's matrix
  via the closest J.P. Morgan line for each BlackRock asset (`CORR_PROXY` in
  `keepp_mc/blackrock_cma.py`).
- **Scenarios** ("AI productivity boom", "Global risk premia rise") change
  expected returns only.

## Modelling notes

- Returns use J.P. Morgan's compound (geometric) figures; the engine derives
  the arithmetic return internally (`g + σ²/2`) rather than double-counting
  variance drag.
- The simulator's median compound return is always at or above a weighted
  average of the Return column. `engine.return_ladder` breaks the gap into
  two non-negative components: **dispersion** (spread of sleeve volatilities,
  present even at ρ = 1) and **diversification** (what correlations below 1
  add on top). The reconciliation panel in the app shows each step.
- The **Keepp vs provider** comparison (`compare.py`) runs the same portfolio
  twice — once with Keepp's return assumptions, once with the provider's
  published figures — using identical random shocks ("common random numbers"),
  so the difference in outcomes is almost entirely from the assumptions, not
  simulation noise.
- Correlations are static — there's no regime-switching or crisis
  correlation spike, so tail outcomes are somewhat optimistic.
- Nominal USD, no fees, no taxes, no external cash flows.
