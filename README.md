# Keepp Capital — Monte Carlo Allocation Simulator

A Streamlit app that simulates a portfolio's future value using Monte Carlo
methods, built on J.P. Morgan's 2026 Long-Term Capital Market Assumptions
(LTCMA). Enter an allocation, and the app runs thousands of simulated
paths to show a range of likely outcomes over 3, 5, and 10 years.

## What it does

- **Base mode** — start from Keepp's target allocation. Returns, volatilities
  and correlations are locked to the J.P. Morgan figures; you can only change
  the weights.
- **Custom mode** — build your own allocation from any of the 58 LTCMA lines
  across three categories (Fixed Income, Equities, Alternatives), with fully
  editable returns and volatilities.
- **Two sets of market assumptions** — switch in the sidebar between J.P.
  Morgan's 2026 LTCMA (58 asset classes) and BlackRock's Capital Market
  Assumptions, August 2026 (34 asset classes, with a base case and two
  scenarios). Each keeps its own portfolio; saved runs remember which one
  they used.
- **Optimizer** — given a set of return/risk assumptions, solves for the
  allocation that maximizes Sharpe ratio, maximizes long-run growth, or hits
  a target return/risk with minimum risk/maximum return.
- **Results** — a percentile table (5th/25th/median/75th/95th) and a wealth
  fan chart for each horizon.
- **Saved runs** — every run is saved automatically to the signed-in user's
  account in a database, so the history survives a refresh, a restart and a
  change of device. Runs can be restored (puts the holdings, figures,
  horizon, number of futures and seed back into the app), renamed, compared,
  deleted (with one-step undo) and exported to Excel.

Correlations come from J.P. Morgan's published 59×59 matrix
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
```

Run these from the repo root after any change to the `keepp_mc` package
(`engine.py`, `optimizer.py`, `universe.py`, `jpm_ltcma.py`, ...).

## Project layout

```
app.py                     Streamlit UI (entry point: `streamlit run app.py`)
keepp_mc/                  Core package — pure Python, no Streamlit
  engine.py                Monte Carlo simulation
  optimizer.py             Mean-variance optimization
  universe.py              Optimization over the full 58-line LTCMA universe
  run_store.py             Run records, run fingerprint (digest) and Excel export
  db.py                    Accounts, login sessions, saved runs (SQLite or Postgres)
  jpm_ltcma.py             The 58 LTCMA lines and correlation loading
  blackrock_cma.py         BlackRock's assumptions: returns by horizon, scenarios, correlation build
  data/
    jpm_correlations.csv   J.P. Morgan's published correlation matrix
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
- Correlations are static — there's no regime-switching or crisis
  correlation spike, so tail outcomes are somewhat optimistic.
- Nominal USD, no fees, no taxes, no external cash flows.
