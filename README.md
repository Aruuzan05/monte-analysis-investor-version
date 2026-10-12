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
(`jpm_correlations.csv`) and are not editable — they're the one thing in the
model nobody has to guess.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at <http://localhost:8501>. Or run `./run_local.sh`, which sets up a
virtualenv and does both steps for you.

## Running the tests

```bash
python test_engine.py
python test_optimizer.py
python test_universe.py
python test_run_store.py
python test_db.py
```

Run these after any change to `engine.py`, `optimizer.py`, `universe.py`, or
`jpm_ltcma.py`.

## Project layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `engine.py` | Monte Carlo simulation |
| `optimizer.py` | Mean-variance optimization |
| `universe.py` | Optimization over the full 58-line LTCMA universe |
| `run_store.py` | Run records, run fingerprint (digest) and Excel export |
| `db.py` | Database: accounts, login sessions, saved runs (SQLite or Postgres) |
| `manage_users.py` | Command-line account admin (add, reset password, list) |
| `jpm_ltcma.py` | The 58 LTCMA lines and correlation loading |
| `jpm_correlations.csv` | J.P. Morgan's published correlation matrix |
| `requirements.txt` | Pinned dependencies |
| `Dockerfile`, `docker-compose.yml`, `run_local.sh` | Deployment options |

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
python manage_users.py add <username>
python manage_users.py reset <username>
python manage_users.py list
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

## Modelling notes

- Returns use J.P. Morgan's compound (geometric) figures; the engine derives
  the arithmetic return internally (`g + σ²/2`) rather than double-counting
  variance drag.
- Correlations are static — there's no regime-switching or crisis
  correlation spike, so tail outcomes are somewhat optimistic.
- Nominal USD, no fees, no taxes, no external cash flows.
