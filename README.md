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
  fan chart for each horizon, plus the ability to save and export runs to
  Excel.

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
| `run_store.py` | Saved-run history and Excel export |
| `jpm_ltcma.py` | The 58 LTCMA lines and correlation loading |
| `jpm_correlations.csv` | J.P. Morgan's published correlation matrix |
| `requirements.txt` | Pinned dependencies |
| `Dockerfile`, `docker-compose.yml`, `run_local.sh` | Deployment options |

## Password protection

The app reads an optional shared password from `st.secrets`. To set one,
create `.streamlit/secrets.toml` (already gitignored):

```toml
app_password = "something-long"
```

With no password set, the app runs open — fine for local use, but set one
before exposing it on a public server.

## Deploying

- **Streamlit Community Cloud**: push this repo to GitHub, create a new app
  pointing at it with main file `app.py`, and set `app_password` under
  **Manage app → Settings → Secrets**.
- **Self-hosted (Docker)**: `docker compose up -d` on any server, then visit
  it at `http://<server-ip>:8501`.

The app makes no external network calls at runtime (no CDNs, no web fonts,
no telemetry), so it loads reliably from anywhere, including behind
restrictive firewalls.

## Modelling notes

- Returns use J.P. Morgan's compound (geometric) figures; the engine derives
  the arithmetic return internally (`g + σ²/2`) rather than double-counting
  variance drag.
- Correlations are static — there's no regime-switching or crisis
  correlation spike, so tail outcomes are somewhat optimistic.
- Nominal USD, no fees, no taxes, no external cash flows.
