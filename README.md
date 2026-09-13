# Keepp Capital — Monte Carlo Allocation Simulator

A Streamlit app for the target-allocation Monte Carlo. It runs in two modes:

- **Base · JPM 2026 LTCMA** — the sleeve list, returns and volatilities are
  locked to the figures behind the July 2027 deck; **the weights are editable.**
  Re-weighting the deck's own assumptions is the question a base case is for
  ("what if we took equities to 25?"), and answering it here keeps the answer on
  figures nobody has touched. *Reset weights to the deck* restores the shipped
  mix.
- **Custom** — weights, returns and volatilities are all editable, and each of
  **J.P. Morgan's three categories** (Fixed Income, Equities, Alternatives)
  gets its own table and its own dropdown scoped to that category's LTCMA
  lines. A category can hold any number of sleeves.

Correlations are **not** editable and not assumed: they come from J.P. Morgan's
published 59 × 59 matrix, shipped as `jpm_correlations.csv`.

Either way the app returns the 3y/5y/10y percentile table and a wealth fan
chart, and weights must total 100% before it will run — *Normalize to 100%*
scales them proportionally in either mode. Switching between modes keeps both
sets of inputs: Base keeps the JPM figures and your weights, Custom remembers
your edits, and neither disturbs the other.

Three categories, 58 LTCMA lines: Fixed Income (21), Equities (18),
Alternatives (19).

Built to run on a mainland-China host and load without a VPN.

---

## Quick start (local, any OS)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at <http://localhost:8501>. On a machine in China, use the mirror:

```bash
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
```

Or just run `./run_local.sh`, which does both steps in a virtualenv.

---

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repo. `.gitignore` already keeps the
   virtualenv, caches and — critically — `.streamlit/secrets.toml` out of it.
2. At <https://share.streamlit.io> choose **New app**, point it at the repo,
   and set **Main file path** to `app.py`.
3. Set the password under **Manage app → Settings → Secrets**, pasting:

   ```toml
   app_password = "something-long"
   ```

   Community Cloud injects that into `st.secrets` at runtime. Do not commit a
   `secrets.toml` file — the app reads the same key either way.
4. Deploy. Dependencies come from `requirements.txt`; `.streamlit/config.toml`
   supplies the theme.

`Dockerfile`, `docker-compose.yml` and `run_local.sh` are unused on Community
Cloud but are kept in the repo so the app can be moved to a self-hosted server
without rebuilding anything.

**Reachability note:** Community Cloud runs on AWS/GCP, which is slow or
blocked from mainland China. If a manager there cannot load the app, the
self-hosting path below is the fix, not a code change.

---

## Self-hosting instead (Alibaba Cloud / Tencent Cloud)

### 1. Pick the instance

An ECS `ecs.t6-c1m2.large` (2 vCPU / 4 GB) or a Tencent CVM S5.MEDIUM4 is more
than enough. A 10,000-path run takes about 200 ms and a few hundred MB. Choose a
mainland region close to your managers — Beijing, Shanghai or Hangzhou.

### 2. Open the port

In the instance's **security group**, allow inbound TCP **8501** from your
office IP range. Do not open it to `0.0.0.0/0` — see Access control below.

### 3. Install Docker with a registry mirror

Docker Hub is throttled from the mainland, so point the daemon at Aliyun's
mirror first:

```bash
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<'EOF'
{ "registry-mirrors": ["https://registry.cn-hangzhou.aliyuncs.com"] }
EOF
sudo systemctl restart docker
```

### 4. Ship the code and start it

```bash
scp -r keepp-streamlit root@<your-server-ip>:/opt/
ssh root@<your-server-ip>
cd /opt/keepp-streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit secrets.toml and set a real password
docker compose up -d
```

Visit `http://<your-server-ip>:8501`.

Update after a code change: `docker compose up -d --build`.

### Without Docker

```bash
sudo apt install -y python3-venv
cd /opt/keepp-streamlit && ./run_local.sh          # to test it
```

Then as a service:

```ini
# /etc/systemd/system/keepp-mc.service
[Unit]
Description=Keepp Monte Carlo Simulator
After=network.target

[Service]
User=keepp
WorkingDirectory=/opt/keepp-streamlit
ExecStart=/opt/keepp-streamlit/.venv/bin/streamlit run app.py \
  --server.address=0.0.0.0 --server.port=8501 --server.headless=true
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now keepp-mc
```

---

## The ICP filing question

**Serving a domain name from a mainland server requires an ICP filing (ICP备案).**
That means real-name registration and a Chinese business licence, and it
typically takes two to four weeks.

**You almost certainly do not need one.** ICP filing attaches to *domain names*.
An internal tool reached at `http://<ip>:8501` by three known people does not
need a filing, and port 8501 is not subject to the port-80/443 blocking that
mainland providers apply to unfiled domains.

So: use the raw IP and port, put a password on it, and skip the filing. Only
start the ICP process if you later want `mc.keeppcapital.com` to resolve to a
mainland host. (Hosting the same code in Hong Kong or Singapore also avoids
ICP entirely and stays reachable from the mainland, at the cost of some
latency — worth knowing as a fallback.)

---

## Why this loads in China when the previous version might not

The app makes **zero external network requests at runtime**. That is a
deliberate property, verified by loading the page in a headless browser and
logging every request — the only host contacted is the app's own server.

What that required:

| Risk | What was done |
|---|---|
| Google Fonts is blocked | System font stacks only, including `PingFang SC` / `Microsoft YaHei` so a Chinese-locale OS renders properly. `theme.font` in `config.toml` stays on a built-in. |
| Streamlit usage telemetry | `gatherUsageStats = false`. The Segment endpoint is unreachable from the mainland and the failed request delays first paint. |
| Chart library CDNs | Altair/Vega ships inside Streamlit's own static bundle. No Plotly, no CDN `<script>`. |
| Slow/blocked pypi.org | Build and install through the Aliyun PyPI mirror. |
| Throttled Docker Hub | Base image pulled from Aliyun's registry mirror. |

**If you edit `app.py`, do not add a webfont `<link>`, a CDN `<script>`, or a
remote image.** Any one of them will hang the page load behind the firewall.
The comment at the top of `app.py` says the same thing.

To re-verify after a change, load the app and watch the network tab: every
request should go to your own host.

---

## Access control

The app reads one shared password from `st.secrets`. On Community Cloud set
it in **Manage app → Settings → Secrets**; when self-hosting, create
`.streamlit/secrets.toml` (gitignored):

```toml
app_password = "something-long"
```

If the file is absent or the value empty, the app runs with no password. That
is fine on a laptop, but set one before exposing the app on a public IP — the
simulator itself holds no client data, but your return and volatility
assumptions are not something to leave open to the internet. Combine it with a
security-group rule limiting inbound 8501 to your office range.

---

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI. No network calls — see the header comment. |
| `engine.py` | The simulation. Pure numpy, no Streamlit import, unit-testable. |
| `jpm_correlations.csv` | J.P. Morgan's published 59 × 59 USD correlation matrix, unmodified. The source of truth — diff it against the report. |
| `jpm_ltcma.py` | The 58 LTCMA lines in three categories, the correlation loader and Higham repair, and the sleeve sync rules. No Streamlit import. Update here for the 2027 LTCMA — every line must appear in the CSV and in a category, or the tests fail. |
| `optimizer.py` | Mean-variance optimization over the allocation lines. Pure numpy/scipy, no Streamlit import. |
| `test_engine.py` | 54 checks: workbook reconciliation, matrix integrity and repair size, sub-matrix PSD over 300 random draws, category coverage, and the sleeve sync rules. |
| `test_optimizer.py` | 42 checks: the closed-form tangency match, the two-asset analytic minimum-variance weight, frontier monotonicity, reconciliation against the stat cards, and named errors on infeasible constraints. |
| `requirements.txt` | Pinned. Verified to resolve and reproduce the workbook. |
| `Dockerfile` | Mirrors are build args, so it works inside and outside China. |
| `docker-compose.yml` | One-command deploy. |
| `run_local.sh` | Virtualenv + run, no Docker. |
| `.streamlit/config.toml` | Theme, telemetry off, XSRF on. Committed. |
| `.gitignore` | Keeps the virtualenv, caches and `secrets.toml` out of git. |

Run the reconciliation any time you change the engine or the optimizer:

```bash
python test_engine.py
python test_optimizer.py
```

---

## Base assumptions

| LTCMA line | Weight | Return | Vol | |
|---|---|---|---|---|
| U.S. Inv Grade Corporate Bonds | 45% | 5.00% | 7.02% | *Keepp — global IG blend* |
| U.S. Cash | 17% | 3.80% | 0.67% | *Keepp — 3M T-bill* |
| AC World Equity | 20% | 12.88% | 14.47% | *Keepp — MSCI World 10y* |
| Private Equity | 10% | 10.20% | 19.78% | J.P. Morgan |
| Diversified Hedge Funds | 0% | 5.30% | 5.75% | J.P. Morgan |
| Commodities | 8% | 4.60% | 18.32% | J.P. Morgan |

The weights above are the deck's and are what *Reset weights to the deck*
restores; in Base they are editable and everything else in the table is not.

Every sleeve sits on a named LTCMA line, and that line selects the sleeve's row
of the published correlation matrix. Three sleeves carry Keepp's own return and
volatility rather than J.P. Morgan's; they are filed under the line they most
closely are. That last column is documentation, not part of the app — the table
in the UI shows four columns and no provenance column; the three Keepp-sourced
sleeves are named in a caption underneath it and in the sources panel.

### Compound vs arithmetic — the correction that matters

The LTCMA publishes both a compound (geometric) and an arithmetic return per
line. **The simulator takes the compound one**: `engine.py` uses
`m = ln(1+g)/12` and derives the arithmetic return itself as `g + σ²/2`.
Feeding the arithmetic column in instead double-counts the variance drag.

The original workbook did exactly that for two sleeves, and both are corrected
here:

| Sleeve | Workbook | JPM compound | JPM arithmetic |
|---|---|---|---|
| Private Equity | 11.90% | **10.20%** | 11.91% |
| Commodities | 6.10% | **4.60%** | 6.15% |

Effect on the base case: expected return 7.80% → **7.51%**, 10y median wealth
2.059× → **2.005×**. The hedge fund volatility also moved, from a 6.50%
estimate to J.P. Morgan's published **5.75%**.

**Hedge funds carry a 0% base weight on purpose.** The deck's target allocation
has no hedge fund sleeve, so a zero weight keeps the Base case numerically
identical to the signed-off workbook. Allocate to it in Custom mode.

The hedge fund correlation row (EQ 0.75, PE 0.65, FI 0.35, CMD 0.35) is
LTCMA-style convention, not a published J.P. Morgan matrix. It is flagged in
the app's sources panel and is editable in Custom mode.

### Correlations — the published matrix

`jpm_correlations.csv` is J.P. Morgan's 2026 LTCMA USD correlation matrix,
59 × 59 (the 58 investable lines plus U.S. Inflation), committed **unmodified**
so it can be diffed against the report. The matrix used in any run is a
sub-matrix of it, selected by each sleeve's correlation row. Nothing is
assumed anywhere in the correlation path.

This replaced a fabricated 6 × 6 plus six guessed within-type numbers. Of the
21 correlation inputs that structure needed, exactly one had been estimated
from data; the rest were convention or judgment. All of it is gone, along with
`engine.expand_correlation()` and the two editable correlation tables — about
80 lines deleted.

**One numerical note.** As published the matrix is symmetric with a unit
diagonal, but it is *not* positive semi-definite: its smallest eigenvalue is
−0.0161. That is normal for figures rounded to two decimals, and is compounded
by U.S. Core Real Estate and U.S. Value-Added Real Estate sharing one
underlying series — identical rows force a zero eigenvalue. Cholesky needs
positive definiteness, so the matrix is repaired once at import with Higham's
nearest-correlation-matrix algorithm.

The repair is cosmetic and the test suite asserts it:

| | |
|---|---|
| Largest single cell moved | **0.0066** (U.S. Large Cap ↔ AC World Equity, 0.97 → 0.9634) |
| Pairs moving 0.01 or more | **0 of 1,711** |
| Min eigenvalue after | −3.3 × 10⁻¹⁵ (numerically zero) |

Repairing the whole matrix once, rather than each sub-matrix at run time, also
means **every sub-matrix the app can build is automatically valid** — every
principal sub-matrix of a PSD matrix is itself PSD. The test suite draws 300
random sub-matrices and confirms none is negative. The repair warning in the
UI is now a guard, not an expected path.

### The line is the sleeve's identity

**There is nothing to choose on the correlation path.** A sleeve's LTCMA line
does two jobs at once: it supplies Return and Vol, and it selects the sleeve's
row of the published matrix. Overwrite either number and the sleeve keeps your
figure and stays on its line.

| Sleeve | Return / Vol | Filed under |
|---|---|---|
| Global IG blend | 5.00 / 7.02 | U.S. Inv Grade Corporate Bonds |
| MSCI World 10y | 12.88 / 14.47 | AC World Equity |
| 3M U.S. T-bill | 3.80 / 0.67 | U.S. Cash |

An earlier build made these three anonymous "Manual entry" rows, which detached
them from every line and so required a second, hand-picked **Correlation proxy**
column — the one input on the correlation path that was neither published nor
derived, and one a portfolio manager could not answer from the numbers in front
of them. Both are gone. Filing a sleeve under a line answers the question once,
at the point where the user is already naming the asset.

The correlation rows are unchanged by the switch: the three proxies were exactly
these three lines, so Base's return, volatility and percentiles are identical to
the previous build.

**Two alternatives that were tried and rejected.** Defaulting each hand-entered
sleeve to its category's broadest line (Fixed Income → U.S. Aggregate Bonds) is
simpler but turns the T-bill sleeve into a bond: portfolio volatility falls
7.58% → 6.94% and P(loss) 0.18% → 0.04%, a silent understatement of risk.
Inferring the line from the return and volatility typed in works for the IG
blend and the T-bill but matches MSCI World realised history to **Japanese
Equity** — the only line JPM forecasts high enough to be near 12.88% — under
every weighting of the two figures. Return and volatility do not identify an
asset.

### What the real matrix changed

Same weights, returns and volatilities — only the correlations:

| Pair | Fabricated | JPM published |
|---|---|---|
| Fixed Income × Equities | 0.68 | **0.56** |
| Equities × Commodities | 0.30 | **0.50** |
| Private Equity × Hedge Funds | 0.65 | **0.76** |
| Equities × Private Equity | 0.80 | 0.81 |

Net effect on the base case: expected return **unchanged at 7.51%** (correlations
do not enter the return), portfolio volatility **7.40% → 7.58%**, 10-year P(loss)
**0.12% → 0.17%**, 5th-percentile wealth 1.359× → 1.340×. The workbook's
equity–bond figure was too high and its equity–commodity figure much too low;
they partly offset.

### Multiple sleeves per category

The allocation table is a flat list of sleeves grouped into the three
categories. Equities can be 12% U.S. Large Cap + 8% Emerging Markets Equity
while still reading as one 20% equity allocation, and the correlation between
those two sleeves is now J.P. Morgan's published 0.73 rather than a guess. The
**+** button on a category's table adds a sleeve; selecting a row and pressing
delete removes it. Each category heading carries its own subtotal, sleeve count
and the number of LTCMA lines available.

### Sub-category dropdowns

In Custom mode each bucket can be set to any LTCMA line from its own family —
21 fixed income, 18 equity, and the 19 alternatives split into private markets,
hedge funds and real assets. Type in the cell to filter. Picking a line fills
Return and Vol from the compound column.

Four rules govern the table:

1. Picking a different line overwrites Return and Vol from the catalogue,
   discarding any previous override — picking a line is how you ask for J.P.
   Morgan's figures.
2. Editing Return or Vol does **not** disturb the line. The sleeve keeps your
   numbers and stays on its line, which is what supplies its correlation row.
   The line is a filing decision, not a claim that the figures beside it are
   J.P. Morgan's — pick a line again (rule 1) to pull their numbers back.
3. A row added by the editor arrives blank; it lands on the category's default
   line at a 0% weight with that line's published figures, so it can neither
   poison the weight total nor sit there without a correlation row.
4. A stored line that is not in its group's list — reachable from stale session
   state after a catalogue change, or from a saved "Manual entry" row predating
   this design — falls back to the category default rather than sticking.

Rule 1 compares row positions against the previous render, which only holds
while the rows line up. Deleting a row shifts everything below it up by one, so
on a shrinking table rule 1 is skipped entirely; otherwise a delete would
silently reset the edits of every sleeve underneath it.

Swapping a line **does** change the sleeve's correlation row, because the line
is the row. Nothing else needs adjusting.

**Worth knowing before a demo:** every forward-looking JPM equity line sits
5–6pp below the 12.88% realised MSCI World figure, with higher volatility.
Switching Public Equities to `AC World Equity` (7.00% / 16.78%) moves 10y
median wealth from 2.005× to 1.808×. It is the single biggest lever in the
model.

The 6x6 correlation matrix is verified positive semi-definite (minimum
eigenvalue 0.067), so it factorises without repair.

---

## Optimizer — the reverse question

The simulation answers *"what does this allocation produce?"*. The Optimizer
section answers the reverse: **given these return and volatility assumptions,
what is the best achievable return-and-volatility pair, and which weights get
there?**

**Decision variables are the individual LTCMA line weights**, not category
totals. Constraints: long-only, summing to 100%, a cash floor on the `U.S. Cash`
line, a per-line cap, and min/max bands on each J.P. Morgan category. Defaults
are a 5% cash floor, no per-line cap, and category bands at the current totals
±10pp.

Four objectives:

| Objective | Solves |
|---|---|
| Maximum Sharpe ratio | the tangency portfolio — best return per unit of risk above `r_f` |
| Maximum long-run growth | `max wᵀμ − ½wᵀΣw` — the highest **median** wealth, which is what the fan chart plots |
| Minimum volatility for a target return | `min wᵀΣw` s.t. `wᵀμ = r` |
| Maximum return for a volatility budget | `max wᵀμ` s.t. `wᵀΣw ≤ b²` |

Outputs: the two headline numbers, the efficient frontier with the current and
optimal portfolios plotted on it, a weights table with category subtotals sorted
by absolute change, and an **Apply optimal weights** button that writes the
solution back into the allocation and re-runs the Monte Carlo. An optional
stability check re-solves 40 times with the expected returns perturbed and
reports how far each line's optimal weight moves.

Available in **both** modes. Base allows weight edits and nothing else, and the
optimizer's decision variables are exactly weights, so it fits Base without
violating it — there it optimizes against the locked J.P. Morgan figures.

### μ must be the ARITHMETIC return — the trap in this feature

Mean-variance optimization is defined on **arithmetic** expected returns. The
engine is not: `engine.py` uses `m = ln(1+g)/12`, so the app's `Return %` column
is consumed as a **compound (geometric)** return and the arithmetic figure is
derived internally as `g + σ²/2`.

`optimizer.inputs_from_percent()` is the single place that conversion happens.
Feeding the compound column straight in would understate every risky line's
expected return in proportion to its variance — under-allocating to exactly the
assets the optimizer exists to size.

Two consequences worth knowing:

- The optimizer's return and volatility for the *current* weights reproduce the
  `EXPECTED RETURN` and `PORTFOLIO VOLATILITY` stat cards **exactly** (asserted
  in `test_optimizer.py` to 1e-12). If they ever diverge, the conversion broke.
- The engine's `g + σ²/2` is close to but not identical to J.P. Morgan's own
  published arithmetic column — for Private Equity it gives 12.16% where JPM
  prints 11.91%. The optimizer matches the engine, because the engine is what
  produces the numbers on the page.

### Why the solver is trust-constr and not SLSQP

Every problem here is a quadratic program, so the Hessian is a constant matrix
that can be handed to the solver. Measured on the same seven random long-only
problems, worst-case weight error against the closed-form tangency:

| Method | Worst error |
|---|---|
| SLSQP | 9.05e-02 |
| trust-constr | 3.14e-08 |
| trust-constr + exact KKT polish | **5.6e-16** |

Three failure modes were found and fixed while building this, all of which
produced confident, wrong answers rather than errors:

1. **SLSQP declares success at a vertex.** Started from an LP vertex it solves a
   linearised subproblem with no descent direction, reports "Optimization
   terminated successfully" after one iteration, and returns the start — a
   22.07% minimum-variance volatility where 0.12% was available. `feasible_start`
   now averages several LP vertices to return an interior point.
2. **`ftol` is absolute.** Portfolio variance is ~0.006, and near-riskless
   portfolios ~1e-6, so against `ftol=1e-12` the solver can converge before it
   moves. The objective is now scaled to O(1).
3. **A warm start from a neighbouring target return can violate the new return
   constraint** badly enough to fail — measured at 10 failures in 50 targets,
   clustered around the optimum. Frontier points now start from a convex blend
   of the two endpoints, which satisfies the target exactly by construction.

Solver tolerances are deliberately loose (~17ms a solve); the accuracy comes
from the KKT polish afterwards, which needs only the correct active set.

### Caveat shown in the app

> Mean-variance solutions are highly sensitive to the return assumptions typed
> above: a 50bp change in one line's expected return can move its optimal weight
> by tens of percentage points. Treat the frontier as a diagnostic of where the
> current allocation sits, not as a target allocation. Correlations here are
> static — they do not rise in a crisis — so the left tail of every portfolio on
> this chart, including the optimal one, is likely flattered.

This is not boilerplate. On the shipped Base case with the category bands opened
up, the tangency portfolio puts **94% in AC World Equity**, and the stability
check shows that weight ranging **9%–95%** under ±0.5% noise on the expected
returns. The bands are what make the answer usable.

---

## Method

Ported 1:1 from `Keepp_Capital_Target_Allocation_Monte_Carlo.xlsx`:

- Cholesky decomposition of the correlation matrix → correlated normals.
- Monthly log drift `m_i = ln(1 + g_i) / 12`; monthly log vol `s_i = σ_i / √12`.
- Asset monthly simple return `r_i = exp(m_i + s_i·z_i) − 1`.
- Portfolio return `Σ w_i·r_i` each month — rebalanced to target weights monthly.
- Reproduces the workbook's analytic identity `arith ≈ geom + σ²/2`.

Reconciliation against the workbook, Base mode, 10,000 paths. Because hedge
funds sit at 0% in Base, adding the sixth asset class leaves these unchanged:

| Metric | App | Workbook |
|---|---|---|
| Analytic expected return | 7.80% | 7.80% |
| Analytic volatility | 7.40% | 7.40% |
| 10y median annualized | 7.49% | 7.52% |
| 10y median wealth | 2.059× | 2.064× |
| 10y P(loss) | 0.09% | 0.09% |

That check feeds the engine the workbook's *own* inputs, so it still ties out
exactly. The shipped Base case uses the corrected compound returns above and
therefore reads 7.51% / 2.005× — the difference is the error, not drift.

Differences are Monte Carlo sampling noise. The seed is fixed (default
`20260722`), so everyone running the same inputs sees the same numbers.

If a hand-edited correlation matrix is not positive semi-definite, the app
repairs it to the nearest valid matrix and shows a warning rather than
crashing.

---

## Modelling limits worth stating to any reader

- Correlations are static. There is no regime-switching or stress-period
  correlation lift inside the simulation, even though the volatility anchors
  are regime-weighted — so the left tail is likely optimistic.
- Lognormal returns are thin-tailed relative to realised markets. No jumps.
- Nominal USD. No fees, no taxes, no cashflows in or out.
- PE returns follow the J.P. Morgan de-smoothed convention; illiquidity and
  vintage risk are understated by construction.
