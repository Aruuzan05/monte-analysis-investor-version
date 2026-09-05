# Keepp Capital — Monte Carlo Allocation Simulator

A Streamlit app for the target-allocation Monte Carlo. It runs in two modes:

- **Base · JPM 2026 LTCMA** — every input locked to the figures behind the
  July 2027 deck. This is the read-only reference case.
- **Custom** — weights, expected returns, volatilities and the full correlation
  matrix are all editable. **Each of the six asset types gets its own table
  and its own dropdown**, scoped to that type's LTCMA lines, and can hold any
  number of sleeves.

Either way the app returns the 3y/5y/10y percentile table and a wealth fan
chart. Switching between modes keeps both sets of inputs: Base always shows the
JPM figures, and Custom remembers your edits.

Six asset classes: Fixed Income, Public Equities, Private Equity & VC,
Diversified Hedge Funds, Commodities & Real Assets, Cash.

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
| `engine.py` | The simulation plus `expand_correlation()`. Pure numpy, no Streamlit import, unit-testable. |
| `jpm_ltcma.py` | The 58 LTCMA lines, the six per-type option lists, and the label/number sync rules. No Streamlit import. Update here for the 2027 LTCMA — and add any new line to a group, or the test will fail. |
| `test_engine.py` | Reconciles Base against the workbook; also exercises a funded hedge fund sleeve and the non-PSD repair path. |
| `requirements.txt` | Pinned. Verified to resolve and reproduce the workbook. |
| `Dockerfile` | Mirrors are build args, so it works inside and outside China. |
| `docker-compose.yml` | One-command deploy. |
| `run_local.sh` | Virtualenv + run, no Docker. |
| `.streamlit/config.toml` | Theme, telemetry off, XSRF on. Committed. |
| `.gitignore` | Keeps the virtualenv, caches and `secrets.toml` out of git. |

Run the reconciliation any time you change the engine:

```bash
python test_engine.py
```

---

## Base assumptions

| Asset class | LTCMA line | Weight | Return | Vol |
|---|---|---|---|---|
| Fixed Income | *manual* — global IG blend | 45% | 5.00% | 7.02% |
| Public Equities & Mutual Funds | *manual* — MSCI World 10y | 20% | 12.88% | 14.47% |
| Private Equity & Venture Capital | Private Equity | 10% | 10.20% | 19.78% |
| Diversified Hedge Funds | Diversified Hedge Funds | 0% | 5.30% | 5.75% |
| Commodities & Real Assets | Commodities | 8% | 4.60% | 18.32% |
| Cash & Equivalents | *manual* — 3M T-bill | 17% | 3.80% | 0.67% |

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

### Multiple sleeves per asset type

The allocation table is a flat list of sleeves. Every row carries the general
asset type it belongs to plus its own LTCMA line, weight, return and volatility,
so Public Equities can be 12% U.S. Large Cap + 8% Emerging Markets Equity while
still reading as one 20% equity allocation. The **+** button on a type's table
adds a sleeve to that type; selecting a row and pressing delete removes it.
Each type's heading carries its own subtotal and sleeve count.

The general six types stay exactly as they were — they are now the
**correlation grouping** rather than the unit of allocation.

#### The correlation assumption this forces

Two sleeves inside one asset type need a correlation between them, and
J.P. Morgan publishes no line-level matrix. So the sleeve-level matrix is built
at runtime from two editable inputs:

* sleeves in **different** types take the published between-type figure;
* sleeves in the **same** type take that type's *within-type correlation*.

Defaults, all editable under **Correlations** in the sidebar:

| Asset type | Within-type |
|---|---|
| Fixed Income | 0.85 |
| Public Equities | 0.85 |
| Private Equity | 0.75 |
| Hedge Funds | 0.80 |
| Commodities & Real Assets | 0.45 |
| Cash | 0.90 |

Real assets get a much lower figure on purpose: commodities, real estate and
infrastructure genuinely do not move together, and a single high number there
would understate the diversification.

This is a real simplification and worth stating to a reader. Within fixed
income especially, long Treasuries and high yield are nothing like 0.85
correlated — if a split matters to a decision, set that number deliberately
rather than accepting the default.

`engine.expand_correlation()` does the expansion and is unit-tested: one sleeve
per type reproduces the 6×6 matrix exactly, and splitting a sleeve in two with
within-correlation 1.0 reproduces the unsplit portfolio's return and volatility
to machine precision.

### The six groups

The 58 LTCMA lines are divided across the six general asset types. Each type is
rendered as its own table, so its dropdown only ever offers its own lines — the
Hedge Funds table lists the five hedge fund strategies and nothing else.

| Asset type | Lines offered |
|---|---|
| Fixed Income | all 21 fixed income lines |
| Public Equities | all 18 equity lines |
| Private Equity | Private Equity, Venture Capital, Direct Lending |
| Hedge Funds | Diversified, Event Driven, Long Bias, Relative Value, Macro |
| Commodities & Real Assets | Commodities, Gold, the four real estate lines, REITs, Infrastructure, Transport, Timberland, Commercial Mortgage Loans |
| Cash | U.S. Cash, Short Duration Gov/Credit, Intermediate Treasuries, Muni 1–15yr |

Every one of the 58 lines is reachable from at least one group — asserted in
`test_engine.py`, so adding a line to the catalogue without grouping it fails
the test rather than going silently missing. The four short-dated fixed income
lines appear in both Fixed Income and Cash on purpose: a cash sleeve can
legitimately hold short-duration government credit.

**Why one table per type rather than one table with a filtered dropdown.**
Streamlit's `SelectboxColumn` carries a single option list for the whole
column, not one per row, so a single shared table physically cannot scope the
list to each row's asset type. Splitting into six editors makes the scoping
structural instead of something enforced after the fact — which also removed
the cross-family guard, its warning messages and the family prefixes on the
labels that the previous build needed.

### Sub-category dropdowns

In Custom mode each bucket can be set to any LTCMA line from its own family —
21 fixed income, 18 equity, and the 19 alternatives split into private markets,
hedge funds and real assets. Type in the cell to filter. Picking a line fills
Return and Vol from the compound column.

Three rules keep the label honest about the numbers:

1. Picking a line overwrites Return and Vol from the catalogue.
2. Editing Return or Vol by hand flips the label to *Manual entry*, so a sleeve
   never claims J.P. Morgan provenance for figures that are no longer theirs.
3. A row added by the editor arrives blank; it defaults to Manual entry with a
   0% weight, so it cannot silently poison the weight total.
4. A stored line that is not in its group's list — only reachable from stale
   session state after the catalogue changes — falls back to Manual entry
   rather than sticking.

Swapping a line does not change its correlation row — that follows the asset
type, not the line. Adjust the type matrix or the within-type figure by hand if
a swap materially changes a sleeve's character.

**Worth knowing before a demo:** every forward-looking JPM equity line sits
5–6pp below the 12.88% realised MSCI World figure, with higher volatility.
Switching Public Equities to `AC World Equity` (7.00% / 16.78%) moves 10y
median wealth from 2.005× to 1.808×. It is the single biggest lever in the
model.

The 6x6 correlation matrix is verified positive semi-definite (minimum
eigenvalue 0.067), so it factorises without repair.

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
