"""
Refresh the historical proxy returns used by the Backtest & stress tab.

    python scripts/update_proxy_returns.py

Downloads every proxy (keepp_mc/backtest_data.py, the data.py logic) and
rewrites keepp_mc/data/proxy_returns.csv. Commit the CSV to update the app.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from keepp_mc import backtest_data as bd  # noqa: E402

R = bd.build_returns()
R.to_csv(bd.SNAPSHOT, float_format="%.8f")
print(bd.coverage(R).to_string())
print(f"\nWrote {len(R)} months ({R.index[0]:%b %Y} – {R.index[-1]:%b %Y}) "
      f"to {bd.SNAPSHOT}")
