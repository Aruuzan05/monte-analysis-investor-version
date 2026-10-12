"""Checks for the database + accounts layer (db.py) and run_store.run_digest.

Run after any change to db.py or run_store.py:

    python test_db.py

Uses an in-memory SQLite database, so it needs no server and leaves nothing
behind. The same code runs against Postgres in production; set
KEEPP_TEST_DATABASE_URL to point this file at a scratch Postgres database to
check that too (every table it touches is dropped first).
"""
import os
import tempfile
import time

import pandas as pd

import db as dbm
import run_store as rs
from engine import simulate

ok = True


def check(label, cond):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")


URL = os.environ.get("KEEPP_TEST_DATABASE_URL", "sqlite://")
if URL != "sqlite://":
    import sqlalchemy as sa
    _e = sa.create_engine(dbm.normalize_url(URL))
    dbm.META.drop_all(_e)
    _e.dispose()
DB = dbm.RunDB(URL)

ALLOC = pd.DataFrame([
    {"Category": "Fixed Income", "LTCMA line": "U.S. Cash",
     "Weight %": 40.0, "Return %": 3.80, "Vol %": 0.67},
    {"Category": "Equities", "LTCMA line": "AC World Equity",
     "Weight %": 60.0, "Return %": 12.88, "Vol %": 14.47},
])
CORR = [[1.0, 0.0], [0.0, 1.0]]
res, _ = simulate(list(ALLOC["Weight %"]), list(ALLOC["Return %"]),
                  list(ALLOC["Vol %"]), CORR, 1000, 5, 7)


def record(label="", alloc=ALLOC, mode="Base"):
    return rs.make_record(0, label, mode, alloc, 1000, 5, 7, res)


def digest(alloc=ALLOC, mode="Base", seed=7):
    return rs.run_digest(mode, alloc.to_dict("records"), 1000, 5, seed)


# ==========================================================================
print("=" * 78)
print("1. PASSWORDS AND ACCOUNTS")
print("=" * 78)
h = dbm.hash_password("correct horse")
check("hash is not the password", "correct horse" not in h)
check("hash verifies the right password", dbm.check_password("correct horse", h))
check("hash rejects a wrong password", not dbm.check_password("correct hors", h))
check("same password hashes differently each time (salted)",
      h != dbm.hash_password("correct horse"))
check("malformed stored hash fails closed", not dbm.check_password("x", "junk"))

alice = DB.create_user("Alice", "password123")
check("username is stored lower-case", DB.username(alice) == "alice")
for bad_user, bad_pw, why in [("alice", "password123", "duplicate username"),
                              ("ALICE", "password123", "duplicate, other case"),
                              ("al", "password123", "username too short"),
                              ("bob smith", "password123", "space in username"),
                              ("bob", "short", "password too short")]:
    try:
        DB.create_user(bad_user, bad_pw)
        check(f"rejects {why}", False)
    except dbm.AuthError:
        check(f"rejects {why}", True)
bob = DB.create_user("bob", "hunter2hunter2")
check("login with right password returns the user id",
      DB.verify_user("ALICE ", "password123") == alice)
check("login with wrong password returns None",
      DB.verify_user("alice", "password124") is None)
check("login for unknown user returns None",
      DB.verify_user("nobody", "password123") is None)
check("user_count counts accounts", DB.user_count() == 2)

# ==========================================================================
print()
print("=" * 78)
print("2. SESSIONS — what keeps a user signed in across a refresh")
print("=" * 78)
tok = DB.create_session(alice)
check("token is long and random", len(tok) >= 40)
check("token resolves to its user", DB.user_for_token(tok) == (alice, "alice"))
check("unknown token resolves to nobody", DB.user_for_token("nope") is None)
check("empty token resolves to nobody", DB.user_for_token("") is None)
with DB.engine.connect() as c:
    stored = [r[0] for r in c.execute(dbm.SESSIONS.select().with_only_columns(
        dbm.SESSIONS.c.token_hash))]
check("raw token is never stored (only its hash)", tok not in stored)
old = DB.create_session(alice, ttl_days=-1)
check("expired token is refused", DB.user_for_token(old) is None)
DB.delete_session(tok)
check("signed-out token is refused", DB.user_for_token(tok) is None)
t2 = DB.create_session(bob)
DB.set_password("bob", "new-password-1")
check("password reset signs the user out everywhere",
      DB.user_for_token(t2) is None)
check("new password works after reset",
      DB.verify_user("bob", "new-password-1") == bob)

# ==========================================================================
print()
print("=" * 78)
print("3. DIGEST — when two runs count as the same scenario")
print("=" * 78)
d0 = digest()
check("digest is stable", d0 == digest())
check("tiny float noise does not change the digest",
      d0 == rs.run_digest("Base", [dict(r, **{"Weight %": r["Weight %"] + 1e-9})
                                   for r in ALLOC.to_dict("records")],
                          1000, 5, 7))
check("seed changes the digest", d0 != digest(seed=8))
check("mode changes the digest", d0 != digest(mode="Custom"))
check("line order changes the digest (Cholesky is order-dependent)",
      d0 != digest(alloc=ALLOC.iloc[::-1].reset_index(drop=True)))
a2 = ALLOC.copy()
a2.loc[0, "Return %"] = 4.0
check("an edited return changes the digest", d0 != digest(alloc=a2))

# ==========================================================================
print()
print("=" * 78)
print("4. AUTO-SAVE — every distinct run saved once, per user")
print("=" * 78)
r1 = DB.save_run(alice, record(), d0, pool=["U.S. Cash", "AC World Equity"])
check("first run is saved", r1 is not None)
check("it gets sequence number 1 and the standard auto label",
      r1.record.seq == 1 and r1.record.label == "Run 1 · Base · 5y")
check("pool is stored", r1.pool == ("U.S. Cash", "AC World Equity"))
check("re-running identical inputs does not save a duplicate",
      DB.save_run(alice, record(), d0) is None)
r2 = DB.save_run(alice, record(alloc=a2), digest(alloc=a2))
check("a changed run is saved as #2", r2 is not None and r2.record.seq == 2)
rb = DB.save_run(bob, record(), d0)
check("another user can save the same scenario (numbering is per user)",
      rb is not None and rb.record.seq == 1)
check("each user only sees their own runs",
      [s.id for s in DB.list_runs(alice)] == [r1.id, r2.id]
      and [s.id for s in DB.list_runs(bob)] == [rb.id])
check("a user cannot read another user's run by id",
      DB.get_run(bob, r1.id) is None)

back = DB.get_run(alice, r1.id).record
check("round-trip: sleeves come back identical", back.sleeves == r1.record.sleeves)
check("round-trip: horizon rows come back identical",
      back.horizon_rows == r1.record.horizon_rows)
check("round-trip: headline numbers identical",
      (back.median_wealth_x, back.p_loss_pct, back.seed)
      == (record().median_wealth_x, record().p_loss_pct, 7))
check("stored records feed the existing Excel export",
      len(rs.to_excel_bytes([s.record for s in DB.list_runs(alice)])) > 1000)

# ==========================================================================
print()
print("=" * 78)
print("5. RENAME, DELETE, UNDO")
print("=" * 78)
check("rename works", DB.rename_run(alice, r1.id, "  Base case  ")
      and DB.get_run(alice, r1.id).record.label == "Base case")
check("blank rename is refused", not DB.rename_run(alice, r1.id, "   "))
check("cannot rename another user's run", not DB.rename_run(bob, r1.id, "x"))
check("cannot delete another user's run",
      DB.delete_runs(bob, [r1.id]) == [] and DB.get_run(alice, r1.id))
gone = DB.delete_runs(alice, [r1.id])
check("delete hides the run", gone == [r1.id] and
      [s.id for s in DB.list_runs(alice)] == [r2.id])
r3 = DB.save_run(alice, record(), d0)
check("re-running a deleted scenario saves it again as a new run",
      r3 is not None and r3.record.seq == 3)
check("undo does not resurrect a deleted run that was since re-run",
      DB.undelete_runs(alice, gone) == 0)
DB.delete_runs(alice, [r3.id])
check("undo brings a deleted run back",
      DB.undelete_runs(alice, [r3.id]) == 1
      and r3.id in [s.id for s in DB.list_runs(alice)])
DB.delete_runs(alice, [r2.id])
DB.delete_runs(alice, [r3.id])
check("only the latest delete can be undone (older batch purged)",
      DB.undelete_runs(alice, [r2.id]) == 0
      and DB.undelete_runs(alice, [r3.id]) == 1)
check("sequence numbers are never reused after a delete",
      DB.save_run(alice, record(alloc=a2), digest(alloc=a2)).record.seq == 4)

# ==========================================================================
print()
print("=" * 78)
print("6. PERSISTENCE — a file database survives the process (= a refresh)")
print("=" * 78)
with tempfile.TemporaryDirectory() as tmp:
    url = f"sqlite:///{tmp}/k.db"
    a = dbm.RunDB(url)
    uid = a.create_user("carol", "password123")
    tk = a.create_session(uid)
    a.save_run(uid, record(), d0)
    a.engine.dispose()
    b = dbm.RunDB(url)                       # fresh engine = app restart
    check("session still valid after restart", b.user_for_token(tk) == (uid, "carol"))
    check("runs still there after restart", len(b.list_runs(uid)) == 1)
    b.engine.dispose()

check("postgres:// URLs are routed to psycopg 3",
      dbm.normalize_url("postgres://u:p@h/db") == "postgresql+psycopg://u:p@h/db")

print()
print("RESULT:", "all checks pass" if ok else "FAILURES ABOVE — investigate")
raise SystemExit(0 if ok else 1)
