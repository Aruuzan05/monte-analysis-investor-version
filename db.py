"""
Persistent storage — user accounts, login sessions and saved runs.

One SQLAlchemy Core schema that runs unchanged on SQLite (local / Docker,
a single file) and on Postgres (Streamlit Community Cloud, where the
container's disk is wiped on every reboot, so the database has to live
elsewhere). Which one is used is decided by the URL handed to RunDB.

No Streamlit import, same as engine.py / optimizer.py / run_store.py — the
whole module is exercised headlessly by test_db.py against in-memory SQLite.

Security notes
--------------
* Passwords are hashed with scrypt (stdlib hashlib), per-user random salt.
  Nothing reversible is stored.
* Login sessions are random 256-bit tokens. Only their SHA-256 is stored, so
  a leaked database does not hand out live sessions.
* Every run query is scoped by user_id. There is no code path that reads,
  renames or deletes a run without the owner's id in the WHERE clause.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

import run_store as rs

SESSION_TTL_DAYS = 30
MIN_PASSWORD_LEN = 8
USERNAME_RE = re.compile(r"^[a-z0-9._@-]{3,64}$")

# scrypt cost. n=2**14, r=8 is ~16 MB and ~50 ms per hash: slow enough to make
# offline guessing expensive, fast enough that a login does not feel slow.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2 ** 14, 8, 1


class AuthError(ValueError):
    """A user-facing account problem (bad username, taken, weak password)."""


# --------------------------------------------------------------------------
# Password hashing
# --------------------------------------------------------------------------
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=_SCRYPT_N,
                        r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${dk.hex()}"


def check_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt_hex, dk_hex = stored.split("$")
        if algo != "scrypt":
            return False
        dk = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex),
                            n=int(n), r=int(r), p=int(p),
                            dklen=len(dk_hex) // 2)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(dk.hex(), dk_hex)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
META = sa.MetaData()

USERS = sa.Table(
    "users", META,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("username", sa.String(64), nullable=False, unique=True),
    sa.Column("pw_hash", sa.String(255), nullable=False),
    sa.Column("created_at", sa.Float, nullable=False),
)

SESSIONS = sa.Table(
    "sessions", META,
    sa.Column("token_hash", sa.String(64), primary_key=True),
    sa.Column("user_id", sa.Integer,
              sa.ForeignKey("users.id", ondelete="CASCADE"),
              nullable=False, index=True),
    sa.Column("created_at", sa.Float, nullable=False),
    sa.Column("expires_at", sa.Float, nullable=False),
)

RUNS = sa.Table(
    "runs", META,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("user_id", sa.Integer,
              sa.ForeignKey("users.id", ondelete="CASCADE"),
              nullable=False, index=True),
    sa.Column("seq", sa.Integer, nullable=False),
    sa.Column("label", sa.String(200), nullable=False),
    sa.Column("created_at", sa.String(32), nullable=False),
    sa.Column("mode", sa.String(16), nullable=False),
    sa.Column("digest", sa.String(64), nullable=False, index=True),
    sa.Column("paths", sa.Integer, nullable=False),
    sa.Column("years", sa.Integer, nullable=False),
    sa.Column("seed", sa.BigInteger, nullable=False),
    sa.Column("exp_return_pct", sa.Float, nullable=False),
    sa.Column("analytic_vol_pct", sa.Float, nullable=False),
    sa.Column("sim_vol_pct", sa.Float, nullable=False),
    sa.Column("median_wealth_x", sa.Float, nullable=False),
    sa.Column("p_loss_pct", sa.Float, nullable=False),
    sa.Column("elapsed_ms", sa.Float, nullable=False),
    # JSON text: {"sleeves": [...], "horizon_rows": [...], "pool": [...]}.
    # Text rather than a JSON column type so SQLite and Postgres behave
    # identically.
    sa.Column("payload", sa.Text, nullable=False),
    # Soft delete: set on Delete, cleared on Undo. The previous batch is
    # purged for good the next time something is deleted.
    sa.Column("deleted_at", sa.Float, nullable=True),
    sa.UniqueConstraint("user_id", "seq", name="uq_runs_user_seq"),
)


@dataclass(frozen=True)
class StoredRun:
    """A run as it lives in the database: the RunRecord the rest of the app
    already understands, plus what is needed to delete or restore it."""
    id: int
    digest: str
    pool: tuple                  # Standard mode: LTCMA lines offered to the optimizer
    record: rs.RunRecord


def normalize_url(url: str) -> str:
    """Accept the URLs hosting providers hand out (postgres://, postgresql://)
    and point them at the psycopg 3 driver that requirements.txt installs."""
    url = url.strip()
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


class RunDB:
    def __init__(self, url: str):
        url = normalize_url(url)
        kw: dict = {"future": True}
        if url.startswith("sqlite"):
            # Streamlit serves each browser session on its own thread.
            kw["connect_args"] = {"check_same_thread": False}
            if url in ("sqlite://", "sqlite:///:memory:"):
                kw["poolclass"] = sa.pool.StaticPool   # one shared in-memory DB
        else:
            # Hosted Postgres (Neon, Supabase) closes idle connections; check
            # before use instead of failing the first query after a lull.
            kw["pool_pre_ping"] = True
            kw["pool_recycle"] = 300
        self.engine = sa.create_engine(url, **kw)
        if url.startswith("sqlite"):
            @sa.event.listens_for(self.engine, "connect")
            def _fk_on(dbapi_conn, _):
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()
        META.create_all(self.engine)

    @property
    def backend(self) -> str:
        return self.engine.dialect.name          # "sqlite" / "postgresql"

    # ---- users -----------------------------------------------------------
    def user_count(self) -> int:
        with self.engine.connect() as c:
            return int(c.execute(sa.select(sa.func.count()).select_from(USERS))
                       .scalar_one())

    def create_user(self, username: str, password: str) -> int:
        username = (username or "").strip().lower()
        if not USERNAME_RE.match(username):
            raise AuthError("Username must be 3–64 characters: letters, "
                            "numbers, dot, dash, underscore or @.")
        if len(password or "") < MIN_PASSWORD_LEN:
            raise AuthError(f"Password must be at least {MIN_PASSWORD_LEN} "
                            f"characters.")
        try:
            with self.engine.begin() as c:
                return int(c.execute(USERS.insert().values(
                    username=username, pw_hash=hash_password(password),
                    created_at=time.time())).inserted_primary_key[0])
        except IntegrityError:
            raise AuthError("That username is already taken.") from None

    def set_password(self, username: str, password: str) -> None:
        if len(password or "") < MIN_PASSWORD_LEN:
            raise AuthError(f"Password must be at least {MIN_PASSWORD_LEN} "
                            f"characters.")
        username = (username or "").strip().lower()
        with self.engine.begin() as c:
            uid = c.execute(sa.select(USERS.c.id)
                            .where(USERS.c.username == username)).scalar()
            if uid is None:
                raise AuthError(f"No user called {username!r}.")
            c.execute(USERS.update().where(USERS.c.id == uid)
                      .values(pw_hash=hash_password(password)))
            # A password reset signs the user out everywhere.
            c.execute(SESSIONS.delete().where(SESSIONS.c.user_id == uid))

    def verify_user(self, username: str, password: str) -> Optional[int]:
        username = (username or "").strip().lower()
        with self.engine.connect() as c:
            row = c.execute(sa.select(USERS.c.id, USERS.c.pw_hash)
                            .where(USERS.c.username == username)).first()
        if row is None:
            # Burn the same time as a real check so response time does not
            # reveal which usernames exist.
            check_password(password or "", _DUMMY_HASH)
            return None
        return int(row.id) if check_password(password or "", row.pw_hash) else None

    def username(self, user_id: int) -> Optional[str]:
        with self.engine.connect() as c:
            return c.execute(sa.select(USERS.c.username)
                             .where(USERS.c.id == user_id)).scalar()

    # ---- sessions --------------------------------------------------------
    def create_session(self, user_id: int,
                       ttl_days: float = SESSION_TTL_DAYS) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self.engine.begin() as c:
            # Opportunistic clean-up of expired sessions.
            c.execute(SESSIONS.delete().where(SESSIONS.c.expires_at < now))
            c.execute(SESSIONS.insert().values(
                token_hash=_token_hash(token), user_id=user_id,
                created_at=now, expires_at=now + ttl_days * 86400))
        return token

    def user_for_token(self, token: Optional[str]) -> Optional[tuple[int, str]]:
        if not token:
            return None
        with self.engine.connect() as c:
            row = c.execute(
                sa.select(USERS.c.id, USERS.c.username, SESSIONS.c.expires_at)
                .join(SESSIONS, SESSIONS.c.user_id == USERS.c.id)
                .where(SESSIONS.c.token_hash == _token_hash(token))).first()
        if row is None or row.expires_at < time.time():
            return None
        return int(row.id), row.username

    def delete_session(self, token: Optional[str]) -> None:
        if not token:
            return
        with self.engine.begin() as c:
            c.execute(SESSIONS.delete()
                      .where(SESSIONS.c.token_hash == _token_hash(token)))

    # ---- runs ------------------------------------------------------------
    def _live(self, user_id: int):
        return sa.and_(RUNS.c.user_id == user_id, RUNS.c.deleted_at.is_(None))

    def find_by_digest(self, user_id: int, digest: str) -> Optional[int]:
        with self.engine.connect() as c:
            return c.execute(sa.select(RUNS.c.id).where(
                self._live(user_id), RUNS.c.digest == digest)
                .order_by(RUNS.c.seq.desc()).limit(1)).scalar()

    def save_run(self, user_id: int, record: rs.RunRecord, digest: str,
                 pool: Sequence[str] = ()) -> Optional[StoredRun]:
        """Insert a run unless one with identical inputs is already saved.

        Returns the new StoredRun, or None for a duplicate. `record.seq` and
        `record.label` are ignored when they are placeholders: the database
        assigns the next per-user sequence number, and an empty label becomes
        the standard auto label.
        """
        if self.find_by_digest(user_id, digest) is not None:
            return None
        payload = json.dumps({"sleeves": record.sleeves,
                              "horizon_rows": record.horizon_rows,
                              "pool": list(pool)}, default=float)
        for _ in range(3):                     # retry on a concurrent seq clash
            try:
                with self.engine.begin() as c:
                    seq = int(c.execute(
                        sa.select(sa.func.coalesce(sa.func.max(RUNS.c.seq), 0))
                        .where(RUNS.c.user_id == user_id)).scalar_one()) + 1
                    label = (record.label or "").strip()
                    if not label or label == f"Run {record.seq}":
                        label = rs.auto_label(seq, record.mode, record.years)
                    new_id = c.execute(RUNS.insert().values(
                        user_id=user_id, seq=seq, label=label[:200],
                        created_at=record.created_at, mode=record.mode,
                        digest=digest, paths=record.paths, years=record.years,
                        seed=record.seed, exp_return_pct=record.exp_return_pct,
                        analytic_vol_pct=record.analytic_vol_pct,
                        sim_vol_pct=record.sim_vol_pct,
                        median_wealth_x=record.median_wealth_x,
                        p_loss_pct=record.p_loss_pct,
                        elapsed_ms=record.elapsed_ms, payload=payload,
                    )).inserted_primary_key[0]
                return self.get_run(user_id, int(new_id))
            except IntegrityError:
                continue
        raise RuntimeError("Could not allocate a run number; try again.")

    @staticmethod
    def _to_stored(row) -> StoredRun:
        p = json.loads(row.payload)
        rec = rs.RunRecord(
            seq=row.seq, label=row.label, created_at=row.created_at,
            mode=row.mode, sleeves=p["sleeves"], paths=row.paths,
            years=row.years, seed=int(row.seed),
            exp_return_pct=row.exp_return_pct,
            analytic_vol_pct=row.analytic_vol_pct,
            sim_vol_pct=row.sim_vol_pct, median_wealth_x=row.median_wealth_x,
            p_loss_pct=row.p_loss_pct, elapsed_ms=row.elapsed_ms,
            horizon_rows=p["horizon_rows"])
        return StoredRun(id=int(row.id), digest=row.digest,
                         pool=tuple(p.get("pool", [])), record=rec)

    def list_runs(self, user_id: int) -> list[StoredRun]:
        with self.engine.connect() as c:
            rows = c.execute(sa.select(RUNS).where(self._live(user_id))
                             .order_by(RUNS.c.seq)).all()
        return [self._to_stored(r) for r in rows]

    def get_run(self, user_id: int, run_id: int) -> Optional[StoredRun]:
        with self.engine.connect() as c:
            row = c.execute(sa.select(RUNS).where(
                self._live(user_id), RUNS.c.id == run_id)).first()
        return self._to_stored(row) if row else None

    def rename_run(self, user_id: int, run_id: int, label: str) -> bool:
        label = (label or "").strip()[:200]
        if not label:
            return False
        with self.engine.begin() as c:
            n = c.execute(RUNS.update().where(
                self._live(user_id), RUNS.c.id == run_id)
                .values(label=label)).rowcount
        return n == 1

    def delete_runs(self, user_id: int, run_ids: Sequence[int]) -> list[int]:
        """Soft-delete `run_ids`. Purges the previous (already undone-able)
        batch first, so exactly one delete can be undone at a time."""
        ids = [int(i) for i in run_ids]
        if not ids:
            return []
        with self.engine.begin() as c:
            c.execute(RUNS.delete().where(RUNS.c.user_id == user_id,
                                          RUNS.c.deleted_at.is_not(None)))
            c.execute(RUNS.update().where(self._live(user_id),
                                          RUNS.c.id.in_(ids))
                      .values(deleted_at=time.time()))
            return [int(i) for i in c.execute(
                sa.select(RUNS.c.id).where(RUNS.c.user_id == user_id,
                                           RUNS.c.deleted_at.is_not(None)))
                .scalars()]

    def undelete_runs(self, user_id: int, run_ids: Sequence[int]) -> int:
        ids = [int(i) for i in run_ids]
        if not ids:
            return 0
        with self.engine.begin() as c:
            # A run whose inputs were re-run (and so re-saved) after the delete
            # stays deleted — restoring it would duplicate the live copy.
            live = set(c.execute(sa.select(RUNS.c.digest)
                                 .where(self._live(user_id))).scalars())
            n = 0
            for rid, dig in c.execute(sa.select(RUNS.c.id, RUNS.c.digest).where(
                    RUNS.c.user_id == user_id, RUNS.c.id.in_(ids),
                    RUNS.c.deleted_at.is_not(None))).all():
                if dig in live:
                    continue
                c.execute(RUNS.update().where(RUNS.c.id == rid)
                          .values(deleted_at=None))
                live.add(dig)
                n += 1
            return n


_DUMMY_HASH = hash_password(secrets.token_hex(8))
