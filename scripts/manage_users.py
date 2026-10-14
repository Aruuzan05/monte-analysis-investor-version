"""
Account admin for the Keepp simulator — run from a terminal, not the app.

    python scripts/manage_users.py add <username>        # prompts for a password
    python scripts/manage_users.py reset <username>      # new password, signs them out
    python scripts/manage_users.py list

Uses the same database the app does: KEEPP_DATABASE_URL if set, otherwise
[database] url from .streamlit/secrets.toml, otherwise the local SQLite file
data/keepp.db. To manage the Community Cloud database from your laptop,
put its URL in your local secrets.toml or export KEEPP_DATABASE_URL.
"""
import getpass
import os
import sys
import tomllib
from pathlib import Path

import sqlalchemy as sa

# Run as a script from anywhere: paths and imports are relative to the repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from keepp_mc import db as dbm  # noqa: E402


def database_url() -> str:
    if os.environ.get("KEEPP_DATABASE_URL"):
        return os.environ["KEEPP_DATABASE_URL"]
    sec = ROOT / ".streamlit" / "secrets.toml"
    if sec.exists():
        url = tomllib.loads(sec.read_text()).get("database", {}).get("url")
        if url:
            return url
    data_dir = Path(os.environ.get("KEEPP_DATA_DIR", ROOT / "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(data_dir / 'keepp.db').as_posix()}"


def _new_password() -> str:
    p1 = getpass.getpass("Password: ")
    p2 = getpass.getpass("Repeat:   ")
    if p1 != p2:
        sys.exit("Passwords do not match.")
    return p1


def main(argv):
    if len(argv) < 2 or argv[1] not in ("add", "reset", "list"):
        sys.exit(__doc__)
    store = dbm.RunDB(database_url())
    cmd = argv[1]
    try:
        if cmd == "list":
            with store.engine.connect() as c:
                rows = c.execute(sa.text(
                    "SELECT u.username, COUNT(r.id) FROM users u "
                    "LEFT JOIN runs r ON r.user_id = u.id AND r.deleted_at IS NULL "
                    "GROUP BY u.username ORDER BY u.username")).all()
            for name, n in rows:
                print(f"{name:<32} {n} saved run{'s' if n != 1 else ''}")
            if not rows:
                print("(no accounts yet)")
        elif len(argv) != 3:
            sys.exit(f"Usage: python scripts/manage_users.py {cmd} <username>")
        elif cmd == "add":
            store.create_user(argv[2], _new_password())
            print(f"Created {argv[2].strip().lower()!r}.")
        else:
            store.set_password(argv[2], _new_password())
            print(f"Password reset for {argv[2].strip().lower()!r}; "
                  f"they have been signed out everywhere.")
    except dbm.AuthError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main(sys.argv)
