#!/usr/bin/env python3
"""
Thin wrapper around Alembic commands for discoverability.

Usage:
    python scripts/migrate.py upgrade [head]        # Apply pending migrations (default: head)
    python scripts/migrate.py downgrade -1           # Rollback one revision
    python scripts/migrate.py current                # Show current revision
    python scripts/migrate.py history                # Show revision history
    python scripts/migrate.py revision "message"     # Create new empty revision
    python scripts/migrate.py revision "message" --autogenerate  # Auto-detect model changes
    python scripts/migrate.py heads                  # Show head revisions
    python scripts/migrate.py check                  # Check if upgrade needed (exit 1 if pending)

Requires DATABASE_URL (or DIRECT_DATABASE_URL) in .env or environment.
"""
import sys
from pathlib import Path

# Ensure project root is importable (for backend.config)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alembic.config import Config as AlembicConfig
from alembic import command

ALEMBIC_INI = str(Path(__file__).resolve().parent.parent / "backend" / "alembic.ini")

USAGE = """\
Usage: python scripts/migrate.py <command> [args]

Commands:
    upgrade [revision]              Apply migrations (default: head)
    downgrade [revision]            Rollback migrations (default: -1)
    current                         Show current revision
    history                         Show revision history
    revision "message" [--autogenerate]  Create new revision
    heads                           Show head revisions
    check                           Check if upgrade needed (exit 1 if pending)
"""


def main() -> None:
    args = sys.argv[1:]

    if not args:
        print(USAGE)
        sys.exit(1)

    cfg = AlembicConfig(ALEMBIC_INI)
    cmd = args[0]

    if cmd == "upgrade":
        revision = args[1] if len(args) > 1 else "head"
        command.upgrade(cfg, revision)
    elif cmd == "downgrade":
        revision = args[1] if len(args) > 1 else "-1"
        command.downgrade(cfg, revision)
    elif cmd == "current":
        command.current(cfg, verbose=True)
    elif cmd == "history":
        command.history(cfg, verbose=True)
    elif cmd == "revision":
        if len(args) < 2:
            print("Error: revision requires a message argument")
            print("  python scripts/migrate.py revision \"describe the change\"")
            sys.exit(1)
        message = args[1]
        autogenerate = "--autogenerate" in args
        command.revision(cfg, message=message, autogenerate=autogenerate)
    elif cmd == "heads":
        command.heads(cfg, verbose=True)
    elif cmd == "check":
        command.check(cfg)
    else:
        print(f"Unknown command: {cmd}")
        print(USAGE)
        sys.exit(1)


if __name__ == "__main__":
    main()
