import os
from pathlib import Path
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from models import Base

DATA_DIR = Path(os.environ.get("MEMORA_DATA_DIR", Path.home() / ".memora"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "memora.db"
ENGINE = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False, "timeout": 30},
)

# Enable WAL mode for better concurrent read/write performance
@event.listens_for(ENGINE, "connect")
def set_wal_mode(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=ENGINE)


def init_db() -> None:
    Base.metadata.create_all(bind=ENGINE)
    _migrate_schema()


def _migrate_schema() -> None:
    """Add columns introduced after initial schema. SQLite ALTER TABLE ADD COLUMN
    is safe to call even if the column already exists (we catch the error)."""
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    migrations = [
        "ALTER TABLE faces ADD COLUMN det_score REAL",
        "ALTER TABLE media_files ADD COLUMN tags TEXT",
    ]
    for stmt in migrations:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    conn.close()


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
