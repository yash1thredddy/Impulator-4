"""Base repository with shared query patterns and write locking for SQLite."""
import threading
from typing import TypeVar, Generic, Optional, List, Type, Any

import structlog
from sqlalchemy.orm import Session

logger = structlog.get_logger(__name__)

T = TypeVar("T")

# Module-level lock shared across ALL repository instances.
# SQLite does not support row-level locking; this serializes all writes.
# RLock (reentrant) so callers holding the lock can safely call repo
# write methods without deadlocking.
_db_write_lock = threading.RLock()


class BaseRepository(Generic[T]):
    """Base repository providing CRUD operations with SQLite write locking.

    Write methods acquire _db_write_lock internally -- callers (services)
    do not manage locking. Read methods do NOT lock (SQLite WAL allows
    concurrent reads).

    All methods receive a SQLAlchemy Session parameter and never create
    their own session (ARCH-02).
    """

    def __init__(self, model: Type[T]):
        self._model = model

    def get_by_id(self, db: Session, id_value: Any, *, id_column: str = "id") -> Optional[T]:
        """Get a single record by primary key or named column."""
        col = getattr(self._model, id_column)
        return db.query(self._model).filter(col == id_value).first()

    def get_all(
        self,
        db: Session,
        *,
        offset: int = 0,
        limit: int = 100,
        order_by=None,
        filters: Optional[List] = None,
    ) -> List[T]:
        """Get paginated records with optional filtering and ordering."""
        query = db.query(self._model)
        if filters:
            for f in filters:
                query = query.filter(f)
        if order_by is not None:
            query = query.order_by(order_by)
        return query.offset(offset).limit(limit).all()

    def count(self, db: Session, *, filters: Optional[List] = None) -> int:
        """Count records with optional filtering."""
        query = db.query(self._model)
        if filters:
            for f in filters:
                query = query.filter(f)
        return query.count()

    def add(self, db: Session, entity: T) -> T:
        """Add entity with write lock. Flushes but does NOT commit."""
        with _db_write_lock:
            db.add(entity)
            db.flush()
            return entity

    def delete(self, db: Session, entity: T) -> None:
        """Delete entity with write lock. Flushes but does NOT commit."""
        with _db_write_lock:
            db.delete(entity)
            db.flush()
