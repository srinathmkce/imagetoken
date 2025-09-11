import sqlite3
from typing import Optional
from pathlib import Path
import os

CACHE_DIR = Path(os.getenv("IMAGE_CACHE_DIR", Path.home() / ".image_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = CACHE_DIR / "ImageTokenDimensionCache.sqlite"


class ImageDimensionCache:
    def __init__(self):
        self._connection = None

    def __enter__(self):
        """Open connection when entering context."""
        self._connection = sqlite3.connect(DB_PATH)
        self._init_cache()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close connection when exiting context."""
        if self._connection:
            self._connection.close()
            self._connection = None

    def _init_cache(self):
        """Create the cache table if it doesn't exist."""
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS dimension_cache (
                url TEXT PRIMARY KEY,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL
            )
        """)
        self._connection.commit()

    def get_cached_dimensions(self, url: str) -> Optional[tuple[int, int]]:
        """Return (width, height) from cache if available, else None."""
        if not self._connection:
            raise RuntimeError("Cache not initialized.")

        cursor = self._connection.execute(
            "SELECT width, height FROM dimension_cache WHERE url = ?", (url,)
        )
        return cursor.fetchone()

    def cache_dimensions(self, url: str, width: int, height: int):
        """Store the (width, height) for a URL in the cache."""
        if not self._connection:
            raise RuntimeError("Cache not initialized.")

        self._connection.execute(
            "INSERT OR REPLACE INTO dimension_cache (url, width, height) VALUES (?, ?, ?)",
            (url, width, height),
        )
        self._connection.commit()

    def delete_dimensions(self, url: str):
        """Delete the cache for a URL"""
        if not self._connection:
            raise RuntimeError("Cache not initialized.")

        self._connection.execute("DELETE FROM dimension_cache WHERE url = ?", (url,))
        self._connection.commit()
