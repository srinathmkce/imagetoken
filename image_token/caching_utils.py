import sqlite3
import hashlib
from typing import Optional
from pathlib import Path

DB_PATH = Path(__file__).parent / "ImageTokenDimensionCache.sqlite"

def get_url_hash(url: str) -> str:
    """Generate a SHA256 hash from the URL (acts as unique key)."""
    return hashlib.sha256(url.encode()).hexdigest()

def init_cache():
    """Creating the cache table if it doesn't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dimension_cache (
                url_hash TEXT PRIMARY KEY,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL
            )
        """)

def get_cached_dimensions(url: str) -> Optional[tuple[int, int]]:
    """Return (width, height) from cache if available, else None."""
    init_cache()
    url_hash = get_url_hash(url)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "SELECT width, height FROM dimension_cache WHERE url_hash = ?",
            (url_hash,)
        )
        row = cursor.fetchone()
        if row:
            return row 
        return None

def cache_dimensions(url: str, width: int, height: int):
    """Store the (width, height) for a URL in the cache."""
    init_cache()
    url_hash = get_url_hash(url)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO dimension_cache (url_hash, width, height) VALUES (?, ?, ?)",
            (url_hash, width, height)
        )
