import json
import pickle
import sqlite3
import os
from src.preprocessing import preprocess

CORPUS_PATH = "data/processed/corpus.json"
PICKLE_PATH = "data/processed/inverted_index.pkl"
DB_PATH = "data/processed/metadata.db"


def load_corpus() -> list[dict]:
    with open(CORPUS_PATH) as f:
        return json.load(f)


def build_inverted_index(corpus: list[dict]) -> dict:
    """
    Builds a simple inverted index:
    token -> list of document IDs that contain the token
    """
    index = {}
    for doc_id, doc in enumerate(corpus):
        for token in set(doc["tokens"]):  # set = no duplicate doc_ids per token
            if token not in index:
                index[token] = []
            index[token].append(doc_id)
    return index


def save_index_pickle(index: dict):
    os.makedirs("data/processed", exist_ok=True)
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(index, f)
    print(f"Inverted index saved to {PICKLE_PATH}")


def load_index_pickle() -> dict:
    with open(PICKLE_PATH, "rb") as f:
        return pickle.load(f)


def build_sqlite_metadata(corpus: list[dict]):
    """
    Stores song metadata in SQLite for fast filtering
    e.g. filter by year, artist, genre
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS songs (
            doc_id INTEGER PRIMARY KEY,
            title TEXT,
            artist TEXT,
            genre TEXT,
            year TEXT
        )
    """)

    c.executemany(
        "INSERT OR REPLACE INTO songs VALUES (?, ?, ?, ?, ?)",
        [
            (
                doc_id,
                doc.get("title", "Unknown"),
                doc.get("artist", "Unknown"),
                doc.get("genre", ""),
                doc.get("year", ""),
            )
            for doc_id, doc in enumerate(corpus)
        ]
    )

    conn.commit()
    conn.close()
    print(f"Metadata saved to {DB_PATH}")


def query_metadata(filters: dict = {}) -> list[int]:
    """
    Query SQLite metadata with optional filters.
    Returns list of matching doc_ids.
    Example: query_metadata({"genre": "rock", "year": "1990"})
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    query = "SELECT doc_id FROM songs WHERE 1=1"
    params = []

    if "genre" in filters:
        query += " AND genre = ?"
        params.append(filters["genre"])
    if "year" in filters:
        query += " AND year = ?"
        params.append(filters["year"])
    if "artist" in filters:
        query += " AND artist LIKE ?"
        params.append(f"%{filters['artist']}%")

    c.execute(query, params)
    results = [row[0] for row in c.fetchall()]
    conn.close()
    return results


if __name__ == "__main__":
    print("Loading corpus...")
    corpus = load_corpus()
    print(f"Loaded {len(corpus)} documents")

    print("Building inverted index...")
    index = build_inverted_index(corpus)
    print(f"Index has {len(index)} unique tokens")

    print("Saving index to pickle...")
    save_index_pickle(index)

    print("Building SQLite metadata database...")
    build_sqlite_metadata(corpus)

    print("All done!")