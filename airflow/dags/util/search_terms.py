import os
import sys
from databricks import sql

SEARCH_TERMS_TABLE = "rag_pipeline.silver.search_terms"

# Where backfilling starts for a brand-new term, and how far each run
# advances the watermark until it catches up to "today". Once caught up,
# subsequent runs naturally behave like the steady-state incremental mode.
BACKFILL_START_DATE = "2020-01-01"
BACKFILL_INCREMENT_DAYS = 30

DEFAULT_TERMS = [
    "Viral Vectors",
    "Protein Aggregation",
    "mRNA Stability",
    "Purification",
    "Filtration",
    "Media" "Analytical Techniques",
    "Federal Regulations",
    "Titer",
]


def get_connection():
    # Establishes connection using standard environment variables
    return sql.connect(
        server_hostname=os.environ["DATABRICKS_HOST"],
        http_path=os.environ["DATABRICKS_HTTP_PATH"],
        access_token=os.environ["DATABRICKS_TOKEN"],
    )
 
 
def create_search_terms_table() -> bool:
    """Creates and seeds the search_terms table if it is missing or empty.
 
    Returns:
        bool: True if the table was newly created/seeded (or was empty),
              False if it already contained rows.
    """
    conn = get_connection()
    cursor = conn.cursor()
 
    try:
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {SEARCH_TERMS_TABLE} (
                term                  STRING,
                active                BOOLEAN,
                last_run_at           TIMESTAMP,
                run_count             INT,
                last_searched_through DATE
            )
        """)
 
        cursor.execute(f"SELECT COUNT(*) FROM {SEARCH_TERMS_TABLE}")
        row_count = cursor.fetchone()[0]
 
        if row_count == 0:
            for term in DEFAULT_TERMS:
                cursor.execute(
                    f"""
                    INSERT INTO {SEARCH_TERMS_TABLE} VALUES (
                        %(term)s, true, NULL, 0, NULL
                    )
                """,
                    {"term": term},
                )
            print(f"Table '{SEARCH_TERMS_TABLE}' created and seeded with default terms.")
            return True
        else:
            print(f"Table '{SEARCH_TERMS_TABLE}' already contains {row_count} row(s). Skipping seed.")
            return False
 
    except Exception as e:
        print(f"An error occurred during table setup: {e}")
        raise e
 
    finally:
        cursor.close()
        conn.close()
 
 
def add_term(term: str, active: bool = True):
    """
    Adds a new search term to the rotation. Since last_run_at starts
    NULL, it will be picked up on the very next DAG run (nulls sort
    first in get_next_term's ORDER BY).
    """
    conn = get_connection()
    cursor = conn.cursor()
 
    cursor.execute(
        f"SELECT COUNT(*) FROM {SEARCH_TERMS_TABLE} WHERE term = %(term)s",
        {"term": term},
    )
    if cursor.fetchone()[0] > 0:
        cursor.close()
        conn.close()
        print(f"Term '{term}' already exists in {SEARCH_TERMS_TABLE}. Skipping insert.")
        return
 
    cursor.execute(
        f"""
        INSERT INTO {SEARCH_TERMS_TABLE} VALUES (
            %(term)s, %(active)s, NULL, 0, NULL
        )
    """,
        {"term": term, "active": active},
    )
    cursor.close()
    conn.close()
 
    print(f"Added term '{term}' (active={active}) to {SEARCH_TERMS_TABLE}.")
 
 
def set_term_active(term: str, active: bool):
    """Pauses or resumes a term without deleting its history."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        f"""
        UPDATE {SEARCH_TERMS_TABLE}
        SET active = %(active)s
        WHERE term = %(term)s
    """,
        {"active": active, "term": term},
    )
    cursor.close()
    conn.close()
 
    print(f"Set term '{term}' active={active}.")
 
 
def compute_next_window(last_searched_through, increment_days: int = None) -> dict:
    """
    Given a term's current last_searched_through (a date, or None if the
    term has never run), computes the (mindate, maxdate) window to search
    next.
 
    - First run (None): starts at BACKFILL_START_DATE.
    - While backfilling: advances by `increment_days` per run (falls back
      to BACKFILL_INCREMENT_DAYS if not given), capped at today.
    - Once the watermark reaches today: mindate == maxdate == today,
      which naturally behaves like "only fetch what's new since
      yesterday" going forward -- no special-casing needed once caught up.
    """
    from datetime import date, timedelta
 
    if increment_days is None:
        increment_days = BACKFILL_INCREMENT_DAYS
 
    today = date.today()
 
    if last_searched_through is None:
        mindate = date.fromisoformat(BACKFILL_START_DATE)
    else:
        mindate = last_searched_through
 
    maxdate = min(mindate + timedelta(days=increment_days), today)
 
    return {
        "mindate": mindate.strftime("%Y/%m/%d"),
        "maxdate": maxdate.strftime("%Y/%m/%d"),
        "maxdate_iso": maxdate.isoformat(),
        "caught_up": maxdate >= today,
    }
 
 
def get_next_term() -> dict:
    """
    Returns the active search term that was run least recently
    (NULL last_run_at, i.e. never run, is prioritized first), along
    with the date it was last searched through (NULL if never searched).
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT term, last_searched_through
        FROM {SEARCH_TERMS_TABLE}
        WHERE active = true
        ORDER BY last_run_at ASC NULLS FIRST
        LIMIT 1
    """)
    row = cursor.fetchone()
    cursor.close()
    conn.close()
 
    if row is None:
        raise ValueError(
            f"No active search terms found in {SEARCH_TERMS_TABLE}. "
            "Add rows or set at least one term to active = true."
        )
 
    return {"term": row[0], "last_searched_through": row[1]}
 
 
def mark_run(term: str, searched_through: str):
    """
    Updates last_run_at, increments run_count, and advances the
    last_searched_through watermark for the given term.
 
    searched_through should be an ISO date string (e.g. "2026-07-12"),
    typically "today" at the time the ingestion run completed.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        f"""
        UPDATE {SEARCH_TERMS_TABLE}
        SET last_run_at = current_timestamp(),
            run_count = run_count + 1,
            last_searched_through = %(searched_through)s
        WHERE term = %(term)s
    """,
        {"term": term, "searched_through": searched_through},
    )
    cursor.close()
    conn.close()
 
    print(f"Marked search term '{term}' as run through {searched_through}.")
 
 
if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser(description="Manage the PubMed search term rotation")
    subparsers = parser.add_subparsers(dest="command")
 
    subparsers.add_parser("seed", help="Create/seed the table with default terms (default action)")
 
    add_parser = subparsers.add_parser("add", help="Add a new search term to the rotation")
    add_parser.add_argument("term", help="Search term to add, e.g. 'Lipid Nanoparticles'")
    add_parser.add_argument("--inactive", action="store_true", help="Add as inactive (paused)")
 
    pause_parser = subparsers.add_parser("pause", help="Deactivate a search term")
    pause_parser.add_argument("term")
 
    resume_parser = subparsers.add_parser("resume", help="Reactivate a search term")
    resume_parser.add_argument("term")
 
    args = parser.parse_args()
 
    try:
        if args.command == "add":
            add_term(args.term, active=not args.inactive)
        elif args.command == "pause":
            set_term_active(args.term, active=False)
        elif args.command == "resume":
            set_term_active(args.term, active=True)
        else:
            was_seeded = create_search_terms_table()
            if was_seeded:
                print("Table didn't exist or was empty, successfully seeded")
                sys.exit(0)
            else:
                print("Table already existed with data")
                sys.exit(3)
    except Exception as err:
        print(f"Fatal error: {err}")
        sys.exit(1)