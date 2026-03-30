"""
Script 04: Full pipeline — URL list → text extraction → LLM → SQLite

Now with dynamic schema generation:
  1. User query → LLM generates extraction schema (what fields to extract)
  2. Input CSV → fetch document text (with browser headers)
  3. Document text + schema → LLM extracts structured data
  4. Store in SQLite with JSON blob for extracted fields

The extraction prompt is completely generic — no hardcoded assumptions
about quotes, threats, speakers, etc. The schema drives everything.
"""

import json
import re
import sqlite3
import time
from datetime import datetime

import pandas as pd
import requests
import trafilatura


# =============================================================================
# CONFIG
# =============================================================================
OLLAMA_MODEL = "qwen3:30b"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TIMEOUT = 300  # seconds per document

# Browser-like headers to avoid bot blocks
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# =============================================================================
# DATABASE SETUP
# =============================================================================

def init_db(db_path: str) -> sqlite3.Connection:
    """Create the database and tables if they don't exist."""
    conn = sqlite3.connect(db_path)
    
    # Articles table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            title TEXT,
            domain TEXT,
            pub_date TEXT,
            source_date TEXT,
            content_text TEXT,
            text_length INTEGER,
            fetch_time_seconds REAL,
            llm_time_seconds REAL,
            processed_at TEXT
        )
    """)
    
    # Extractions table — uses JSON blob for dynamic fields
    conn.execute("""
        CREATE TABLE IF NOT EXISTS extractions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER,
            extracted_data TEXT,
            FOREIGN KEY (article_id) REFERENCES articles(id)
        )
    """)
    
    # Pipeline metadata — stores the query and schema for this run
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_query TEXT,
            extraction_focus TEXT,
            schema_json TEXT,
            created_at TEXT
        )
    """)
    
    conn.commit()
    return conn


# =============================================================================
# SCHEMA GENERATION
# =============================================================================

SCHEMA_GENERATION_PROMPT = """You are a data extraction expert designing a tabular dataset schema. Given a user's research query, determine what structured fields should be extracted to answer the query.
 
USER QUERY: "{user_query}"
 
Your job: design the fields for a dataset where each ROW is one extracted item (a statement, event, data point, etc.) and each COLUMN is a field.
 
IMPORTANT CONTEXT:
- Ensure that field descriptions are precise and comprehensive.
- Every field should capture a dimension that VARIES across rows. If a field would contain the same value in every row, it is a filter, not a useful column — do not include it.
- Think about what would make this dataset useful for ANALYSIS: what breakdowns, groupings, or comparisons would a researcher want? Add fields for those dimensions.
 
Return ONLY a JSON object:
{{
    "extraction_focus": "one sentence: what constitutes a single row in this dataset",
    "fields": [
        {{
            "name": "field_name_in_snake_case",
            "description": "Clear instruction for what to put in this field. Specify the format and semantic meaning. For missing values, use null."
        }}
    ]
}}
 
Guidelines:
- The FIRST field should be the core content — the main thing being extracted per row.
- For that core field, specify whether it should be semantically: text from the article, a summary, a name, a number that represents a metric, etc. 
- If appropriate, add classification/category fields with EXPLICIT allowed values that enable grouping and analysis.
- Add date fields only if timing is part of the query — specify YYYY-MM-DD format.
- For date fields: use null when the date is not explicitly stated. Do NOT fabricate placeholder dates.
- Use snake_case for field names.
 
Return ONLY the JSON object, no other text.
"""



def generate_schema(user_query: str, model: str = OLLAMA_MODEL) -> dict:
    """Generate extraction schema from user query using LLM, then review it."""
    prompt = SCHEMA_GENERATION_PROMPT.format(user_query=user_query)
    raw, elapsed = _call_llm(prompt)
    print(f"  Schema generation: {elapsed:.1f}s")
    
    parsed = parse_llm_json(raw)
    if not parsed or "fields" not in parsed:
        print("  WARNING: Schema generation failed.")
        return None
    
    # Layer 2: review schema for restrictive language
    reviewed = review_schema(parsed)
    return reviewed


# =============================================================================
# SCHEMA REVIEW (Layer 2 — remove over-restrictive descriptions)
# =============================================================================

SCHEMA_REVIEW_PROMPT = """You are reviewing a data extraction schema for quality. Your job is to ensure that field descriptions do NOT contain language that would cause an extraction system to skip valid content.

ORIGINAL USER QUERY: "{user_query}"

CURRENT SCHEMA:
{schema_json}

Review each field's "description" and fix any of these problems:

1. EXCLUSION LANGUAGE: Remove phrases like "exact text only", "must be explicitly stated", "do not summarize", "do not interpret", "exclude if not...", "only if explicitly...", "verbatim only". These cause the extractor to skip paraphrased, reported, or summarized content that IS relevant.

2. OVERLY NARROW SCOPE: If the core content field restricts to only one presentation style (e.g., "direct quotes only"), broaden it to accept any faithful representation: direct quotes, paraphrases, reported speech, summaries.

3. KEEP EVERYTHING ELSE: Do not change field names, do not add or remove fields, do not change allowed values for category fields, do not change the extraction_focus. Only edit description text.

If no changes are needed, return the schema unchanged.

Return ONLY the corrected JSON schema object (same structure as input), no other text.
"""


def review_schema(schema: dict, model: str = OLLAMA_MODEL) -> dict:
    """Review schema descriptions and strip over-restrictive language.
    
    This is a focused second LLM call — it only edits field descriptions,
    never changes structure, field names, or allowed values.
    """
    prompt = SCHEMA_REVIEW_PROMPT.format(
        user_query="",  # Not strictly needed — schema already reflects the query
        schema_json=json.dumps(schema, indent=2),
    )
    raw, elapsed = _call_llm(prompt)
    print(f"  Schema review: {elapsed:.1f}s")
    
    reviewed = parse_llm_json(raw)
    
    # Validate: must have same structure and field names
    if not reviewed or "fields" not in reviewed:
        print("  Schema review failed — using original schema")
        return schema
    
    original_names = [f["name"] for f in schema["fields"]]
    reviewed_names = [f["name"] for f in reviewed["fields"]]
    
    if original_names != reviewed_names:
        print("  Schema review changed field names — using original schema")
        return schema
    
    # Show what changed
    changes = 0
    for orig, rev in zip(schema["fields"], reviewed["fields"]):
        if orig["description"] != rev["description"]:
            changes += 1
            print(f"  Revised '{orig['name']}': {orig['description'][:60]}...")
            print(f"       →  {rev['description'][:60]}...")
    
    if changes == 0:
        print("  Schema review: no changes needed")
    else:
        print(f"  Schema review: {changes} field(s) revised")
    
    return reviewed


# =============================================================================
# EXTRACTION PROMPT (generic, schema-driven)
# =============================================================================

def build_extraction_prompt(user_query: str, schema: dict, content_text: str) -> str:
    """Build a generic extraction prompt driven by the schema."""
    
    # Format the fields for the prompt
    field_descriptions = []
    field_names = []
    for f in schema["fields"]:
        field_descriptions.append(f'- "{f["name"]}": {f["description"]}')
        field_names.append(f["name"])
    
    fields_block = "\n".join(field_descriptions)
    
    # Build example object with field names
    example_obj = ", ".join(f'"{name}": "..."' for name in field_names)
    
    return f"""You are a structured data extraction system. Given data and a research query, extract all relevant data points into a JSON array.

RESEARCH QUERY: "{user_query}"

EXTRACTION FOCUS: {schema["extraction_focus"]}

FIELDS — for each extracted item, return an object with:
{fields_block}

RULES:
1. Extract every relevant instance.
2. Be faithful to the source. Do not invent values.
3. Each extracted item must be understandable on its own.
4. Be consistent — every item in your output should follow the same format and level of detail for each field.
5. If information for a field is not available, use null.
6. If no relevant items are found, return: []

Return ONLY a JSON array, no other text.

SOURCE DATA:
{content_text}
"""


# =============================================================================
# LLM CALL + PARSING (internal to this module)
# =============================================================================

def _call_llm(prompt: str) -> tuple[str, float]:
    """Call the local LLM and return (raw_response, seconds)."""
    start = time.time()
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        raw = response.json()["message"]["content"]
    except requests.exceptions.Timeout:
        raw = "TIMEOUT"
    except Exception as e:
        raw = f"ERROR: {e}"
    elapsed = time.time() - start
    return raw, elapsed


def parse_llm_json(raw_response: str) -> dict | list | None:
    """Parse JSON from LLM response, handling thinking tags and format variations."""
    clean = raw_response
    if "</think>" in clean:
        clean = clean.split("</think>")[-1].strip()
    
    # Try direct parse
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON array
    match = re.search(r'\[.*\]', clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON object
    match = re.search(r'\{.*\}', clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    return None


def parse_extraction_response(raw_response: str) -> list[dict]:
    """Parse LLM extraction response into a list of dicts."""
    result = parse_llm_json(raw_response)
    
    if result is None:
        return []
    
    # Normalize to list
    if isinstance(result, dict):
        # LLM might wrap in {"results": [...]} or similar
        if any(isinstance(v, list) for v in result.values()):
            for v in result.values():
                if isinstance(v, list):
                    return v
        return [result]
    
    if isinstance(result, list):
        return result
    
    return []


# =============================================================================
# POST-PROCESSING
# =============================================================================

def postprocess_extractions(results: list[dict], schema: dict) -> list[dict]:
    """Clean up LLM extraction output.
    
    1. Drop rows where the core content field (first field) is empty/blank.
    2. Normalize empty strings to null for consistency.
    """
    if not results or not schema.get("fields"):
        return results
    
    core_field = schema["fields"][0]["name"]
    cleaned = []
    
    for r in results:
        # Drop rows with empty core content
        core_value = r.get(core_field, "")
        if not core_value or (isinstance(core_value, str) and not core_value.strip()):
            continue
        
        # Normalize empty strings to None across all fields
        for key, value in r.items():
            if isinstance(value, str) and not value.strip():
                r[key] = None
        
        cleaned.append(r)
    
    dropped = len(results) - len(cleaned)
    if dropped:
        print(f"    (dropped {dropped} empty extraction(s))")
    
    return cleaned


# =============================================================================
# TEXT EXTRACTION (fixed — browser headers)
# =============================================================================

def fetch_content(url: str) -> tuple[str | None, str | None, float]:
    """Fetch and extract document text using browser headers + trafilatura."""
    start = time.time()
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=20, allow_redirects=True)
        if resp.status_code != 200:
            return None, None, time.time() - start
        
        resp.encoding = resp.apparent_encoding or resp.encoding
        downloaded = resp.text
        
        if downloaded:
            text = trafilatura.extract(downloaded, favor_recall=True)
            metadata = trafilatura.extract_metadata(downloaded, default_url=url)
            pub_date = metadata.date if metadata else None
        else:
            text = None
            pub_date = None
    except Exception as e:
        print(f"    fetch error: {e}")
        text = None
        pub_date = None
    
    elapsed = time.time() - start
    return text, pub_date, elapsed


def is_english(text: str, threshold: float = 0.9) -> bool:
    """Quick check — if >90% of characters are ASCII, it's probably English."""
    if not text:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return ascii_count / len(text) > threshold


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_documents(
    user_query: str,
    csv_path: str,
    db_path: str,
    max_documents: int = 50,
    schema: dict | None = None,
):
    """
    Run the full extraction pipeline on a CSV of document URLs.
    
    Args:
        user_query: The original natural language query
        csv_path: Path to CSV with document URLs
        db_path: SQLite database path
        max_documents: Maximum documents to process
        schema: Pre-generated schema (if None, will generate from user_query)
    """
    
    # =========================================================================
    # STEP 0: Generate extraction schema
    # =========================================================================
    print("=" * 70)
    print("STEP 0: Schema Generation")
    print("=" * 70)
    print(f"Query: \"{user_query}\"\n")
    
    if schema is None:
        schema = generate_schema(user_query)
    
    print(f"  Focus: {schema['extraction_focus']}")
    print(f"  Fields ({len(schema['fields'])}):")
    for f in schema["fields"]:
        print(f"    - {f['name']}: {f['description']}")
    print()
    
    # =========================================================================
    # STEP 1: Load document URLs
    # =========================================================================
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} documents from {csv_path}")
    
    before = len(df)
    df = df.drop_duplicates(subset="title", keep="first")
    print(f"After dedup by title: {len(df)} documents ({before - len(df)} duplicates removed)")
    
    # Don't pre-slice — iterate through all URLs, stop when we have enough successful fetches
    print(f"Will process up to {max_documents} successfully fetched documents\n")
    
    # =========================================================================
    # STEP 2: Init database and store metadata
    # =========================================================================
    conn = init_db(db_path)
    
    conn.execute("""
        INSERT INTO pipeline_metadata (user_query, extraction_focus, schema_json, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        user_query,
        schema["extraction_focus"],
        json.dumps(schema),
        datetime.now().isoformat(),
    ))
    conn.commit()
    
    # =========================================================================
    # STEP 3: Process documents
    # =========================================================================
    timing = {
        "fetch_times": [],
        "llm_times": [],
        "total_extractions": 0,
        "articles_processed": 0,
        "articles_skipped": 0,
        "articles_no_results": 0,
        "fetch_failures": 0,
    }
    
    pipeline_start = time.time()
    
    for i, row in df.iterrows():
        # Stop when we've successfully processed enough documents
        if timing["articles_processed"] >= max_documents:
            break
        
        url = row["url"]
        title = str(row.get("title", ""))
        domain = str(row.get("domain", ""))
        source_date = str(row.get("seendate", ""))
        
        print(f"[{timing['articles_processed'] + 1}/{max_documents}] {title[:80]}...")
        
        # Check if already processed
        existing = conn.execute(
            "SELECT id FROM articles WHERE url = ?", (url,)
        ).fetchone()
        if existing:
            print(f"    Already in database, skipping.")
            timing["articles_skipped"] += 1
            continue
        
        # --- Fetch document text ---
        print(f"    Fetching text...", end=" ")
        content_text, pub_date, fetch_time = fetch_content(url)
        timing["fetch_times"].append(fetch_time)
        
        if not content_text:
            print(f"FAILED ({fetch_time:.1f}s)")
            timing["fetch_failures"] += 1
            conn.execute("""
                INSERT INTO articles (url, title, domain, pub_date, source_date, content_text, text_length, fetch_time_seconds, llm_time_seconds, processed_at)
                VALUES (?, ?, ?, ?, ?, NULL, 0, ?, 0, ?)
            """, (url, title, domain, pub_date, source_date, fetch_time, datetime.now().isoformat()))
            conn.commit()
            continue  # Don't count against max — try next URL
        
        print(f"OK ({len(content_text)} chars, {fetch_time:.1f}s)")
        
        if not is_english(content_text):
            print(f"    Skipped (non-English content)")
            timing["fetch_failures"] += 1
            conn.execute("""
                INSERT INTO articles (url, title, domain, pub_date, source_date, content_text, text_length, fetch_time_seconds, llm_time_seconds, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """, (url, title, domain, pub_date, source_date, content_text, len(content_text), fetch_time, datetime.now().isoformat()))
            conn.commit()
            continue
        
        # --- LLM extraction ---
        print(f"    Running LLM extraction...", end=" ")
        prompt = build_extraction_prompt(user_query, schema, content_text)
        raw_response, llm_time = _call_llm(prompt)
        timing["llm_times"].append(llm_time)
        print(f"done ({llm_time:.1f}s)")
        
        # --- Parse + post-process ---
        results = parse_extraction_response(raw_response)
        results = postprocess_extractions(results, schema)
        
        # --- Store in database ---
        cursor = conn.execute("""
            INSERT INTO articles (url, title, domain, pub_date, source_date, content_text, text_length, fetch_time_seconds, llm_time_seconds, processed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (url, title, domain, pub_date, source_date, content_text, len(content_text), fetch_time, llm_time, datetime.now().isoformat()))
        article_id = cursor.lastrowid
        
        for r in results:
            conn.execute("""
                INSERT INTO extractions (article_id, extracted_data)
                VALUES (?, ?)
            """, (article_id, json.dumps(r)))
        
        conn.commit()
        
        timing["articles_processed"] += 1
        timing["total_extractions"] += len(results)
        
        if len(results) == 0:
            timing["articles_no_results"] += 1
            print(f"    No relevant items found.")
        else:
            print(f"    Extracted {len(results)} item(s):")
            for r in results:
                # Show first two fields as preview
                preview_fields = list(r.items())[:2]
                preview = ", ".join(f"{k}: {str(v)[:60]}" for k, v in preview_fields)
                print(f"      → {preview}")
        
        print()
    
    pipeline_end = time.time()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = pipeline_end - pipeline_start
    
    print("=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Query:                   \"{user_query}\"")
    print(f"Schema fields:           {', '.join(f['name'] for f in schema['fields'])}")
    print(f"Total time:              {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Documents processed:     {timing['articles_processed']}")
    print(f"Fetch failures (skipped):{timing['fetch_failures']}")
    print(f"Documents skipped (dups):{timing['articles_skipped']}")
    print(f"Documents with no results:{timing['articles_no_results']}")
    print(f"Total extractions:       {timing['total_extractions']}")
    print()
    
    if timing["fetch_times"]:
        print(f"Fetch times:  avg {sum(timing['fetch_times'])/len(timing['fetch_times']):.1f}s, "
              f"max {max(timing['fetch_times']):.1f}s, "
              f"total {sum(timing['fetch_times']):.1f}s")
    
    if timing["llm_times"]:
        print(f"LLM times:    avg {sum(timing['llm_times'])/len(timing['llm_times']):.1f}s, "
              f"max {max(timing['llm_times']):.1f}s, "
              f"total {sum(timing['llm_times']):.1f}s")
    
    if timing["articles_processed"] > 0:
        avg_per_doc = total_time / timing["articles_processed"]
        print(f"\nAvg per document:        {avg_per_doc:.1f}s")
        print(f"Projected for 250 documents: {avg_per_doc * 250 / 60:.0f} minutes")
    
    # Show what's in the database
    print(f"\n{'=' * 70}")
    print("DATABASE CONTENTS")
    print("=" * 70)
    
    articles_count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    extractions_count = conn.execute("SELECT COUNT(*) FROM extractions").fetchone()[0]
    
    print(f"Articles in DB:     {articles_count}")
    print(f"Extractions in DB:  {extractions_count}")
    
    # Show all extractions
    print(f"\nAll extracted items:")
    rows = conn.execute("""
        SELECT e.extracted_data, a.domain, a.url
        FROM extractions e
        JOIN articles a ON e.article_id = a.id
        ORDER BY a.source_date
    """).fetchall()
    
    for row in rows:
        data_json, domain, url = row
        data = json.loads(data_json)
        # Show first three fields as preview
        preview_fields = list(data.items())[:3]
        preview = " | ".join(f"{k}: {str(v)[:50]}" for k, v in preview_fields)
        print(f"\n  {preview}")
        print(f"      Source: {domain}")
    
    conn.close()
    print(f"\nDatabase saved to: {db_path}")


# =============================================================================
# CONVENIENCE: Export extractions to flat DataFrame
# =============================================================================

def export_to_dataframe(db_path: str) -> pd.DataFrame:
    """Load extractions from DB and flatten JSON into columns."""
    conn = sqlite3.connect(db_path)
    
    rows = conn.execute("""
        SELECT e.id, e.extracted_data,
               a.url, a.title, a.domain, a.pub_date, a.source_date
        FROM extractions e
        JOIN articles a ON e.article_id = a.id
        ORDER BY a.source_date
    """).fetchall()
    
    records = []
    for row in rows:
        eid, data_json, url, title, domain, pub_date, source_date = row
        data = json.loads(data_json)
        data["_extraction_id"] = eid
        data["_source_url"] = url
        data["_source_title"] = title
        data["_source_domain"] = domain
        data["_pub_date"] = pub_date
        data["_source_date"] = source_date
        records.append(data)
    
    conn.close()
    return pd.DataFrame(records)


# =============================================================================
if __name__ == "__main__":
    # Config — update these for your run
    USER_QUERY = "Find all threats that Trump has made to Iran in 2025 and 2026"
    INPUT_CSV = "input_urls.csv"
    DB_PATH = "pipeline_dynamic.db"
    OUTPUT_CSV = "pipeline_output.csv"
    MAX_DOCUMENTS = 10

    # Run pipeline
    process_documents(
        user_query=USER_QUERY,
        csv_path=INPUT_CSV,
        db_path=DB_PATH,
        max_documents=MAX_DOCUMENTS,
    )

    # Export to CSV
    print(f"\n{'=' * 70}")
    print("EXPORTING TO CSV")
    print(f"{'=' * 70}")
    
    df = export_to_dataframe(DB_PATH)
    
    if len(df) == 0:
        print("No extractions to export.")
    else:
        # Reorder columns: dynamic fields first, then metadata
        meta_cols = [c for c in df.columns if c.startswith("_")]
        data_cols = [c for c in df.columns if not c.startswith("_")]
        df = df[data_cols + meta_cols]
        
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"Saved: {OUTPUT_CSV} ({len(df)} rows, {len(df.columns)} columns)")
        print(f"Columns: {', '.join(df.columns)}")
        print(f"\nPreview:")
        # Show first few rows, truncating wide columns
        pd.set_option("display.max_colwidth", 60)
        pd.set_option("display.width", 200)
        print(df[data_cols].to_string(index=False))
