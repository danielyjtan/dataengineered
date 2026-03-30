"""
Orchestrator — single entry point from user query to dataset.

Flow:
  1. Decompose query → sub-tasks matched to sources
  2. For each actionable sub-task:
     a. Load source spec from registry
     b. Query the API (using api_query)
     c. DETECT response type → route to correct pipeline:
        - Article links → fetch HTML → LLM extraction → CSV
        - Structured data → parse directly → CSV
     d. Export result
  3. Report: what succeeded, what couldn't be fulfilled

The pipeline type is determined at RESPONSE TIME by inspecting what the
API actually returns, not by metadata or configuration.

Usage:
    python orchestrator.py "Find all threats Trump made to Iran in 2025 and 2026"

Dependencies:
    - source_registry.py   (registered sources + specs)
    - decompose_query.py   (query planning)
    - api_query.py         (API URL construction + execution)
    - queries_to_sqlite.py (article fetch + LLM extraction + CSV export)
"""

import json
import os
import sys
import time
from datetime import datetime

import pandas as pd

from source_registry import get_spec, get_available_sources
from decompose_query import decompose_query, print_plan
from api_query import run_query
from queries_to_sqlite import (
    generate_schema,
    process_documents,
    export_to_dataframe,
    init_db,
)


# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR = "pipeline_output"
DEFAULT_MAX_ITEMS = 50


# =============================================================================
# RESPONSE TYPE DETECTION (the router)
# =============================================================================

def detect_response_type(data) -> str:
    """Inspect API response data and determine the pipeline type.
    
    Logic: check if records contain URL fields. If yes → article_links.
    If no → structured_data. That's it.
    
    Returns:
        "article_links"   — response contains URLs to fetch and extract from
        "structured_data" — response contains usable data directly
        "unknown"         — can't determine (empty/null response)
    """
    if data is None:
        return "unknown"

    records = _extract_records(data)
    if not records:
        return "unknown"

    # Check a sample of records for URL-like content
    sample = records[:10]
    records_with_urls = 0

    for record in sample:
        if not isinstance(record, dict):
            continue
        has_url = False
        for key, value in record.items():
            # Check by field name
            if key.lower() in ("url", "link", "href", "source_url", "article_url"):
                has_url = True
                break
            # Check by value pattern
            if isinstance(value, str) and (
                value.startswith("http://") or value.startswith("https://")
            ):
                has_url = True
                break
        if has_url:
            records_with_urls += 1

    # If majority of sampled records contain URLs, it's article links
    if records_with_urls > len(sample) / 2:
        return "article_links"

    return "structured_data"


def _extract_records(data) -> list[dict]:
    """Extract a flat list of record dicts from various API response formats."""
    if isinstance(data, list):
        # Could be [metadata, [records]] (World Bank) or [record, record, ...]
        if len(data) >= 2 and isinstance(data[1], list):
            return data[1]  # World Bank format
        return [r for r in data if isinstance(r, dict) and len(r) > 0]

    if isinstance(data, dict):
        # Look for a list value (e.g., {"articles": [...]})
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict):
                    return value
        # Don't return the dict itself unless it has real content
        if len(data) > 0:
            return [data]

    return []


# =============================================================================
# PIPELINE: ARTICLE EXTRACTION
# =============================================================================

def run_article_pipeline(
    data,
    description: str,
    task_index: int,
    output_dir: str,
    max_items: int,
) -> dict:
    """Pipeline for API responses that contain article URLs.
    
    Steps: extract URLs → save as CSV → fetch HTML → schema generation →
           LLM extraction → export CSV
    """
    # --- Extract article URLs into a CSV ---
    records = _extract_records(data)
    if not records:
        return _fail("No records found in API response")

    df = pd.DataFrame(records)

    # Ensure standard columns exist
    for col in ["url", "title", "seendate", "domain"]:
        if col not in df.columns:
            df[col] = ""

    articles_csv = os.path.join(output_dir, f"task_{task_index}_articles.csv")
    df.to_csv(articles_csv, index=False)
    print(f"    Saved {len(df)} article URLs to CSV")

    # --- Generate extraction schema ---
    print(f"\n  Generating extraction schema...")
    schema = generate_schema(description)

    if not schema:
        return _fail("No schema generated.")


    print(f"  Focus: {schema['extraction_focus']}")
    print(f"  Fields: {', '.join(f['name'] for f in schema['fields'])}")

    # --- Fetch articles + LLM extraction ---
    db_path = os.path.join(output_dir, f"task_{task_index}.db")
    process_documents(
        user_query=description,
        csv_path=articles_csv,
        db_path=db_path,
        max_documents=max_items,
        schema=schema,
    )

    # --- Export ---
    output_csv = os.path.join(output_dir, f"task_{task_index}_output.csv")
    df_out = export_to_dataframe(db_path)

    if len(df_out) == 0:
        return {
            "success": True,
            "csv_path": None,
            "rows": 0,
            "pipeline_type": "article_extraction",
            "message": "Pipeline ran but no items were extracted from the articles",
        }

    meta_cols = [c for c in df_out.columns if c.startswith("_")]
    data_cols = [c for c in df_out.columns if not c.startswith("_")]
    df_out = df_out[data_cols + meta_cols]
    df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return {
        "success": True,
        "csv_path": output_csv,
        "rows": len(df_out),
        "columns": data_cols,
        "pipeline_type": "article_extraction",
        "message": f"Extracted {len(df_out)} items from {len(df)} articles",
    }


# =============================================================================
# PIPELINE: STRUCTURED DATA
# =============================================================================

def run_structured_pipeline(
    data,
    description: str,
    task_index: int,
    output_dir: str,
) -> dict:
    """Pipeline for API responses that contain structured data directly.
    
    Steps: parse records → flatten to DataFrame → export CSV
    No HTML fetching, no LLM extraction needed.
    Returns all records — no item limit (no LLM cost per record).
    """
    records = _extract_records(data)
    if not records:
        return _fail("No records found in API response")

    # Flatten to DataFrame
    df = pd.json_normalize(records)

    if len(df) == 0:
        return _fail("Records could not be parsed into a table")

    # Clean up column names (replace dots from nested JSON)
    df.columns = [c.replace(".", "_") for c in df.columns]

    output_csv = os.path.join(output_dir, f"task_{task_index}_output.csv")
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    data_cols = list(df.columns)
    print(f"    Parsed {len(df)} records with {len(data_cols)} columns")

    return {
        "success": True,
        "csv_path": output_csv,
        "rows": len(df),
        "columns": data_cols,
        "pipeline_type": "structured_data",
        "message": f"Parsed {len(df)} records directly from API response",
    }


# =============================================================================
# EXECUTE A SINGLE SUB-TASK
# =============================================================================

def execute_subtask(
    task: dict,
    user_query: str,
    task_index: int,
    output_dir: str,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> dict:
    """Execute a single sub-task from the decomposition plan.

    This is source-agnostic: it queries the API, detects what kind of
    data came back, and routes to the appropriate pipeline.
    """
    source_id = task["source_id"]
    description = task["description"]
    source_name = task.get("source_name", source_id)

    print(f"\n{'=' * 70}")
    print(f"EXECUTING TASK {task_index}: {description}")
    print(f"Source: {source_name}")
    print(f"{'=' * 70}")

    # --- Load API spec ---
    spec = get_spec(source_id)
    if not spec:
        return _fail(f"No API spec found for source '{source_id}'")

    # --- Query the API ---
    print(f"\n  Querying {source_name} API...")
    query_result = run_query(description, spec)

    if not query_result:
        return _fail(f"API query failed for: {description}")

    data = query_result["data"]
    print(f"  → {query_result.get('num_results', 0)} results from API")

    # --- Detect response type and route ---
    response_type = detect_response_type(data)
    print(f"  → Response type detected: {response_type}")

    if response_type == "article_links":
        return run_article_pipeline(
            data=data,
            description=description,
            task_index=task_index,
            output_dir=output_dir,
            max_items=max_items,
        )
    elif response_type == "structured_data":
        return run_structured_pipeline(
            data=data,
            description=description,
            task_index=task_index,
            output_dir=output_dir,
        )
    else:
        return _fail(f"Could not determine how to process API response (type: {response_type})")


def _fail(message: str) -> dict:
    """Shorthand for a failed result."""
    return {"success": False, "csv_path": None, "rows": 0, "message": message}


# =============================================================================
# MERGE STRUCTURED OUTPUTS FROM SAME SOURCE
# =============================================================================

def merge_same_source_outputs(results: list, output_files: list, output_dir: str) -> tuple:
    """Merge structured_data CSVs that came from the same source into one file.
    
    Article extraction outputs are left untouched (different schemas).
    
    Returns updated (output_files, results) with merged entries replacing originals.
    """
    # Group structured_data results by source_id
    structured_groups = {}  # source_id → list of (index, result)
    for i, r in enumerate(results):
        outcome = r["outcome"]
        source_id = r["task"].get("source_id")
        if (outcome.get("success") and 
            outcome.get("pipeline_type") == "structured_data" and
            source_id):
            if source_id not in structured_groups:
                structured_groups[source_id] = []
            structured_groups[source_id].append((i, r))

    # Only merge groups with 2+ results
    groups_to_merge = {k: v for k, v in structured_groups.items() if len(v) > 1}

    if not groups_to_merge:
        return output_files, results

    for source_id, group in groups_to_merge.items():
        print(f"\n  Merging {len(group)} structured outputs from {source_id}...")

        # Read and concat all CSVs
        dfs = []
        csv_paths_to_remove = []
        task_descriptions = []

        for idx, r in group:
            csv_path = r["outcome"]["csv_path"]
            if csv_path and os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                dfs.append(df)
                csv_paths_to_remove.append(csv_path)
                task_descriptions.append(r["task"]["description"])

        if not dfs:
            continue

        merged_df = pd.concat(dfs, ignore_index=True)

        # Try to pivot to wide format (e.g., GDP/population/trade as columns per year)
        # Requires: an indicator column + a date column + a value column
        indicator_col = None
        for col in ["indicator_value", "indicator_id"]:
            if col in merged_df.columns:
                indicator_col = col
                break

        if indicator_col and "date" in merged_df.columns and "value" in merged_df.columns:
            try:
                # Drop metadata columns before pivot
                drop_cols = ["unit", "obs_status", "decimal", "message",
                             "countryiso3code", "country_id", "country_value"]
                pivot_df = merged_df.drop(
                    columns=[c for c in drop_cols if c in merged_df.columns]
                )

                wide_df = pivot_df.pivot_table(
                    index="date", columns=indicator_col, values="value", aggfunc="first"
                ).reset_index()
                wide_df.columns.name = None  # Remove multi-index name

                # Clean column names
                wide_df.columns = [
                    str(c).replace(" ", "_").replace(",", "").replace("(", "").replace(")", "").lower()
                    for c in wide_df.columns
                ]

                merged_df = wide_df
                print(f"    Pivoted to wide format: {list(wide_df.columns)}")
            except Exception as e:
                print(f"    Could not pivot to wide format: {e}")
                print(f"    Using stacked format instead")

        # Write merged CSV
        source_name = group[0][1]["task"].get("source_name", source_id)
        merged_filename = f"{source_id}_merged.csv"
        merged_path = os.path.join(output_dir, merged_filename)
        merged_df.to_csv(merged_path, index=False, encoding="utf-8-sig")

        print(f"    Combined {len(dfs)} files → {len(merged_df)} rows")
        print(f"    Saved to {merged_filename}")

        # Remove individual CSVs
        for csv_path in csv_paths_to_remove:
            if os.path.exists(csv_path):
                os.remove(csv_path)

        # Update output_files: remove old, add merged
        output_files = [f for f in output_files if f not in csv_paths_to_remove]
        output_files.append(merged_path)

        # Update results: replace first entry with merged info, mark others as merged
        first_idx = group[0][0]
        results[first_idx]["outcome"] = {
            "success": True,
            "csv_path": merged_path,
            "rows": len(merged_df),
            "columns": list(merged_df.columns),
            "pipeline_type": "structured_data",
            "message": f"Merged {len(dfs)} queries into {len(merged_df)} rows",
            "merged_from": task_descriptions,
        }
        results[first_idx]["task"]["description"] = (
            f"{source_name} data (merged {len(dfs)} indicators)"
        )

        # Mark subsequent entries as merged
        for idx, r in group[1:]:
            results[idx]["outcome"] = {
                "success": True,
                "csv_path": None,
                "rows": 0,
                "pipeline_type": "structured_data",
                "message": f"Merged into {merged_filename}",
                "merged_into": merged_path,
            }

    # Filter out merged-away results from display
    results = [r for r in results if r["outcome"].get("csv_path") is not None 
               or r["outcome"].get("merged_into") is None]

    return output_files, results

def run_pipeline(user_query: str, max_items: int = DEFAULT_MAX_ITEMS) -> dict:
    """
    Single entry point: user query → dataset(s).
    """
    pipeline_start = time.time()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("PIPELINE ORCHESTRATOR")
    print("=" * 70)
    print(f"Query:   \"{user_query}\"")
    print(f"Output:  {output_dir}")
    print()

    # =========================================================================
    # STEP 1: Decompose query
    # =========================================================================
    print("STEP 1: Decomposing query...")
    plan = decompose_query(user_query)
    print_plan(plan)
    print()

    # =========================================================================
    # STEP 2: Check feasibility
    # =========================================================================
    tasks = plan.get("sub_tasks", [])
    actionable = [t for t in tasks if t.get("source_available")]
    unfulfilled = [t for t in tasks if not t.get("source_available")]

    if not actionable:
        print("No sub-tasks can be executed. Stopping.")
        if plan.get("user_message"):
            print(f"  → {plan['user_message']}")
        return {
            "query": user_query,
            "plan": plan,
            "results": [],
            "output_files": [],
            "elapsed": time.time() - pipeline_start,
        }

    if unfulfilled:
        print(f"⚠ {len(unfulfilled)} sub-task(s) cannot be fulfilled:")
        for t in unfulfilled:
            print(f"    ✗ {t['description']} — {t.get('reason', 'no source')}")
        print(f"  Proceeding with {len(actionable)} actionable sub-task(s).\n")

    # =========================================================================
    # STEP 3: Execute actionable sub-tasks
    # =========================================================================
    results = []
    output_files = []

    for i, task in enumerate(actionable, 1):
        outcome = execute_subtask(
            task=task,
            user_query=user_query,
            task_index=i,
            output_dir=output_dir,
            max_items=max_items,
        )
        results.append({"task": task, "outcome": outcome})

        if outcome.get("csv_path"):
            output_files.append(outcome["csv_path"])

    # =========================================================================
    # STEP 3.5: Merge structured data outputs from same source
    # =========================================================================
    output_files, results = merge_same_source_outputs(results, output_files, output_dir)

    # =========================================================================
    # STEP 4: Summary
    # =========================================================================
    elapsed = time.time() - pipeline_start

    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"Query:          \"{user_query}\"")
    print(f"Total time:     {elapsed:.0f}s ({elapsed/60:.1f} minutes)")
    print(f"Sub-tasks:      {len(tasks)} total, {len(actionable)} executed, {len(unfulfilled)} unfulfilled")

    for i, r in enumerate(results, 1):
        outcome = r["outcome"]
        status = "✓" if outcome["success"] else "✗"
        pipeline = outcome.get("pipeline_type", "unknown")
        print(f"\n  Task {i}: {r['task']['description']}")
        print(f"    [{status}] {outcome['message']}")
        print(f"    Pipeline: {pipeline}")
        if outcome.get("columns"):
            print(f"    Columns: {', '.join(outcome['columns'][:8])}")

    if unfulfilled:
        print(f"\n  Unfulfilled:")
        for t in unfulfilled:
            print(f"    ✗ {t['description']}")

    if output_files:
        print(f"\n  Output files:")
        for f in output_files:
            print(f"    {f}")
    else:
        print(f"\n  No output files generated.")

    # Save metadata
    meta_path = os.path.join(output_dir, "pipeline_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "query": user_query,
            "plan": plan,
            "results": [
                {"task_description": r["task"]["description"], "outcome": r["outcome"]}
                for r in results
            ],
            "output_files": output_files,
            "elapsed": elapsed,
            "timestamp": timestamp,
        }, f, indent=2, default=str)

    return {
        "query": user_query,
        "plan": plan,
        "results": results,
        "output_files": output_files,
        "elapsed": elapsed,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python orchestrator.py "Find all threats Trump made to Iran in 2025 and 2026"')
        print()
        print("Options:")
        print("  --max N    Maximum items to process (default: 50)")
        sys.exit(1)

    # Parse args
    args = sys.argv[1:]
    max_items = DEFAULT_MAX_ITEMS

    if "--max" in args:
        idx = args.index("--max")
        max_items = int(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

    query = " ".join(args)
    run_pipeline(query, max_items=max_items)
