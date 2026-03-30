"""
Query Decomposer — splits a user query into sub-tasks matched to data sources.

Given a natural language query and the source registry, the LLM:
  1. Identifies distinct information needs within the query
  2. Matches each need to the best available data source
  3. Flags needs that can't be fulfilled (no matching source)
  4. Produces a structured plan for the orchestrator

This runs BEFORE any data fetching — it's the planning step.

Dependencies:
  - source_registry.py (reads registered sources)
  - generate_api_spec.py (for LLM calls)
"""

import json
import time
from datetime import datetime

from source_registry import format_registry_for_prompt, get_source, get_available_sources
from generate_api_spec import call_llm, parse_json_response


# =============================================================================
# DECOMPOSITION PROMPT
# =============================================================================

DECOMPOSITION_PROMPT = """You are a research query planner. Given a user's query and a catalog of available data sources, decompose the query into sub-tasks that can each be answered by a specific data source.

TODAY'S DATE: {current_date}

USER QUERY: "{user_query}"

AVAILABLE DATA SOURCES:
{registry_text}

Analyze the query and break it into the minimum number of independent sub-tasks. Each sub-task should map to exactly one data source.

Return ONLY a JSON object:
{{
    "sub_tasks": [
        {{
            "description": "What this sub-task needs to find",
            "source_id": "The ID of the data source to use, or 'none' if no source can answer this",
            "reason": "Why this source was chosen, or why no source fits"
        }}
    ],
    "overall_feasibility": "one of: 'fully_feasible' (all sub-tasks have available sources), 'partially_feasible' (some sub-tasks lack sources), 'not_feasible' (no sub-tasks can be fulfilled)",
    "user_message": "If partially or not feasible: a plain-language message explaining what can and cannot be done. If fully feasible: null."
}}

Rules:
- Do NOT split a query unnecessarily. If the entire query maps to one source, return one sub-task.
- IMPORTANT EXCEPTION — multiple data points from one source: If the user requests multiple specific metrics, indicators, or data types from the same source (e.g., "GDP, population, and trade"), you MUST create a separate sub-task for each one. This is required because most APIs can only retrieve one type of data per call. Do NOT combine them into a single sub-task even though they use the same source.
- Each sub-task must be independently answerable — it should not depend on the output of another sub-task.
- If a sub-task doesn't match ANY registered source, use source_id "none" and explain in the reason.
- Match sub-tasks to sources based on the source's description and capabilities.
- Each sub-task description must be a complete, actionable sentence starting with a verb (e.g., "Get Iran's trade data" not "Iran's trade"). The description is passed directly to the API query builder, so it must be specific enough to construct the right API call.
- Only include date/time qualifiers in a sub-task description if the user's query explicitly applies those dates to that specific information need. Dates often apply to only part of a multi-topic query — do not assume they apply to every sub-task.
- Do NOT reject queries based on dates. Any dates referenced in the query are valid. Your job is to route, not to judge whether data exists.

Return ONLY the JSON object, no other text.
"""


# =============================================================================
# DECOMPOSE
# =============================================================================

def decompose_query(user_query: str) -> dict:
    """Split a user query into sub-tasks matched to data sources.
    
    Returns:
        {
            "sub_tasks": [...],
            "overall_feasibility": "fully_feasible" | "partially_feasible" | "not_feasible",
            "user_message": str | None,
            "elapsed": float,
        }
    """
    registry_text = format_registry_for_prompt()

    if registry_text == "No data sources registered.":
        return {
            "sub_tasks": [],
            "overall_feasibility": "not_feasible",
            "user_message": "No data sources are registered. Please register at least one source first.",
            "elapsed": 0,
        }

    prompt = DECOMPOSITION_PROMPT.format(
        current_date=datetime.now().strftime("%Y-%m-%d"),
        user_query=user_query,
        registry_text=registry_text,
    )

    start = time.time()
    raw = call_llm(prompt)
    elapsed = time.time() - start

    result = parse_json_response(raw)
    if not result or "sub_tasks" not in result:
        # Fallback: treat entire query as single task, guess best source
        available = get_available_sources()
        fallback_source = available[0]["id"] if available else "none"
        return {
            "sub_tasks": [{
                "description": user_query,
                "source_id": fallback_source,
                "reason": "Decomposition failed — defaulting to first available source",
            }],
            "overall_feasibility": "fully_feasible" if available else "not_feasible",
            "user_message": None,
            "elapsed": elapsed,
        }

    # Enrich sub-tasks with source metadata
    for task in result["sub_tasks"]:
        source_id = task.get("source_id", "none")
        source = get_source(source_id)
        if source:
            task["source_name"] = source["name"]
            task["source_available"] = source["status"] == "ready"
        elif source_id == "none":
            task["source_name"] = None
            task["source_available"] = False
        else:
            task["source_name"] = None
            task["source_available"] = False
            task["reason"] = f"{task.get('reason', '')} (source '{source_id}' not found in registry)"

    # Override LLM's feasibility judgment — compute from actual source availability
    tasks = result["sub_tasks"]
    actionable = [t for t in tasks if t.get("source_available")]
    unfulfilled = [t for t in tasks if not t.get("source_available")]

    if len(actionable) == len(tasks):
        result["overall_feasibility"] = "fully_feasible"
        result["user_message"] = None
    elif len(actionable) > 0:
        result["overall_feasibility"] = "partially_feasible"
        unfulfilled_descs = [t["description"] for t in unfulfilled]
        result["user_message"] = (
            f"{len(actionable)} sub-task(s) can be executed. "
            f"{len(unfulfilled)} sub-task(s) cannot be fulfilled: {'; '.join(unfulfilled_descs)}"
        )
    else:
        result["overall_feasibility"] = "not_feasible"
        result["user_message"] = "No registered data source can answer this query."

    result["elapsed"] = elapsed
    return result


# =============================================================================
# DISPLAY
# =============================================================================

def print_plan(plan: dict):
    """Pretty-print the decomposition plan."""
    tasks = plan.get("sub_tasks", [])

    print(f"\n  Sub-tasks ({len(tasks)}):")
    for i, task in enumerate(tasks, 1):
        available = task.get("source_available", False)
        source_name = task.get("source_name", "None")

        if source_name and available:
            status = "✓ AVAILABLE"
        elif source_name and not available:
            status = "✗ NOT AVAILABLE"
        else:
            status = "✗ NO MATCHING SOURCE"

        print(f"\n    Task {i}: {task['description']}")
        print(f"      Source:       {source_name or 'None'} [{status}]")
        print(f"      Reason:       {task['reason']}")

    print(f"\n  Feasibility: {plan.get('overall_feasibility', 'unknown')}")
    if plan.get("user_message"):
        print(f"  Message:     {plan['user_message']}")
    print(f"  Time:        {plan.get('elapsed', 0):.1f}s")


# =============================================================================
# TEST SUITE
# =============================================================================

TEST_QUERIES = [
    # Simple single-source query
    "Find all threats that Trump has made to Iran in 2025 and 2026",

    # Multi-source: one available, one not
    "Find all threats that Trump has made to Iran in 2025 and 2026 and give me Iran's economic data",

    # Single source, different topic
    "What have Federal Reserve governors said about interest rates in 2026?",

    # Likely no matching source
    "What is the current stock price of Apple?",

    # Complex single source
    "Did Trump and other members of the Trump administration say similar things about Iraq?",

    # Very broad / ambiguous
    "Tell me everything about Iran",
]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run test suite
        print("=" * 70)
        print("QUERY DECOMPOSITION TEST")
        print("=" * 70)

        # Show what the decomposer can see
        print("\nRegistered sources:")
        registry_text = format_registry_for_prompt()
        for line in registry_text.split("\n"):
            if line.strip():
                print(f"  {line}")
        print()

        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"\n{'─' * 70}")
            print(f"QUERY {i}: \"{query}\"")
            print(f"{'─' * 70}")

            plan = decompose_query(query)
            print_plan(plan)

            # Assessment
            tasks = plan.get("sub_tasks", [])
            actionable = [t for t in tasks if t.get("source_available")]
            unfulfilled = [t for t in tasks if not t.get("source_available")]

            print(f"\n  Assessment:")
            if actionable:
                print(f"    ✓ {len(actionable)} sub-task(s) can be executed")
            if unfulfilled:
                print(f"    ✗ {len(unfulfilled)} sub-task(s) cannot be fulfilled")

        print(f"\n{'=' * 70}")
        print("DONE")
        print(f"{'=' * 70}")

    else:
        # Interactive mode
        print("=" * 70)
        print("QUERY DECOMPOSER")
        print("=" * 70)

        registry_text = format_registry_for_prompt()
        if "No data sources" in registry_text:
            print("\n  No sources registered. Run: python source_registry.py register")
            sys.exit(1)

        print("\nRegistered sources:")
        for line in registry_text.split("\n"):
            if line.strip():
                print(f"  {line}")

        print("\nEnter a query (or 'quit' to exit):\n")
        while True:
            query = input("Query: ").strip()
            if not query or query.lower() in ("quit", "exit", "q"):
                break

            plan = decompose_query(query)
            print_plan(plan)
            print()
