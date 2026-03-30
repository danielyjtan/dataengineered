"""
Source Registry — add and validate data sources.

User provides three things:
  1. name         — what to call this source ("GDELT", "World Bank")
  2. description  — plain language: what data does this API provide?
  3. doc_urls     — list of documentation page URLs

The system then validates the source through a multi-step pipeline:
  Step 1: Fetch docs     — can we access the URLs and get readable text?
  Step 2: Generate spec  — run decomposed LLM calls to produce an API spec
  Step 3: Validate spec  — does the spec have the minimum required fields?
  Step 4: Test call      — can we hit the API and get a real response?

Sources are stored as JSON files in a registry directory.

Dependencies:
  - generate_api_spec.py (spec generation)
"""

import json
import os
import time
from datetime import datetime

import requests

from generate_api_spec import (
    fetch_documentation,
    generate_api_spec_decomposed,
    merge_specs,
    post_process_spec,
    call_llm,
    parse_json_response,
)


# =============================================================================
# CONFIG
# =============================================================================

REGISTRY_DIR = "source_registry"
SPECS_DIR = os.path.join(REGISTRY_DIR, "specs")


# =============================================================================
# STORAGE
# =============================================================================

def _ensure_dirs():
    """Create registry directories if they don't exist."""
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    os.makedirs(SPECS_DIR, exist_ok=True)


def _source_path(source_id: str) -> str:
    return os.path.join(REGISTRY_DIR, f"{source_id}.json")


def _spec_path(source_id: str) -> str:
    return os.path.join(SPECS_DIR, f"{source_id}_spec.json")


def _make_id(name: str) -> str:
    """Convert a name to a filesystem-safe ID."""
    return name.lower().strip().replace(" ", "_").replace("-", "_")


# =============================================================================
# STEP 1: FETCH DOCS
# =============================================================================

def validate_docs(doc_urls: list[str]) -> dict:
    """Check that documentation URLs are accessible and return readable text.
    
    Returns:
        {"passed": bool, "results": [{url, status, char_count, error}], "message": str}
    """
    results = []
    all_passed = True

    for url in doc_urls:
        entry = {"url": url, "status": None, "char_count": 0, "error": None}
        try:
            text = fetch_documentation(url)
            entry["char_count"] = len(text)

            if len(text) < 200:
                entry["status"] = "too_short"
                entry["error"] = f"Only {len(text)} chars retrieved — may not be real documentation"
                all_passed = False
            else:
                entry["status"] = "ok"

        except Exception as e:
            entry["status"] = "failed"
            entry["error"] = str(e)
            all_passed = False

        results.append(entry)
        print(f"    {url[:60]}... → {entry['status']} ({entry['char_count']} chars)")

    if all_passed:
        message = f"All {len(doc_urls)} documentation page(s) accessible"
    else:
        failed = [r for r in results if r["status"] != "ok"]
        message = f"{len(failed)}/{len(doc_urls)} page(s) had issues"

    return {"passed": all_passed, "results": results, "message": message}


# =============================================================================
# STEP 2: GENERATE SPEC (with Layer 1 smart merge + Layer 2 self-review)
# =============================================================================

def generate_spec(doc_urls: list[str]) -> dict:
    """Generate API spec from documentation URLs with smart merge and review.
    
    Layer 1 — Smart merge:
      - Generates a spec per URL independently
      - Checks if all specs share the same base_url
      - If different base_urls: picks the primary, warns about the rest
      - If same base_url: merges as before
    
    Layer 2 — Self-review:
      - After spec generation, gives the LLM the raw docs + generated spec
      - Asks it to find missing parameters, operators, or filters
      - Patches the spec with any findings
    
    Returns:
        {"passed": bool, "spec": dict|None, "message": str, "elapsed": float,
         "warnings": list[str]}
    """
    start = time.time()
    warnings = []

    try:
        # ==== LAYER 1: Generate per-URL, then smart merge ====

        if len(doc_urls) == 1:
            # Single page — straightforward
            doc_text = fetch_documentation(doc_urls[0])
            spec = generate_api_spec_decomposed(doc_text)
            if spec:
                spec = post_process_spec(spec)
            doc_texts = {doc_urls[0]: doc_text}
        else:
            # Multiple pages — generate independently, then check base URLs
            partial_specs = []
            doc_texts = {}

            for url in doc_urls:
                page_name = url.split("/")[-1][:50] if "/" in url else url[:50]
                print(f"\n{'=' * 60}")
                print(f"Processing: {page_name}")
                print(f"{'=' * 60}")

                try:
                    doc_text = fetch_documentation(url)
                    doc_texts[url] = doc_text
                    print(f"  Fetched {len(doc_text)} chars")

                    page_spec = generate_api_spec_decomposed(doc_text)
                    if page_spec:
                        page_spec["_source_url"] = url
                        partial_specs.append(page_spec)
                    else:
                        print(f"  Failed to generate spec for this page")
                except Exception as e:
                    print(f"  Error processing {url}: {e}")

            if not partial_specs:
                elapsed = time.time() - start
                return {
                    "passed": False,
                    "spec": None,
                    "message": "No specs generated from any page",
                    "elapsed": elapsed,
                    "warnings": [],
                }

            # --- Layer 1: Check base URLs before merging ---
            # Normalize URLs: strip trailing slashes, then group URLs where
            # one is a prefix of another (handles /v2 vs /v2/ vs / variations)
            def normalize_base_url(url: str) -> str:
                return url.rstrip("/")

            def urls_are_same_api(url_a: str, url_b: str) -> bool:
                a = normalize_base_url(url_a)
                b = normalize_base_url(url_b)
                return a == b or a.startswith(b) or b.startswith(a)

            # Group specs by API (not by exact URL string)
            groups = []  # list of (canonical_url, [specs])
            for ps in partial_specs:
                base = ps.get("base_url", "unknown")
                placed = False
                for group in groups:
                    if urls_are_same_api(base, group[0]):
                        group[1].append(ps)
                        # Use the longest URL as canonical (most specific)
                        if len(normalize_base_url(base)) > len(normalize_base_url(group[0])):
                            group[0] = base
                        placed = True
                        break
                if not placed:
                    groups.append([base, [ps]])

            if len(groups) > 1:
                # Different APIs detected!
                print(f"\n{'=' * 60}")
                print(f"⚠ DIFFERENT API ENDPOINTS DETECTED")
                print(f"{'=' * 60}")
                for base, specs in groups:
                    sources = [s.get("_source_url", "?")[:60] for s in specs]
                    print(f"  {base}")
                    for src in sources:
                        print(f"    ← {src}")

                # Pick the group with the most pages (or first if tied)
                groups.sort(key=lambda g: len(g[1]), reverse=True)
                primary_base, primary_specs = groups[0]
                other_bases = [g[0] for g in groups[1:]]

                warnings.append(
                    f"Documentation pages describe {len(groups)} different API endpoints. "
                    f"Using primary endpoint: {primary_base}. "
                    f"Excluded endpoints: {', '.join(other_bases)}. "
                    f"Register excluded endpoints as separate sources if needed."
                )

                print(f"\n  → Using primary: {primary_base} ({len(primary_specs)} page(s))")
                print(f"  → Excluding: {', '.join(other_bases)}")

                partial_specs = primary_specs
            else:
                # All pages describe the same API — merge all
                partial_specs = groups[0][1]

            # Merge (only same-endpoint specs)
            if len(partial_specs) == 1:
                spec = post_process_spec(partial_specs[0])
            else:
                print(f"\n{'=' * 60}")
                print(f"MERGING {len(partial_specs)} PARTIAL SPECS")
                print(f"{'=' * 60}")
                spec = merge_specs(partial_specs)
                spec = post_process_spec(spec)

        if spec is None:
            elapsed = time.time() - start
            return {
                "passed": False,
                "spec": None,
                "message": "Spec generation returned nothing",
                "elapsed": elapsed,
                "warnings": warnings,
            }

        # Clean up internal fields
        spec.pop("_source_url", None)

        # Print spec summary
        print(f"\n  Spec summary:")
        print(f"    Name: {spec.get('name')}")
        print(f"    Base URL: {spec.get('base_url')}")
        print(f"    Parameters: {len(spec.get('parameters', []))}")
        qs = spec.get("query_syntax")
        if qs:
            print(f"    Operators: {len(qs.get('operators', []))}")
            print(f"    Filters: {qs.get('filters_in_query', [])}")

        # ==== LAYER 2: Self-review ====
        print(f"\n  Running spec self-review...")
        all_doc_text = "\n\n---\n\n".join(doc_texts.values())
        review_result = review_spec(spec, all_doc_text)

        if review_result.get("patches_applied", 0) > 0:
            spec = review_result["spec"]
            print(f"  → Applied {review_result['patches_applied']} patch(es)")
            for patch in review_result.get("patches", []):
                print(f"    + {patch}")
        else:
            print(f"  → No gaps found")

        if review_result.get("warnings"):
            warnings.extend(review_result["warnings"])

        elapsed = time.time() - start
        return {
            "passed": True,
            "spec": spec,
            "message": f"Spec generated and reviewed",
            "elapsed": elapsed,
            "warnings": warnings,
        }

    except Exception as e:
        return {
            "passed": False,
            "spec": None,
            "message": f"Spec generation error: {e}",
            "elapsed": time.time() - start,
            "warnings": warnings,
        }


# =============================================================================
# LAYER 2: SPEC SELF-REVIEW
# =============================================================================

SPEC_REVIEW_PROMPT = """You are reviewing an auto-generated API specification against the original documentation to find gaps.

GENERATED SPEC:
{spec_json}

ORIGINAL DOCUMENTATION:
{doc_text}

Compare the spec against the documentation carefully. Look for:
1. PARAMETERS mentioned in the docs but missing from the spec's "parameters" list
2. QUERY OPERATORS or search syntax mentioned in the docs but missing from the spec's "operators" list
3. FILTERS (colon-prefixed syntax patterns that go inside a query parameter) mentioned in the docs but missing from "filters_in_query"
4. Incorrect default values or constraints in existing parameters

For each gap found, provide the exact details needed to add it.

Return ONLY a JSON object:
{{
    "missing_parameters": [
        {{
            "name": "param_name",
            "location": "query_string",
            "type": "string",
            "required": false,
            "default": null,
            "description": "What this parameter does, from the docs"
        }}
    ],
    "missing_operators": [
        {{
            "name": "Operator Name",
            "syntax": "exact:syntax as shown in docs",
            "description": "What it does",
            "constraints": "Any limitations"
        }}
    ],
    "missing_filters": ["filter_prefix"],
    "corrections": [
        {{
            "field": "what needs fixing",
            "current": "current value",
            "should_be": "correct value",
            "reason": "why"
        }}
    ]
}}

If no gaps are found, return empty arrays for all fields.
Return ONLY the JSON object, no other text.
"""


def review_spec(spec: dict, doc_text: str) -> dict:
    """Review a generated spec against the original docs and patch gaps.
    
    Returns:
        {"spec": dict, "patches_applied": int, "patches": [str], "warnings": [str]}
    """
    # Truncate doc text if too long (keep it under ~30K for the LLM)
    if len(doc_text) > 30000:
        doc_text = doc_text[:30000] + "\n... (truncated)"

    prompt = SPEC_REVIEW_PROMPT.format(
        spec_json=json.dumps(spec, indent=2),
        doc_text=doc_text[:30000],
    )

    start = time.time()
    raw = call_llm(prompt)
    elapsed = time.time() - start
    print(f"    Review call: {elapsed:.1f}s")

    result = parse_json_response(raw)
    if not result:
        return {"spec": spec, "patches_applied": 0, "patches": [], "warnings": ["Spec review LLM call failed"]}

    patches = []
    warnings = []

    # Patch missing parameters
    existing_params = {p["name"] for p in spec.get("parameters", [])}
    for param in result.get("missing_parameters", []):
        name = param.get("name", "")
        if name and name not in existing_params:
            spec["parameters"].append(param)
            patches.append(f"Added parameter: {name}")
            existing_params.add(name)

    # Patch missing operators
    if spec.get("query_syntax"):
        existing_ops = set()
        for op in spec["query_syntax"].get("operators", []):
            existing_ops.add(op.get("name", "").lower())
            existing_ops.add(op.get("syntax", "").split(":")[0].lower().strip('"'))

        for op in result.get("missing_operators", []):
            op_name = op.get("name", "")
            op_syntax_prefix = op.get("syntax", "").split(":")[0].lower().strip('"')
            # Check if this operator is truly new (not a duplicate with different name)
            if op_name.lower() not in existing_ops and op_syntax_prefix not in existing_ops:
                spec["query_syntax"]["operators"].append(op)
                patches.append(f"Added operator: {op_name} ({op.get('syntax', '')})")

    # Patch missing filters
    if spec.get("query_syntax"):
        existing_filters = set(spec["query_syntax"].get("filters_in_query", []))
        for f in result.get("missing_filters", []):
            if f and f not in existing_filters:
                spec["query_syntax"]["filters_in_query"].append(f)
                patches.append(f"Added filter: {f}")

    # Apply corrections
    for correction in result.get("corrections", []):
        field = correction.get("field", "")
        should_be = correction.get("should_be", "")
        reason = correction.get("reason", "")
        if field and should_be:
            warnings.append(f"Suggested correction: {field} → {should_be} ({reason})")

    return {
        "spec": spec,
        "patches_applied": len(patches),
        "patches": patches,
        "warnings": warnings,
    }


# =============================================================================
# STEP 3: VALIDATE SPEC
# =============================================================================

REQUIRED_FIELDS = ["name", "base_url", "parameters"]


def validate_spec(spec: dict) -> dict:
    """Check that the generated spec has the minimum required structure.
    
    Returns:
        {"passed": bool, "warnings": [str], "message": str}
    """
    warnings = []

    # Check required top-level fields
    for field in REQUIRED_FIELDS:
        if not spec.get(field):
            warnings.append(f"Missing required field: '{field}'")

    # Check base_url looks like a URL
    base_url = spec.get("base_url", "")
    if base_url and not base_url.startswith("http"):
        warnings.append(f"base_url doesn't look like a URL: '{base_url}'")

    # Check parameters are a non-empty list
    params = spec.get("parameters", [])
    if not params:
        warnings.append("No parameters found — the API may not be callable")
    else:
        unnamed = [p for p in params if not p.get("name")]
        if unnamed:
            warnings.append(f"{len(unnamed)} parameter(s) missing a name")

    # Check for url_template or examples
    has_template = bool(spec.get("url_template"))
    has_examples = bool((spec.get("query_syntax") or {}).get("examples"))
    if not has_template and not has_examples:
        warnings.append("No url_template or examples found — API call construction may be unreliable")

    # Classify severity
    critical = [w for w in warnings if "Missing required" in w or "No parameters" in w]
    passed = len(critical) == 0

    if passed and not warnings:
        message = "Spec is valid"
    elif passed:
        message = f"Spec is valid with {len(warnings)} warning(s)"
    else:
        message = f"Spec has {len(critical)} critical issue(s)"

    return {"passed": passed, "warnings": warnings, "message": message}


# =============================================================================
# STEP 4: TEST CALL
# =============================================================================

TEST_URL_PROMPT = """Given this API specification, construct ONE valid test URL that should return a successful response.

API SPECIFICATION:
{spec_json}

Rules:
- Use real parameter values from the examples or parameter descriptions in the spec.
- Include format=json if the API supports a format parameter.
- Keep it minimal — use the fewest parameters needed for a valid call.
- The URL must be complete and ready to execute with no modifications.
- Pay attention to the base_url and url_template — construct the URL so path parameters are embedded in the path, not as query strings.
- If the spec contains examples with query values, use those as a reference for valid parameter values.

Return ONLY the complete URL, nothing else.
"""


def generate_test_url(spec: dict) -> str | None:
    """Ask the LLM to construct a valid test URL from the spec."""
    prompt = TEST_URL_PROMPT.format(spec_json=json.dumps(spec, indent=2))

    start = time.time()
    raw = call_llm(prompt)
    elapsed = time.time() - start
    print(f"    LLM generated test URL ({elapsed:.1f}s)")

    # Extract URL from response (LLM might add explanation despite instructions)
    url = raw.strip()
    # If it's wrapped in quotes or backticks, clean up
    url = url.strip("`'\"")
    # If the LLM returned multiple lines, take the first one that looks like a URL
    for line in url.split("\n"):
        line = line.strip().strip("`'\"")
        if line.startswith("http"):
            return line

    return None


def test_api_call(spec: dict) -> dict:
    """Test the API by having the LLM construct a valid URL, then executing it.
    
    Returns:
        {"passed": bool, "status_code": int|None, "message": str}
    """
    # Ask LLM to build a test URL from the spec
    print("    Generating test URL from spec...")
    test_url = generate_test_url(spec)

    if not test_url:
        return {
            "passed": False,
            "status_code": None,
            "message": "LLM could not construct a test URL from the spec",
        }

    print(f"    Test URL: {test_url[:120]}...")

    # Execute the URL
    try:
        resp = requests.get(test_url, timeout=15)
        status = resp.status_code

        if status == 200:
            try:
                data = resp.json()
                return {
                    "passed": True,
                    "status_code": 200,
                    "response_type": type(data).__name__,
                    "message": f"API responded with {type(data).__name__} data",
                    "test_url": test_url,
                }
            except ValueError:
                content_type = resp.headers.get("Content-Type", "")
                return {
                    "passed": True,
                    "status_code": 200,
                    "response_type": content_type[:50],
                    "message": f"API responded (non-JSON: {content_type[:50]})",
                    "test_url": test_url,
                }
        elif status == 429:
            return {
                "passed": True,
                "status_code": 429,
                "message": "API returned 429 Rate Limited — API is reachable but rate limited",
                "test_url": test_url,
            }
        elif status == 403:
            return {
                "passed": False,
                "status_code": 403,
                "message": "API returned 403 Forbidden — may require authentication",
                "test_url": test_url,
            }
        else:
            return {
                "passed": False,
                "status_code": status,
                "message": f"API returned HTTP {status} — the spec may have issues",
                "test_url": test_url,
            }

    except requests.exceptions.Timeout:
        return {
            "passed": False,
            "status_code": None,
            "message": "Request timed out (15s)",
            "test_url": test_url,
        }
    except requests.exceptions.ConnectionError as e:
        return {
            "passed": False,
            "status_code": None,
            "message": f"Connection failed: {e}",
            "test_url": test_url,
        }
    except Exception as e:
        return {
            "passed": False,
            "status_code": None,
            "message": f"Unexpected error: {e}",
            "test_url": test_url,
        }


# =============================================================================
# REGISTRATION ORCHESTRATOR
# =============================================================================

def register_source(
    name: str,
    description: str,
    doc_urls: list[str],
    skip_test_call: bool = False,
) -> dict:
    """
    Register a new data source. Runs the full validation pipeline.
    
    Args:
        name: Human-readable name (e.g., "GDELT")
        description: What data this API provides
        doc_urls: List of documentation page URLs
        skip_test_call: Skip the test API call (useful if API is rate-limited)
    
    Returns:
        Source record dict with status
    """
    _ensure_dirs()
    source_id = _make_id(name)

    print("=" * 70)
    print(f"REGISTERING SOURCE: {name}")
    print("=" * 70)
    print(f"  ID:          {source_id}")
    print(f"  Description: {description[:80]}...")
    print(f"  Doc URLs:    {len(doc_urls)} page(s)")
    print()

    source = {
        "id": source_id,
        "name": name,
        "description": description,
        "doc_urls": doc_urls,
        "status": "pending",
        "spec_path": None,
        "validation": {},
        "registered_at": datetime.now().isoformat(),
    }

    # Step 1: Fetch docs
    print("Step 1: Checking documentation URLs...")
    doc_result = validate_docs(doc_urls)
    source["validation"]["docs"] = doc_result
    print(f"  → {doc_result['message']}\n")

    if not doc_result["passed"]:
        source["status"] = "failed_docs"
        _save_source(source_id, source)
        print(f"✗ Registration failed at Step 1: documentation not accessible")
        return source

    # Step 2: Generate spec (with smart merge + self-review)
    print("Step 2: Generating API spec from documentation...")
    print("  (This takes several minutes — 5 LLM calls per page + review pass)\n")
    spec_result = generate_spec(doc_urls)
    source["validation"]["spec_generation"] = {
        "passed": spec_result["passed"],
        "message": spec_result["message"],
        "elapsed": spec_result["elapsed"],
    }
    print(f"\n  → {spec_result['message']} ({spec_result['elapsed']:.0f}s)")

    # Show any warnings from Layer 1/2
    for w in spec_result.get("warnings", []):
        print(f"    ⚠ {w}")
    print()

    if not spec_result["passed"]:
        source["status"] = "failed_spec_generation"
        _save_source(source_id, source)
        print(f"✗ Registration failed at Step 2: could not generate API spec")
        return source

    spec = spec_result["spec"]

    # Step 3: Validate spec
    print("Step 3: Validating generated spec...")
    val_result = validate_spec(spec)
    source["validation"]["spec_validation"] = val_result
    print(f"  → {val_result['message']}")
    for w in val_result.get("warnings", []):
        print(f"    ⚠ {w}")
    print()

    if not val_result["passed"]:
        source["status"] = "failed_spec_validation"
        _save_spec(source_id, spec)
        source["spec_path"] = _spec_path(source_id)
        _save_source(source_id, source)
        print(f"✗ Registration failed at Step 3: generated spec has critical issues")
        return source

    # Save spec
    _save_spec(source_id, spec)
    source["spec_path"] = _spec_path(source_id)

    # Step 4: Test call
    if skip_test_call:
        print("Step 4: Skipped (test call disabled)\n")
        source["validation"]["test_call"] = {"passed": None, "message": "Skipped"}
    else:
        print("Step 4: Testing API connectivity...")
        test_result = test_api_call(spec)
        source["validation"]["test_call"] = test_result
        print(f"  → {test_result['message']}\n")

        if not test_result["passed"]:
            source["status"] = "failed_test_call"
            _save_source(source_id, source)
            print(f"✗ Registration failed at Step 4: API not reachable")
            return source

    # All passed
    source["status"] = "ready"
    _save_source(source_id, source)

    print("=" * 70)
    print(f"✓ SOURCE REGISTERED: {name}")
    print("=" * 70)
    print(f"  Status:     ready")
    print(f"  Spec:       {source['spec_path']}")
    print(f"  Base URL:   {spec.get('base_url', 'N/A')}")
    print(f"  Parameters: {len(spec.get('parameters', []))}")
    if spec.get("url_template"):
        print(f"  Template:   {spec['url_template']}")

    return source


# =============================================================================
# STORAGE HELPERS
# =============================================================================

def _save_source(source_id: str, source: dict):
    _ensure_dirs()
    with open(_source_path(source_id), "w") as f:
        json.dump(source, f, indent=2)


def _save_spec(source_id: str, spec: dict):
    _ensure_dirs()
    with open(_spec_path(source_id), "w") as f:
        json.dump(spec, f, indent=2)


# =============================================================================
# REGISTRY API — for other modules to use
# =============================================================================

def list_sources() -> list[dict]:
    """List all registered sources."""
    _ensure_dirs()
    sources = []
    for filename in sorted(os.listdir(REGISTRY_DIR)):
        if filename.endswith(".json"):
            with open(os.path.join(REGISTRY_DIR, filename)) as f:
                sources.append(json.load(f))
    return sources


def get_source(source_id: str) -> dict | None:
    path = _source_path(source_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_available_sources() -> list[dict]:
    return [s for s in list_sources() if s.get("status") == "ready"]


def get_spec(source_id: str) -> dict | None:
    path = _spec_path(source_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def remove_source(source_id: str) -> bool:
    removed = False
    for path in [_source_path(source_id), _spec_path(source_id)]:
        if os.path.exists(path):
            os.remove(path)
            removed = True
    return removed


def format_registry_for_prompt() -> str:
    """Format sources for inclusion in LLM prompts (used by query decomposer)."""
    sources = list_sources()
    if not sources:
        return "No data sources registered."

    lines = []
    for source in sources:
        status = "AVAILABLE" if source["status"] == "ready" else f"NOT AVAILABLE ({source['status']})"
        lines.append(f"SOURCE: {source['name']} [{status}]")
        lines.append(f"  ID: {source['id']}")
        lines.append(f"  Description: {source['description']}")

        spec = get_spec(source["id"])
        if spec:
            lines.append(f"  Base URL: {spec.get('base_url', 'N/A')}")
            param_names = [p.get("name", "?") for p in spec.get("parameters", [])[:5]]
            lines.append(f"  Parameters: {', '.join(param_names)}")
            if spec.get("query_syntax"):
                ops = [o.get("name", "?") for o in spec["query_syntax"].get("operators", [])[:5]]
                if ops:
                    lines.append(f"  Query operators: {', '.join(ops)}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("=" * 70)
        print("SOURCE REGISTRY")
        print("=" * 70)
        sources = list_sources()
        if not sources:
            print("  No sources registered.\n")
            print("  Usage:")
            print("    python source_registry.py register              # interactive")
            print("    python source_registry.py remove <source_id>    # remove a source")
            print("    python source_registry.py prompt                # show decomposer view")
        else:
            for s in sources:
                icon = "✓" if s["status"] == "ready" else "✗"
                print(f"\n  [{icon}] {s['name']} ({s['id']}) — {s['status']}")
                print(f"      {s['description'][:70]}...")
                if s.get("spec_path"):
                    print(f"      Spec: {s['spec_path']}")

    elif sys.argv[1] == "register":
        print("=" * 70)
        print("REGISTER NEW DATA SOURCE")
        print("=" * 70)
        print()
        name = input("Source name (e.g., 'GDELT'): ").strip()
        if not name:
            print("Name is required.")
            sys.exit(1)

        description = input("Description (what data does this API provide?): ").strip()
        if not description:
            print("Description is required.")
            sys.exit(1)

        print("Documentation URLs (one per line, empty line to finish):")
        doc_urls = []
        while True:
            url = input("  URL: ").strip()
            if not url:
                break
            if not url.startswith("http"):
                print("    ⚠ Must start with http:// or https://")
                continue
            doc_urls.append(url)

        if not doc_urls:
            print("At least one documentation URL is required.")
            sys.exit(1)

        print()
        register_source(name, description, doc_urls)

    elif sys.argv[1] == "remove":
        if len(sys.argv) < 3:
            print("Usage: python source_registry.py remove <source_id>")
            sys.exit(1)
        if remove_source(sys.argv[2]):
            print(f"Removed: {sys.argv[2]}")
        else:
            print(f"Not found: {sys.argv[2]}")

    elif sys.argv[1] == "prompt":
        print(format_registry_for_prompt())
