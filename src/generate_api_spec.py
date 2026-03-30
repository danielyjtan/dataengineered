"""
Auto-generate API spec from documentation.

Two modes:
  1. Single page:  generate_api_spec_decomposed(doc_text)
  2. Multi-page:   generate_api_spec_from_urls([url1, url2, ...])

Architecture:
  - Decomposed LLM calls (5 focused calls per page)
  - Multi-page merge (programmatic union of partial specs)
  - Post-processing (dedup path params, derive filters)

Dependencies: pip install html2text httpx
"""

import json
import time
import re
import requests
import httpx
import html2text


# =============================================================================
# FETCH DOCUMENTATION
# =============================================================================

def fetch_documentation(url: str, max_chars: int = 50000) -> str:
    """Fetch a documentation page and convert to clean text preserving URLs and code."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
    }
    resp = httpx.get(url, headers=headers, follow_redirects=True, timeout=15)
    h = html2text.HTML2Text()
    h.ignore_images = True
    h.ignore_emphasis = True
    h.body_width = 0
    text = h.handle(resp.text)
    if len(text) > max_chars:
        print(f"  Truncating from {len(text)} to {max_chars} chars")
        text = text[:max_chars]
    return text


# =============================================================================
# LLM CALL HELPER
# =============================================================================

def call_llm(prompt: str, model: str = "qwen3:30b", timeout: int = 600) -> str:
    """Send a prompt to the local LLM and return the raw text response."""
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1}
        },
        timeout=timeout,
    )
    raw = response.json()["message"]["content"]
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    return raw


def parse_json_response(raw: str) -> dict | list | None:
    """Parse JSON from LLM response, handling extra text."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try finding JSON object
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        # Try finding JSON array
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


# =============================================================================
# CALL 1: Base URL and Method
# =============================================================================

def extract_base_url(doc_text: str, model: str = "qwen3:30b") -> dict:
    prompt = f"""Look at this API documentation and find the base URL for API requests.

INSTRUCTIONS:
- Find ALL example URLs in the documentation
- Identify the common base URL that all examples share
- Determine the HTTP method (GET or POST)
- Also provide the API name

Return ONLY a JSON object:
{{
    "name": "Human-readable API name",
    "base_url": "The common base URL from the example URLs",
    "method": "GET or POST",
    "example_urls_found": ["list the first 5 example URLs you found"]
}}

Return ONLY the JSON, no other text.

DOCUMENTATION:
{doc_text}
"""
    print("  Call 1: Extracting base URL and method...")
    start = time.time()
    raw = call_llm(prompt, model)
    result = parse_json_response(raw)
    print(f"    Done ({time.time() - start:.1f}s)")
    if result:
        print(f"    Name: {result.get('name')}")
        print(f"    Base URL: {result.get('base_url')}")
        print(f"    Method: {result.get('method')}")
        print(f"    Example URLs found: {len(result.get('example_urls_found', []))}")
    return result


# =============================================================================
# CALL 2: Parameters
# =============================================================================

def extract_parameters(doc_text: str, model: str = "qwen3:30b") -> list:
    prompt = f"""Look at this API documentation and list every parameter the API accepts.

IMPORTANT: APIs have TWO types of parameters:
1. PATH parameters — embedded in the URL path itself, like /country/BR/ or /indicator/NY.GDP.MKTP.CD/
   Look at the example URLs. If parts of the URL path change between examples (e.g., different 
   country codes or indicator codes appear in the same position), those are path parameters.
2. QUERY STRING parameters — appear after the ? in the URL, like ?format=json&page=2

List BOTH types.

For each parameter, provide:
- name: the exact parameter name
- location: "path" if embedded in the URL path, "query_string" if after the ?
- type: string, integer, or boolean
- required: true or false
- default: the default value, or null
- description: what this parameter does. INCLUDE ALL VALID VALUES, format requirements, and constraints.

CRITICAL INSTRUCTIONS:
1. Pay close attention to date/time formats, encoding requirements, and parameter constraints

Return ONLY a JSON array of parameter objects. No other text.

DOCUMENTATION:
{doc_text}
"""
    print("  Call 2: Extracting parameters...")
    start = time.time()
    raw = call_llm(prompt, model)
    result = parse_json_response(raw)
    print(f"    Done ({time.time() - start:.1f}s)")
    if result:
        if isinstance(result, dict) and "parameters" in result:
            result = result["parameters"]
        print(f"    Found {len(result)} parameters")
        for p in result:
            loc = p.get('location', '?')
            print(f"      - {p.get('name')} ({loc})")
    return result


# =============================================================================
# CALL 3: Query Operators
# =============================================================================

def extract_operators(doc_text: str, model: str = "qwen3:30b") -> dict:
    prompt = f"""Look at this API documentation and determine if the API has a complex query/search language.

A complex query language means the API has operators like boolean OR, proximity search, phrase search, 
field-specific filters (e.g. domain:cnn.com, language:spanish), etc. that go INSIDE a query parameter value.

If the API has NO complex query language (just simple key-value parameters), return:
{{"has_query_syntax": false}}

If the API DOES have a complex query language, list every operator. For each operator:
- name: human-readable name
- syntax: a CONCRETE example exactly as it appears in the documentation. Use real values from the 
  docs, not abstract placeholders. If the docs show "sourcelang:spanish", write "sourcelang:spanish", 
  NOT "sourcelang:lang".
- description: what it does
- constraints: any limits

Return ONLY a JSON object:
{{
    "has_query_syntax": true,
    "description": "Brief description of the query language",
    "operators": [
        {{"name": "...", "syntax": "...", "description": "...", "constraints": "..."}}
    ]
}}

Return ONLY the JSON, no other text.

DOCUMENTATION:
{doc_text}
"""
    print("  Call 3: Extracting query operators...")
    start = time.time()
    raw = call_llm(prompt, model)
    result = parse_json_response(raw)
    print(f"    Done ({time.time() - start:.1f}s)")
    if result:
        has_qs = result.get("has_query_syntax", False)
        print(f"    Has query syntax: {has_qs}")
        if has_qs:
            ops = result.get("operators", [])
            print(f"    Operators found: {len(ops)}")
            for op in ops:
                print(f"      - {op.get('name')}: {op.get('syntax')}")
    return result


# =============================================================================
# CALL 4: Classify Filters (from Call 3 output)
# =============================================================================

def classify_filters(operators: list, model: str = "qwen3:30b") -> list:
    prompt = f"""Here is a list of query operators for an API. Some of these operators use a 
name:value syntax (like "domain:cnn.com" or "sourcelang:spanish").

For each operator that has a colon in its syntax, extract just the prefix before the colon.

OPERATORS:
{json.dumps(operators, indent=2)}

Return ONLY a JSON array of the prefixes. For example, if the operators include 
syntax "domain:cnn.com" and "sourcelang:spanish", return ["domain", "sourcelang"].

Only include operators that have a name:value pattern with a colon. Skip operators like 
"phrase" or "(a OR b)" that don't use the colon syntax.

Return ONLY the JSON array, no other text.
"""
    print("  Call 4: Classifying filters...")
    start = time.time()
    raw = call_llm(prompt, model)
    result = parse_json_response(raw)
    print(f"    Done ({time.time() - start:.1f}s)")
    if result:
        print(f"    Filters identified: {result}")
    return result


# =============================================================================
# CALL 5: Examples, Rate Limits, Auth, Quirks, URL Template
# =============================================================================

def extract_metadata(doc_text: str, parameters: list, model: str = "qwen3:30b") -> dict:
    # Check if any parameters are path-based
    has_path_params = any(p.get("location") == "path" for p in (parameters or []))

    url_template_instruction = ""
    if has_path_params:
        path_params = [p["name"] for p in parameters if p.get("location") == "path"]
        url_template_instruction = f"""
- url_template: This API has path parameters: {path_params}. Look at the example URLs in the 
  documentation and extract the URL path pattern showing where these parameters go. Use curly 
  braces for variables. For example, if URLs look like /v2/country/br/indicator/NY.GDP.MKTP.CD, 
  the template is "/v2/country/{{country}}/indicator/{{indicator}}"."""
    else:
        url_template_instruction = "\n- url_template: set to null (this API uses only query string parameters)"

    prompt = f"""Look at this API documentation and extract the following information.

Return a JSON object with these fields:
- url_template: see instructions below
- examples: at least 3 example queries showing different features of the API. Each example should 
  have "natural_language" (what someone would ask) and "query_value" (the actual query in 
  human-readable form, NOT URL-encoded)
- rate_limits: any rate limiting mentioned, and max results per request
- authentication: type (none, api_key, oauth, bearer_token) and description
- quirks_and_warnings: list of important gotchas or limitations
- response_format: the default response type (json, xml, csv, html)
{url_template_instruction}

Return ONLY the JSON object, no other text.

DOCUMENTATION:
{doc_text}
"""
    print("  Call 5: Extracting metadata (examples, rate limits, auth, quirks)...")
    start = time.time()
    raw = call_llm(prompt, model)
    result = parse_json_response(raw)
    print(f"    Done ({time.time() - start:.1f}s)")
    if result:
        print(f"    url_template: {result.get('url_template')}")
        print(f"    Examples: {len(result.get('examples', []))}")
        # print(f"    Auth: {result.get('authentication', {}).get('type', '?')}")
        print(f"    Quirks: {len(result.get('quirks_and_warnings', []))}")
        auth = result.get('authentication', {})
        if isinstance(auth, dict):
            print(f"    Auth: {auth.get('type', '?')}")
        else:
            print(f"    Auth: {auth}")
    return result


# =============================================================================
# ASSEMBLE SPEC (from single page)
# =============================================================================

def assemble_spec(base_info: dict, parameters: list, operators_info: dict,
                  filters: list, metadata: dict) -> dict:
    """Combine outputs from all calls into a single spec."""

    spec = {
        "name": base_info.get("name", "Unknown API"),
        "description": "",
        "base_url": base_info.get("base_url", ""),
        "url_template": metadata.get("url_template") if metadata else None,
        "method": base_info.get("method", "GET"),
        "parameters": parameters or [],
    }

    # Query syntax
    if operators_info and operators_info.get("has_query_syntax"):
        spec["query_syntax"] = {
            "description": operators_info.get("description", ""),
            "operators": operators_info.get("operators", []),
            "filters_in_query": filters or [],
            "examples": metadata.get("examples", []) if metadata else [],
        }
    else:
        spec["query_syntax"] = None

    # Response format
    if metadata and metadata.get("response_format"):
        rf = metadata["response_format"]
        if isinstance(rf, str):
            spec["response_format"] = {"type": rf, "results_key": None, "fields": []}
        else:
            spec["response_format"] = rf
    else:
        spec["response_format"] = {"type": "json", "results_key": None, "fields": []}

    # Rate limits
    spec["rate_limits"] = metadata.get("rate_limits", {}) if metadata else {}

    # Authentication
    spec["authentication"] = metadata.get("authentication", {"type": "none"}) if metadata else {"type": "none"}

    # Quirks
    spec["quirks_and_warnings"] = metadata.get("quirks_and_warnings", []) if metadata else []

    return spec


# =============================================================================
# SINGLE PAGE SPEC GENERATION
# =============================================================================

def generate_api_spec_decomposed(doc_text: str, model: str = "qwen3:30b") -> dict:
    """Generate an API spec from a single documentation page using decomposed LLM calls."""

    print("\n  Starting decomposed spec generation...\n")

    # Call 1: Base URL and method
    base_info = extract_base_url(doc_text, model)
    if not base_info:
        print("  ERROR: Failed to extract base URL")
        return None

    # Call 2: Parameters
    parameters = extract_parameters(doc_text, model)
    if not parameters:
        print("  WARNING: No parameters found")
        parameters = []

    # Call 3: Query operators
    operators_info = extract_operators(doc_text, model)

    # Call 4: Classify filters (only if we have operators)
    filters = []
    if operators_info and operators_info.get("has_query_syntax"):
        operators = operators_info.get("operators", [])
        if operators:
            filters = classify_filters(operators, model)
            if not filters:
                # Fallback: derive programmatically from operator syntax
                print("    Fallback: deriving filters from operator syntax...")
                filters = []
                for op in operators:
                    syntax = op.get("syntax", "")
                    if ":" in syntax:
                        prefix = syntax.split(":")[0]
                        filters.append(prefix)
                print(f"    Derived: {filters}")

    # Call 5: Metadata (examples, rate limits, auth, quirks, url_template)
    metadata = extract_metadata(doc_text, parameters, model)

    # Assemble
    print("\n  Assembling spec...")
    spec = assemble_spec(base_info, parameters, operators_info, filters, metadata)

    print(f"  ✓ Spec assembled: {spec.get('name')}")
    print(f"    Base URL: {spec.get('base_url')}")
    print(f"    Parameters: {len(spec.get('parameters', []))}")

    qs = spec.get("query_syntax")
    if qs:
        print(f"    Operators: {len(qs.get('operators', []))}")
        print(f"    Filters: {qs.get('filters_in_query', [])}")
    else:
        print(f"    Query syntax: None (simple REST API)")

    print(f"    URL template: {spec.get('url_template')}")

    return spec


# =============================================================================
# MULTI-PAGE MERGE
# =============================================================================

def merge_specs(partial_specs: list[dict]) -> dict:
    """Merge multiple partial specs into one complete spec.
    Each partial spec comes from a different documentation page.
    Uses union logic: parameters, operators, filters, quirks are combined.
    Conflicts resolved by: longest description wins, path > query_string,
    most common base_url wins.
    """

    if not partial_specs:
        return None
    if len(partial_specs) == 1:
        return partial_specs[0]

    merged = {
        "name": "",
        "description": "",
        "base_url": "",
        "url_template": None,
        "method": "GET",
        "parameters": [],
        "query_syntax": None,
        "response_format": {"type": "json", "results_key": None, "fields": []},
        "rate_limits": {},
        "authentication": {"type": "none"},
        "quirks_and_warnings": [],
    }

    # Track what we've seen for dedup
    seen_param_names = set()
    seen_operator_names = set()
    seen_filters = set()
    seen_quirks = set()
    seen_fields = set()
    base_url_votes = {}
    all_examples = []

    for spec in partial_specs:
        if not spec:
            continue

        # Name: take the longest (most descriptive)
        name = spec.get("name", "")
        if len(name) > len(merged["name"]):
            merged["name"] = name

        # Description: take the longest
        desc = spec.get("description", "")
        if len(desc) > len(merged["description"]):
            merged["description"] = desc

        # Base URL: vote — most common wins
        base = spec.get("base_url", "")
        if base:
            base_url_votes[base] = base_url_votes.get(base, 0) + 1

        # URL template: take first non-null
        if spec.get("url_template") and not merged["url_template"]:
            merged["url_template"] = spec["url_template"]

        # Method: take first non-default
        method = spec.get("method", "GET")
        if method != "GET":
            merged["method"] = method

        # Parameters: union by name
        for param in spec.get("parameters", []):
            name = param.get("name", "")
            if name and name not in seen_param_names:
                merged["parameters"].append(param)
                seen_param_names.add(name)
            elif name in seen_param_names:
                # Update existing if new version has more detail
                for existing in merged["parameters"]:
                    if existing["name"] == name:
                        # Prefer path over query_string (more specific)
                        if param.get("location") == "path" and existing.get("location") != "path":
                            existing["location"] = "path"
                        # Take longer description
                        if len(param.get("description", "")) > len(existing.get("description", "")):
                            existing["description"] = param["description"]
                        break

        # Query syntax: merge operators, filters, examples
        qs = spec.get("query_syntax")
        if qs and qs.get("operators"):
            if not merged["query_syntax"]:
                merged["query_syntax"] = {
                    "description": qs.get("description", ""),
                    "operators": [],
                    "filters_in_query": [],
                    "examples": [],
                }

            # Description: take longest
            if len(qs.get("description", "")) > len(merged["query_syntax"]["description"]):
                merged["query_syntax"]["description"] = qs["description"]

            # Operators: union by name
            for op in qs.get("operators", []):
                op_name = op.get("name", "")
                if op_name and op_name not in seen_operator_names:
                    merged["query_syntax"]["operators"].append(op)
                    seen_operator_names.add(op_name)

            # Filters: union
            for f in qs.get("filters_in_query", []):
                if f not in seen_filters:
                    merged["query_syntax"]["filters_in_query"].append(f)
                    seen_filters.add(f)

            # Examples: collect all
            for ex in qs.get("examples", []):
                all_examples.append(ex)

        # Response format: take most specific
        rf = spec.get("response_format", {})
        if isinstance(rf, dict):
            for field in rf.get("fields", []):
                if field not in seen_fields:
                    merged["response_format"]["fields"].append(field)
                    seen_fields.add(field)
            if rf.get("type") and rf["type"] != "json":
                merged["response_format"]["type"] = rf["type"]
            if rf.get("results_key"):
                merged["response_format"]["results_key"] = rf["results_key"]

        # Authentication: take first non-none
        auth = spec.get("authentication", {})
        if isinstance(auth, dict) and auth.get("type", "none") != "none":
            merged["authentication"] = auth

        # Rate limits: take most detailed
        rl = spec.get("rate_limits", {})
        if isinstance(rl, dict) and len(str(rl)) > len(str(merged["rate_limits"])):
            merged["rate_limits"] = rl

        # Quirks: union
        for q in spec.get("quirks_and_warnings", []):
            if q and q not in seen_quirks:
                merged["quirks_and_warnings"].append(q)
                seen_quirks.add(q)

    # Finalize base URL
    if base_url_votes:
        merged["base_url"] = max(base_url_votes, key=base_url_votes.get)

    # Finalize examples (dedup and limit to 5)
    seen_ex = set()
    for ex in all_examples:
        key = ex.get("query_value", "")
        if key not in seen_ex:
            if merged["query_syntax"]:
                merged["query_syntax"]["examples"].append(ex)
            seen_ex.add(key)
    if merged["query_syntax"] and len(merged["query_syntax"]["examples"]) > 5:
        merged["query_syntax"]["examples"] = merged["query_syntax"]["examples"][:5]

    return merged


# =============================================================================
# POST-PROCESSING
# =============================================================================

def post_process_spec(spec: dict) -> dict:
    """Clean up a spec after generation or merge.
    - Fix url_template overlap with base_url
    - Dedup path params using url_template as source of truth
    - Dedup filters_in_query
    - Derive missing filters from operator syntax (fallback)
    """
    if not spec:
        return spec

    # 0. Fix url_template / base_url overlap
    #    If base_url ends with a path like /v2/ and url_template starts with /v2/,
    #    the LLM duplicated the prefix. Strip it from the template.
    base_url = spec.get("base_url", "").rstrip("/")
    template = spec.get("url_template", "") or ""
    if base_url and template:
        from urllib.parse import urlparse
        base_path = urlparse(base_url).path.rstrip("/")  # e.g., "/v2"
        
        # Check if template starts with the base path (the overlap)
        if base_path and template.startswith(base_path):
            old_template = template
            template = template[len(base_path):]  # strip the overlapping prefix
            if not template.startswith("/"):
                template = "/" + template
            spec["url_template"] = template
            print(f"  Post-process: fixed url_template overlap with base_url")
            print(f"    {old_template} → {template}")

    # 1. Dedup path params: only keep those referenced in url_template
    template = spec.get("url_template", "") or ""
    if template:
        before = len(spec["parameters"])
        spec["parameters"] = [
            p for p in spec["parameters"]
            if p.get("location") != "path" or p["name"] in template
        ]
        removed = before - len(spec["parameters"])
        if removed:
            print(f"  Post-process: removed {removed} duplicate path params not in url_template")

    # 2. Dedup filters_in_query
    qs = spec.get("query_syntax")
    if qs:
        filters = qs.get("filters_in_query", [])
        if filters:
            deduped = list(dict.fromkeys(filters))  # preserves order
            if len(deduped) < len(filters):
                print(f"  Post-process: deduped filters_in_query from {len(filters)} to {len(deduped)}")
            qs["filters_in_query"] = deduped

        # 3. If filters_in_query is empty but operators exist, derive from syntax
        if not qs.get("filters_in_query") and qs.get("operators"):
            derived = []
            for op in qs["operators"]:
                syntax = op.get("syntax", "")
                if ":" in syntax:
                    prefix = syntax.split(":")[0]
                    if prefix not in derived:
                        derived.append(prefix)
            if derived:
                qs["filters_in_query"] = derived
                print(f"  Post-process: derived filters_in_query from operators: {derived}")

    return spec


# =============================================================================
# MULTI-PAGE SPEC GENERATION (main entry point for multi-page APIs)
# =============================================================================

def generate_api_spec_from_urls(urls: list[str], model: str = "qwen3:30b") -> dict:
    """Generate an API spec from multiple documentation pages.
    Each page is processed independently, then results are merged.
    """

    if len(urls) == 1:
        doc_text = fetch_documentation(urls[0])
        spec = generate_api_spec_decomposed(doc_text, model)
        return post_process_spec(spec) if spec else None

    partial_specs = []

    for url in urls:
        page_name = url.split("/")[-1][:50] if "/" in url else url[:50]
        print(f"\n{'='*60}")
        print(f"Processing: {page_name}")
        print(f"{'='*60}")

        try:
            doc_text = fetch_documentation(url)
            print(f"  Fetched {len(doc_text)} chars", flush=True)

            spec = generate_api_spec_decomposed(doc_text, model)
            if spec:
                partial_specs.append(spec)
                path_params = [p["name"] for p in spec.get("parameters", []) if p.get("location") == "path"]
                print(f"  Path params found: {path_params}")
            else:
                print(f"  Failed to generate spec for this page")
        except Exception as e:
            print(f"  Error processing {url}: {e}")

    if not partial_specs:
        print("  No specs generated from any page")
        return None

    # Merge
    print(f"\n{'='*60}")
    print(f"MERGING {len(partial_specs)} PARTIAL SPECS")
    print(f"{'='*60}")

    merged = merge_specs(partial_specs)

    # Post-process
    merged = post_process_spec(merged)

    if merged:
        path_params = [p["name"] for p in merged.get("parameters", []) if p.get("location") == "path"]
        print(f"  Name: {merged['name']}")
        print(f"  Base URL: {merged['base_url']}")
        print(f"  Total parameters: {len(merged['parameters'])}")
        print(f"  Path parameters: {path_params}")
        print(f"  URL template: {merged.get('url_template')}")

        qs = merged.get("query_syntax")
        if qs:
            print(f"  Operators: {len(qs.get('operators', []))}")
            print(f"  Filters: {qs.get('filters_in_query', [])}")

    return merged


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single page:  python generate_api_spec.py <doc_url> [output.json]")
        print("  Multi-page:   python generate_api_spec.py <url1> <url2> ... [output.json]")
        sys.exit(1)

    # Separate URLs from output file
    args = sys.argv[1:]
    urls = [a for a in args if a.startswith("http")]
    output_file = [a for a in args if not a.startswith("http")]
    output_file = output_file[0] if output_file else "api_spec_auto.json"

    print("=" * 60)
    print("API SPEC GENERATION")
    print(f"  Pages: {len(urls)}")
    print(f"  Output: {output_file}")
    print("=" * 60)

    if len(urls) == 1:
        doc_text = fetch_documentation(urls[0])
        spec = generate_api_spec_decomposed(doc_text)
        spec = post_process_spec(spec)
    else:
        spec = generate_api_spec_from_urls(urls)

    if spec:
        with open(output_file, "w") as f:
            json.dump(spec, f, indent=2)
        print(f"\n✓ Saved to {output_file}")
    else:
        print("\n✗ Failed to generate spec.")
