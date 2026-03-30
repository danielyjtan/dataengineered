"""
Script 02: Intelligent API query system with strategy layer and self-correction.

Components:
  1. Query Analyzer — LLM extracts entities and intent from user query
  2. Strategy Selector — rules pick search strategy based on query type + API spec capabilities
  3. API Call Builder — LLM constructs the API call using chosen strategy
  4. Self-Correction — heuristics evaluate results, switch strategy if needed
"""

import json
import time
import re
import requests


# =============================================================================
# LOAD API SPEC
# =============================================================================

def load_spec(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def spec_has_capability(spec: dict, capability: str) -> bool:
    """Check if the API spec documents a particular capability."""
    qs = spec.get("query_syntax")
    if not qs:
        return False
    
    # Check old flat format
    if capability == "proximity_search":
        if qs.get("proximity_search"):
            return True
    elif capability == "boolean_or":
        if qs.get("boolean_or"):
            return True
    elif capability == "phrase_search":
        if qs.get("phrase_search"):
            return True
    elif capability == "language_filter":
        filters = qs.get("filters_in_query", [])
        if any("sourcelang" in f or "language" in f.lower() for f in filters):
            return True
    
    # Check new operators array format
    operators = qs.get("operators", [])
    for op in operators:
        op_name = op.get("name", "").lower()
        if capability == "proximity_search" and ("near" in op_name or "proxim" in op_name):
            return True
        elif capability == "boolean_or" and "or" in op_name:
            return True
        elif capability == "phrase_search" and ("phrase" in op_name or "exact" in op_name):
            return True
    
    return False


# =============================================================================
# COMPONENT 1: Query Analyzer
# =============================================================================

QUERY_ANALYSIS_PROMPT = """Analyze this user query and extract structured information about what they're searching for.

USER QUERY: "{user_query}"

Return ONLY a JSON object with:
{{
    "entities": ["list of key entities — people, countries, organizations mentioned"],
    "relationship": "the relationship or action between entities (e.g. 'threatening', 'trading with', 'visiting'), or null if no relationship",
    "query_type": "one of: relationship_between_entities, topic_search, entity_search, broad_exploration",
    "time_range": {{
        "start": "YYYY-MM-DD or null if not specified",
        "end": "YYYY-MM-DD or null if not specified",
        "description": "e.g. 'February 2026' or 'last week' or null"
    }},
    "language": "language the user is writing in (e.g. 'English')",
    "metric_requested": "the specific data point or measurement the user is asking about (e.g. 'GDP', 'population', 'military expenditure'), or null if not specified"
}}

Definitions:
- relationship_between_entities: user wants to find how two or more entities interact (e.g. "Trump threatening Iran")
- topic_search: user wants articles about a specific topic (e.g. "articles about climate change")
- entity_search: user wants articles about a single entity (e.g. "news about Apple")
- broad_exploration: user wants a general overview (e.g. "what's happening in the Middle East")

Return ONLY the JSON object, no other text.
"""


def analyze_query(user_query: str, model: str = "qwen3:30b") -> dict:
    """Extract entities, intent, and structure from user query."""
    prompt = QUERY_ANALYSIS_PROMPT.format(user_query=user_query)
    
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1}
        },
        timeout=300,
    )
    
    raw = response.json()["message"]["content"]
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return None


# =============================================================================
# COMPONENT 2: Strategy Selector
# =============================================================================

STRATEGIES = {
    "proximity": {
        "description": "Use proximity/near operator with key entities",
        "best_for": "relationship_between_entities",
        "priority": 1,
    },
    "phrase": {
        "description": "Use exact phrase search",
        "best_for": "topic_search",
        "priority": 2,
    },
    "and": {
        "description": "Use implicit AND with key terms",
        "best_for": "broad_exploration",
        "priority": 3,
    },
    "or": {
        "description": "Use OR for synonyms/related terms",
        "best_for": "entity_search",
        "priority": 4,
    },
}

def select_strategy(analysis: dict, spec: dict, failed_strategies: list = None) -> dict:
    """Pick the best search strategy based on entity count and API capabilities.
    
    Simplified from original:
    - Ignores query_type (was unreliable)
    - Only uses proximity and AND (phrase and OR caused problems)
    - Strategy based on entity count only
    - Instructions always use entities, never relationship words
    """
    
    if failed_strategies is None:
        failed_strategies = []
    
    entities = analysis.get("entities", [])
    time_range = analysis.get("time_range", {})
    metric = analysis.get("metric_requested")
    language = analysis.get("language", "English")
    
    # Simple rule: 2+ entities → proximity (if available) → AND
    #              1 entity   → AND
    strategy_order = []
    
    if len(entities) >= 2:
        if spec_has_capability(spec, "proximity_search"):
            strategy_order.append("proximity")
        strategy_order.append("and")
    else:
        strategy_order.append("and")
    
    # Filter out failed strategies
    available = [s for s in strategy_order if s not in failed_strategies]
    if not available:
        return None
    
    chosen = available[0]
    
    # Build instructions
    instructions = {
        "strategy_name": chosen,
        "strategy_description": STRATEGIES.get(chosen, {}).get("description", chosen),
    }
    
    if chosen == "proximity":
        instructions["search_entities"] = entities[:2]
        instructions["instruction"] = (
            f"Use the proximity/near operator with these entities: "
            f"'{entities[0]}' and '{entities[1]}'. "
            f"Do NOT include verbs or relationship words inside the near operator — "
            f"only the two entity names."
        )
    elif chosen == "and":
        instructions["instruction"] = (
            f"Search using all of these entities as keywords: {', '.join(entities)}. "
            f"All entities must appear in the results."
        )
    
    # Add metric instruction if specified
    if metric:
        instructions["metric_instruction"] = (
            f"The user is specifically asking about: {metric}. "
            f"Use the most appropriate parameter/indicator for this metric, not a default."
        )
    
    # Add language filter instruction
    if spec_has_capability(spec, "language_filter"):
        instructions["language_instruction"] = (
            f"The user is querying in {language}. Add the appropriate language filter "
            f"to ensure results are in {language}. "
            f"Use the exact syntax shown in the spec's operators."
        )
    
    # Add date instructions if user specified a time range
    if time_range.get("start") or time_range.get("end") or time_range.get("description"):
        instructions["date_instruction"] = (
            f"The user wants results from: {time_range.get('description') or ''} "
            f"(start: {time_range.get('start') or 'not specified'}, "
            f"end: {time_range.get('end') or 'not specified'}). "
            f"Use the appropriate date parameters from the spec."
        )
    
    return instructions


# def select_strategy(analysis: dict, spec: dict, failed_strategies: list = None) -> dict:
#     """Pick the best search strategy based on query analysis and API capabilities."""
    
#     if failed_strategies is None:
#         failed_strategies = []
    
#     query_type = analysis.get("query_type", "broad_exploration")
#     entities = analysis.get("entities", [])
    
#     # Build ordered list of strategies to try
#     strategy_order = []
    
#     if query_type == "relationship_between_entities" and len(entities) >= 2:
#         if spec_has_capability(spec, "proximity_search"):
#             strategy_order.append("proximity")
#         strategy_order.append("and")
#         strategy_order.append("phrase")
    
#     elif query_type == "topic_search":
#         if spec_has_capability(spec, "phrase_search"):
#             strategy_order.append("phrase")
#         strategy_order.append("and")
    
#     elif query_type == "entity_search":
#         strategy_order.append("and")
#         if spec_has_capability(spec, "boolean_or"):
#             strategy_order.append("or")
    
#     else:  # broad_exploration
#         strategy_order.append("and")
    
#     # Filter out already-failed strategies
#     available = [s for s in strategy_order if s not in failed_strategies]
    
#     if not available:
#         return None  # all strategies exhausted
    
#     chosen = available[0]
    
#     # Build strategy instructions for the API call builder
#     instructions = {
#         "strategy_name": chosen,
#         "strategy_description": STRATEGIES[chosen]["description"],
#     }
    
#     if chosen == "proximity":
#         # Pick the two most important entities for near search
#         instructions["search_entities"] = entities[:2]
#         instructions["instruction"] = (
#             f"Use the proximity/near operator with these two entities: "
#             f"'{entities[0]}' and '{entities[1]}'. "
#             f"Do NOT include verbs or relationship words inside the near operator — only the two entity names."
#         )
    
#     elif chosen == "phrase":
#         topic = " ".join(entities)
#         relationship = analysis.get("relationship", "")
#         if relationship:
#             instructions["instruction"] = (
#                 f"Use exact phrase search for the entities: {topic}. "
#                 f"The relationship is '{relationship}' — include the entities in the search, "
#                 f"not the relationship word."
#             )
#         else:
#             instructions["instruction"] = f"Use exact phrase search for: {topic}"
    
#     elif chosen == "and":
#         instructions["instruction"] = f"Search using implicit AND with terms: {' '.join(entities)}"
    
#     elif chosen == "or":
#         instructions["instruction"] = f"Search using OR between these terms: {', '.join(entities)}"

#     # Add metric instruction if specified
#     metric = analysis.get("metric_requested")
#     if metric:
#         instructions["metric_instruction"] = (
#             f"The user is specifically asking about: {metric}. "
#             f"Use the most appropriate parameter/indicator for this metric, not a default."
#         )
    
#     # Add language filter instruction if available
#     if spec_has_capability(spec, "language_filter"):
#         user_lang = analysis.get("language", "English")
#         instructions["language_instruction"] = (
#             f"The user is querying in {user_lang}. Add the appropriate language filter "
#             f"to ensure results are in {user_lang}."
#         )
    
#     # Add date instructions if user specified a time range
#     time_range = analysis.get("time_range", {})
#     if time_range.get("start") or time_range.get("end") or time_range.get("description"):
#         instructions["date_instruction"] = (
#             f"The user wants results from: {time_range.get('description') or ''} "
#             f"(start: {time_range.get('start') or 'not specified'}, "
#             f"end: {time_range.get('end') or 'not specified'}). "
#             f"Use the appropriate date parameters from the spec."
#         )
    
#     return instructions


# =============================================================================
# COMPONENT 3: API Call Builder
# =============================================================================

BUILD_CALL_PROMPT = """You are an API call builder. Given an API specification and specific search instructions,
construct the correct API call.

API SPECIFICATION:
{spec_json}

SEARCH INSTRUCTIONS:
{instructions_json}

RULES:
1. Follow the search instructions exactly — they tell you which operator/strategy to use.
2. Follow the exact syntax patterns from the API spec.
3. Always include format=json for machine-readable output.
4. Maximize results — use the highest allowed value for max records.
5. For date parameters, use the EXACT format specified in the spec (count the digits carefully).
6. Include any language filters as instructed.
7. Do NOT URL-encode the query value — the HTTP library will handle encoding.
8. Pay attention to parameter LOCATION in the spec. Parameters with location "path" must be 
   embedded in the URL path, NOT as query string parameters. Parameters with location 
   "query_string" go after the ? as key=value pairs. Look at the examples in the spec and 
   the url_template to see the correct URL structure.

Return ONLY a JSON object with:
{{
    "url": "The complete URL ready to execute (with query values NOT URL-encoded)",
    "params_used": {{"parameter": "value"}},
    "reasoning": "Brief explanation"
}}

Return ONLY the JSON object, no other text.
"""


def build_api_call(spec: dict, instructions: dict, model: str = "qwen3:30b") -> dict:
    """Construct an API call using the spec and strategy instructions."""
    
    prompt = BUILD_CALL_PROMPT.format(
        spec_json=json.dumps(spec, indent=2),
        instructions_json=json.dumps(instructions, indent=2)
    )
    
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1}
        },
        timeout=600,
    )
    
    raw = response.json()["message"]["content"]
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return None


# =============================================================================
# COMPONENT 4: Self-Correction
# =============================================================================

def evaluate_results(response_status: int, response_text: str, data,
                     analysis: dict) -> dict:
    """Evaluate API results and determine if correction is needed."""
    
    evaluation = {
        "success": False,
        "issue": None,
        "error_message": None,
        "suggestion": None,
    }
    
    # Tier 1: HTTP errors
    if response_status == 403:
        evaluation["issue"] = "forbidden"
        evaluation["error_message"] = "403 Forbidden — URL path may be wrong"
        evaluation["suggestion"] = "Check base_url matches the spec's example URLs exactly"
        return evaluation
    
    if response_status == 429:
        evaluation["issue"] = "rate_limited"
        evaluation["error_message"] = "429 Rate Limited"
        evaluation["suggestion"] = "Wait longer and retry the same URL"
        return evaluation
    
    if response_status != 200:
        evaluation["issue"] = "http_error"
        evaluation["error_message"] = f"HTTP {response_status}: {response_text[:200]}"
        evaluation["suggestion"] = "Check URL construction against spec"
        return evaluation
    
    # Tier 1: Response isn't JSON
    if data is None:
        if response_text:
            evaluation["issue"] = "api_error"
            evaluation["error_message"] = response_text[:300]
            evaluation["suggestion"] = "Feed the error message back to the LLM to fix"
        else:
            evaluation["issue"] = "not_json"
            evaluation["error_message"] = "Response is not valid JSON"
            evaluation["suggestion"] = "Ensure format=json is included in the URL"
        return evaluation
    
    # Tier 2: Got data — check if it's non-empty
    # Handle both dict and list response formats
    has_data = False
    num_results = 0
    
    if isinstance(data, dict):
        # Look for any list of results in the response
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                has_data = True
                num_results = max(num_results, len(value))
    elif isinstance(data, list):
        # Some APIs return a list directly (e.g. World Bank returns [metadata, data])
        has_data = len(data) > 0
        for item in data:
            if isinstance(item, list):
                num_results = max(num_results, len(item))
            elif isinstance(item, dict):
                num_results += 1
    
    if not has_data:
        evaluation["issue"] = "no_results"
        evaluation["error_message"] = "Query returned empty results"
        evaluation["suggestion"] = "Try a broader search strategy"
        return evaluation
    
    # Tier 2: Check language (only for search APIs with article titles)
    user_lang = analysis.get("language", "English")
    if user_lang.lower() == "english" and isinstance(data, dict):
        articles = data.get("articles", [])
        if articles:
            non_ascii_titles = sum(
                1 for a in articles[:10]
                if any(ord(c) > 127 for c in a.get("title", ""))
            )
            if non_ascii_titles > 5:
                evaluation["issue"] = "wrong_language"
                evaluation["error_message"] = f"{non_ascii_titles}/10 titles appear to be non-English"
                evaluation["suggestion"] = "Add sourcelang:English to the query"
                return evaluation
    
    evaluation["success"] = True
    evaluation["num_results"] = num_results
    return evaluation


def clean_url(url: str) -> str:
    """Fix common LLM URL construction errors.
    
    Known issues:
      - Double-slash prefix duplication: /v2//v2/country → /v2/country
        (LLM concatenates base_url + template, doubling the prefix)
    
    Does NOT touch consecutive segments without // (e.g., GDELT's /doc/doc
    is legitimate and left alone).
    """
    import re
    from urllib.parse import urlparse, urlunparse
    
    parsed = urlparse(url)
    path = parsed.path
    
    # Fix double-slash duplicated prefixes: /segment//segment/ → /segment/
    # The // is the signal that the LLM accidentally doubled a path prefix
    changed = True
    while changed:
        new_path = re.sub(r'/([^/]+)//\1(?=/|$)', r'/\1', path)
        changed = (new_path != path)
        path = new_path
    
    # Fix any remaining stray double slashes
    while "//" in path:
        path = path.replace("//", "/")
    
    cleaned = urlunparse(parsed._replace(path=path))
    
    if cleaned != url:
        print(f"    URL cleaned: {url[:80]}...")
        print(f"            →  {cleaned[:80]}...")
    
    return cleaned


def execute_and_evaluate(url: str, analysis: dict) -> tuple:
    """Execute an API call and evaluate the results."""
    
    try:
        resp = requests.get(url, timeout=30)
        
        data = None
        if resp.status_code == 200:
            try:
                data = resp.json()
            except json.JSONDecodeError:
                # Try cleaning
                try:
                    cleaned = resp.text.replace('\n', ' ').replace('\r', '')
                    data = json.loads(cleaned)
                except:
                    pass
        
        evaluation = evaluate_results(resp.status_code, resp.text, data, analysis)
        return data, evaluation
    
    except Exception as e:
        return None, {
            "success": False,
            "issue": "connection_error",
            "error_message": str(e),
            "suggestion": "Check URL validity and network connection"
        }


def self_correct(spec: dict, instructions: dict, failed_url: str,
                 evaluation: dict, model: str = "qwen3:30b") -> dict:
    """Ask the LLM to fix a failed API call based on the error."""
    
    prompt = f"""You previously constructed an API call that failed. Fix it.

API SPECIFICATION:
{json.dumps(spec, indent=2)}

ORIGINAL INSTRUCTIONS:
{json.dumps(instructions, indent=2)}

URL THAT FAILED:
{failed_url}

ERROR:
{evaluation.get('error_message', 'Unknown error')}

SUGGESTED FIX:
{evaluation.get('suggestion', 'Review the URL against the spec')}

Generate a corrected API call. Return ONLY a JSON object with:
{{
    "url": "The corrected URL",
    "changes_made": "What you fixed"
}}

Return ONLY the JSON object, no other text.
"""
    
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1}
        },
        timeout=600,
    )
    
    raw = response.json()["message"]["content"]
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return None


# =============================================================================
# ORCHESTRATOR: Ties everything together
# =============================================================================

def _make_success_result(user_query, analysis, instructions, url, attempt, data, evaluation):
    """Build the success return dict and print preview."""
    num_results = evaluation.get("num_results", 0)
    print(f"    ✓ SUCCESS — {num_results} results")
    print(f"    Response type: {type(data).__name__}")
    if isinstance(data, dict):
        print(f"    Top-level keys: {list(data.keys())[:5]}")
    elif isinstance(data, list):
        print(f"    List with {len(data)} items")
        if len(data) > 1 and isinstance(data[1], list):
            print(f"    Data array: {len(data[1])} entries")
            for item in data[1][:3]:
                if isinstance(item, dict):
                    preview = {k: v for k, v in list(item.items())[:4]}
                    print(f"      {preview}")
    return {
        "query": user_query,
        "analysis": analysis,
        "strategy": instructions["strategy_name"],
        "url": url,
        "attempts": attempt + 1,
        "num_results": num_results,
        "data": data,
    }


def _wait_for_rate_limit(url: str, analysis: dict, max_retries: int = 5) -> tuple:
    """Retry the same URL with exponential backoff until rate limit clears.
    
    Returns (data, evaluation) from the first successful attempt,
    or the last failed evaluation if all retries exhausted.
    """
    backoff = 15
    for retry in range(max_retries):
        print(f"    Rate limited. Waiting {backoff}s... (retry {retry + 1}/{max_retries})")
        time.sleep(backoff)
        data, evaluation = execute_and_evaluate(url, analysis)
        if evaluation["success"]:
            return data, evaluation
        if evaluation["issue"] != "rate_limited":
            return data, evaluation  # Different error now, let caller handle
        backoff = min(backoff * 2, 120)  # Cap at 2 minutes
    return data, evaluation


def run_query(user_query: str, spec: dict, max_attempts: int = 3,
              model: str = "qwen3:30b") -> dict:
    """Full pipeline: analyze → strategize → build → execute → self-correct.
    
    Rate limiting is handled separately from attempts. Only real failures
    (bad URL, no results, wrong language) consume an attempt.
    """
    
    print(f"\n{'='*60}")
    print(f"QUERY: \"{user_query}\"")
    print(f"{'='*60}")
    
    # Step 1: Analyze the query
    print(f"\n  Step 1: Analyzing query...")
    start = time.time()
    analysis = analyze_query(user_query, model)
    print(f"    Done ({time.time() - start:.1f}s)")
    
    if not analysis:
        print("    ERROR: Failed to analyze query")
        return None
    
    print(f"    Entities: {analysis.get('entities')}")
    print(f"    Relationship: {analysis.get('relationship')}")
    print(f"    Query type: {analysis.get('query_type')}")
    print(f"    Time range: {analysis.get('time_range')}")
    print(f"    Language: {analysis.get('language')}")
    
    failed_strategies = []
    
    for attempt in range(max_attempts):
        print(f"\n  --- Attempt {attempt + 1}/{max_attempts} ---")
        
        # Step 2: Select strategy
        print(f"  Step 2: Selecting search strategy...")
        instructions = select_strategy(analysis, spec, failed_strategies)
        
        if not instructions:
            print("    All strategies exhausted. Giving up.")
            break
        
        print(f"    Strategy: {instructions['strategy_name']}")
        print(f"    Instruction: {instructions.get('instruction', '')[:80]}")
        
        # Step 3: Build API call
        print(f"  Step 3: Building API call...")
        start = time.time()
        api_call = build_api_call(spec, instructions, model)
        print(f"    Done ({time.time() - start:.1f}s)")
        
        if not api_call:
            print("    ERROR: Failed to build API call")
            failed_strategies.append(instructions["strategy_name"])
            continue
        
        url = api_call.get("url", "")
        url = url.replace('\\"', '"').replace("\\'", "'")
        url = clean_url(url)
        print(f"    URL: {url[:120]}...")
        print(f"    Reasoning: {api_call.get('reasoning', '')[:100]}")
        
        # Step 4: Execute and evaluate
        print(f"  Step 4: Executing...")
        data, evaluation = execute_and_evaluate(url, analysis)
        
        # Handle rate limiting separately — don't burn an attempt
        if not evaluation["success"] and evaluation["issue"] == "rate_limited":
            print(f"    ✗ Rate limited")
            data, evaluation = _wait_for_rate_limit(url, analysis)
        
        if evaluation["success"]:
            return _make_success_result(
                user_query, analysis, instructions, url, attempt, data, evaluation
            )
        
        # Real failure — decide how to correct
        print(f"    ✗ FAILED: {evaluation['issue']}")
        print(f"    Error: {evaluation.get('error_message', '')[:100]}")
        
        if evaluation["issue"] in ("api_error", "forbidden", "not_json"):
            # Ask LLM to fix the URL
            print(f"    Self-correcting...")
            start = time.time()
            correction = self_correct(spec, instructions, url, evaluation, model)
            print(f"    Done ({time.time() - start:.1f}s)")
            
            if correction:
                corrected_url = correction.get("url", "")
                corrected_url = clean_url(corrected_url)
                print(f"    Corrected URL: {corrected_url[:120]}...")
                print(f"    Changes: {correction.get('changes_made', '')[:100]}")
                
                time.sleep(6)
                data, evaluation = execute_and_evaluate(corrected_url, analysis)
                
                # Handle rate limiting on corrected URL too
                if not evaluation["success"] and evaluation["issue"] == "rate_limited":
                    data, evaluation = _wait_for_rate_limit(corrected_url, analysis)
                
                if evaluation["success"]:
                    return _make_success_result(
                        user_query, analysis, instructions, corrected_url, attempt, data, evaluation
                    )
                else:
                    print(f"    Corrected URL also failed: {evaluation['issue']}")
        
        elif evaluation["issue"] == "wrong_language":
            # Don't change strategy, just add language filter
            print(f"    Adding language filter and retrying...")
            instructions["language_instruction"] = (
                "CRITICAL: You MUST add sourcelang:English inside the query parameter. "
                "The previous attempt returned non-English results."
            )
            start = time.time()
            api_call = build_api_call(spec, instructions, model)
            print(f"    Rebuilt ({time.time() - start:.1f}s)")
            
            if api_call:
                url = api_call.get("url", "")
                url = url.replace('\\"', '"').replace("\\'", "'")
                url = clean_url(url)
                time.sleep(6)
                data, evaluation = execute_and_evaluate(url, analysis)
                
                if not evaluation["success"] and evaluation["issue"] == "rate_limited":
                    data, evaluation = _wait_for_rate_limit(url, analysis)
                
                if evaluation["success"]:
                    return _make_success_result(
                        user_query, analysis, instructions, url, attempt, data, evaluation
                    )
        
        elif evaluation["issue"] == "no_results":
            failed_strategies.append(instructions["strategy_name"])
            print(f"    Switching to next strategy...")
        
        # Rate limit between attempts
        time.sleep(6)
    
    print(f"\n  All attempts failed for: \"{user_query}\"")
    return None


