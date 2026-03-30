# DataEngineered

DataEngineered is a data discovery and extraction system. It takes a natural language research query, discovers relevant data across multiple APIs, extracts structured data using LLMs, and produces analysis-ready tabular datasets.

## What It Does

You ask a question like:

```
"Find any indication of threats Trump has made towards Iran in 2025-2026,
 and give me Iran's GDP, population, and trade data"
```

The system has the following features:
1. **Query Decomposition**: Breaks the user's query into sub-tasks and routes each to the right data source
2. **Schema Creation**: Generates a data schema according to the requirements implied by the user's query
3. **API Querying**: Queries the identified data sources' APIs automatically 
4. **Data Extraction**: Extracts the appropriate data values, e.g.
   - Article links → fetches HTML → LLM extracts structured data per schema
   - Structured data → parses JSON directly into tables
5. **Outputs** one CSV per data source, plus pipeline metadata for observability of the entire process

## Architecture

```
User query
  → Query Decomposition (LLM splits query, matches to registered data sources)
  → For each sub-task:
      → Load API spec from source registry
      → Query Analysis (LLM extracts entities, relationships, time range)
      → API Call Construction (LLM builds URL from spec)
      → Self-Correction Loop (retry on failure, LLM fixes URLs)
      → Response Type Detection (article_links vs structured_data)
      → BRANCH:
          article_links  → Schema Generation → LLM Review → Fetch → Extract → SQLite → CSV
          structured_data → JSON flatten → CSV
  → Merge same-source outputs (wide-format pivot)
  → Output: CSVs + pipeline_metadata.json
```

## Key Design Decisions

- **Source registry with 4-step validation**: When registering a new API, the system fetches documentation, generates a spec via focused LLM calls, validates the spec structure, and executes a test API call — all before accepting the source.

- **Query decomposition and task feasibility assessment**: The system uses an LLM to break a user query down into separate sub-tasks. Based on the data sources present in the source registry, the system decides if the sub-task can be feasibly carried out using a data source inside the source registry, or whether no appropriate data source exists to fulfill the sub-task.

- **Dynamic schema generation with LLM review**: The extraction schema (what fields to extract from articles) is generated from the query by one LLM call, then reviewed by a second LLM call that strips over-restrictive language (e.g., "exact text only", "must be explicitly stated"). This two-pass approach reduces extraction yield variance caused by overly narrow field descriptions.

- **Self-correcting API calls**: If a constructed URL returns an error, the LLM analyzes the failure and rebuilds the URL. Rate limiting is handled separately from real failures.

## Project Structure

```
├── orchestrator.py          # Entry point — decompose → execute → merge → output
├── decompose_query.py       # LLM splits query into sub-tasks matched to sources
├── source_registry.py       # Register and validate new API data sources
├── generate_api_spec.py     # Generate API specs from documentation pages
├── api_query.py             # Construct and execute API calls with self-correction
├── queries_to_sqlite.py     # Article fetch → LLM extraction → SQLite → CSV export
├── utils.py                 # Shared utilities 
└── source_registry/         # Stored source configs and API specs
    ├── specs/               # Generated API specifications (JSON)
    └── *.json               # Source metadata
```

## Registered Data Sources

| Source | Type | Pipeline | Status |
|--------|------|----------|--------|
| **GDELT** | Global news/event archive | article_links → LLM extraction | Working |
| **World Bank** | Economic indicators API | structured_data → direct parse | Working |

Currently, adding a new source can be done through the interactive registration CLI (see "Register a new data source" section below):

```bash
python source_registry.py register
```

## Usage

### Run a query

```bash
python orchestrator.py "Your research question here" --max 20
```

`--max` controls how many web articles get successfully fetched and extracted (for tasks that involve retrieving webpages). Structured data sources return all records regardless.

### Register a new data source

```bash
python source_registry.py register
# Follow prompts: name, description, API documentation URLs
# System validates automatically (fetch docs → generate spec → test call)
```

**Important**: Only publicly available API sources that do not require authentication are supported.

### Test query decomposition

To test and see that the system is running properly on your system, enter the following:

```bash
python decompose_query.py test
```

## Test Results

The pipeline has been tested across diverse query types:

| Query | Sources Used | Result |
|-------|-------------|--------|
| Trump threats to Iran + Iran economic data | GDELT + World Bank | 5-17 extractions + 3 indicators merged to wide format |
| Tesla stock news + US unemployment | GDELT + World Bank | 9 extractions + 66 unemployment records |
| Weather forecast + cinema listings | None (correctly rejected) | Graceful failure — reported as not feasible |
| AI regulation by world leaders | GDELT (limited) | Query returned irrelevant articles (see Known Limitations) |
| Multi-country GDP + population | World Bank | Data returned but pivot collapsed countries (see Known Limitations) |

## Setup

### Requirements

- Python 3.12+
- Ollama ([Ollama](https://ollama.ai)). The Qwen3 30B A3B (qwen3:30b) model from Ollama is recommended as the testing and development of this system has been done around this model so far, although you are free to use any model you choose.

### Installation

```bash
git clone https://github.com/danielyjtan/dataengineered.git
cd dataengineered

pip install -r requirements.txt

# Install and start Ollama
ollama pull qwen3:30b
```

### Configuration

The LLM model and endpoint are configured at the top of each script. By default:
- **Model**: `qwen3:30b` (via Ollama)
- **Endpoint**: `http://localhost:11434/api/chat`

To use a different model, update the `OLLAMA_MODEL` variable in `queries_to_sqlite.py`, `api_query.py`, and `generate_api_spec.py`.

No API keys are required for the currently registered sources (GDELT and World Bank are free public APIs).

## Known Limitations

### Extraction yield variance
The LLM generates slightly different extraction schemas each run. One run may extract 17 items from 20 articles; the next may extract 5 from the same articles. The Layer 2 schema review mitigates but doesn't eliminate this nondeterminism.

### Topic-based queries without named entities
The query analysis step extracts named entities (people, countries, organizations) to build API search terms. Queries like "What have world leaders said about AI regulation" contain no named entities — only a topic. The system falls back to generic search terms, returning irrelevant articles. A proper fix requires expanding the query analysis to extract topic keywords alongside named entities, following an intent-detection and slot-filling approach.

### Multi-value path parameters
The system cannot handle a certain class of queries which support multiple values for the same path parameter (e.g., World Bank's `country/BR;IN;ZA/indicator/...`). The API call builder doesn't know this syntax and falls back to `country=all`, returning data for all countries instead of just the requested ones.

### Single-entity pivot assumption
The wide-format merge pivots on `date` only. Multi-entity queries (e.g., GDP for three countries) collapse into one row per date, silently dropping data. The fix requires threading API spec metadata (path parameter definitions) into the merge function to determine composite groupby dimensions.

### Article fetch failures
Approximately 40% of news article URLs are unreachable due to paywalls, bot detection, or dead links. The pipeline skips these and continues until it reaches the target count of successful fetches.

### GET-only API support
The system supports REST APIs with GET requests. POST-body APIs, GraphQL, and APIs requiring authentication are not currently supported.

## Roadmap

- [ ] Query understanding redesign: expand entity extraction to include topic keywords and concept terms, add slot normalization for mapping extracted terms to API-specific codes (e.g., country names → ISO codes, indicator names → World Bank codes)
- [ ] Multi-value path parameter support
- [ ] Spec-aware pivot with composite groupby dimensions
- [ ] Centralized config file (model name, endpoint, timeouts)
- [ ] Consolidate LLM call functions into shared module
- [ ] Refactor prompts into template files
- [ ] Fetch failure diagnostics (log HTTP status codes)
- [ ] OpenAPI/Swagger auto-detection during source registration
- [ ] Pagination support (offset, cursor, link-header patterns)
- [ ] User feedback loop for schema refinement
- [ ] User-defined schemas: accept field names from the user instead of relying on LLM-generated schemas
- [ ] Interactive execution: pause at key decision points (decomposition, schema review) for user approval

## License

MIT
