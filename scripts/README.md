
---
## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `LANGUAGE_NAME` | `Bambara` | Target language name (appears in prompts). |
| `BATCH_SIZE` | `20` | Conversations per API call. |
| `CONCURRENT_BATCHES` | `20` | Parallel batch processors. |
| `MAX_WORKERS` | `20` | Async worker semaphore (same as above). |
| `RATE_LIMITER` | `100` | Requests per minute (Gemini Tier 3). |
| `MAX_RETRIES` | `5` | Exponential back-off retries. |
| `TIMEOUT` | `600.0` | General request timeout (seconds). |
| `BATCH_TIMEOUT` | `1200` | Per-batch timeout (20 min). |
| `MODEL_NAME` | `gemini-2.5-pro` | Model identifier. |
| `BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` | OpenAI-compatible endpoint. |
| `MAX_TOKENS` | `2000000` | Upper bound for response length. |
| `TEMPERATURE` | `0.1` | Low temperature → deterministic, rule-based output. |
| `TOP_P` | `0.9` | Nucleus sampling. |
| `DATASETS_FOLDER` | `datasets` | Folder with source JSONL files. |

All values can be overridden **without touching code**.

---