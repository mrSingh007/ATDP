# ATDP Data Proxy

An open-source, ATDP-inspired data proxy for capturing learning-ready agent trajectories.

ATDP Data Proxy sits between an agent application and an OpenAI-compatible LLM endpoint. It forwards chat requests to the upstream model provider and records structured trajectory events that can later be used for debugging, replay, memory updates, evaluation datasets, SFT/DPO/RL-style training datasets, and governed self-evolution workflows.

This project is inspired by the Agent Trajectory Data Protocol ideas described in the AReaL2.0 / self-evolving agents research direction. It is not a full online RL training system.

## What This Project Does

- Proxies OpenAI-compatible `chat.completions` requests.
- Captures non-streaming and streaming LLM calls.
- Records typed trajectory events: `llm.request`, `llm.response`, `llm.error`, `tool.call`, `tool.result`, `mcp.tool.call`, `mcp.tool.result`, `retrieval.query`, `retrieval.result`, `memory.read`, `memory.write`, `human.feedback`, `guardrail.decision`, `replay.result`, and session lifecycle events.
- Stores trajectories in SQLite and JSONL.
- Maintains event IDs, session IDs, trajectory IDs, parent links, event hashes, previous-event hashes, replay metadata, governance metadata, and redaction/training eligibility metadata.
- Supports late reward attachment without rewriting the original event.
- Exports ATDP Step JSONL and minimal AReaL-style JSONL.
- Computes simple evolution signals such as missing rewards, tool failures, retrieval gaps, replay gaps, candidate training slices, and LNL-style missing-domain-knowledge recommendations.
- Redacts common sensitive data before storage/export.
- Offers a small Python boundary SDK for tools, MCP, retrieval, memory, browser/file actions, feedback, guardrails, replay results, and final outcomes.

## How To Test ATDP Locally
Check ollama example in [How2Test.md](./How2Test.md)

## What This Project Is Not

- It is not a complete AReaL2.0 online RL system.
- It does not train model weights.
- It does not run train workers, inference worker pools, or weight synchronization.
- It does not autonomously modify memory, prompts, tools, or models.

The proxy captures the learning substrate. A separate evolution worker or control plane can consume the trajectories and decide whether to update memory, RAG indexes, prompts, tool schemas, guardrails, model datasets, or nothing.

## Why This Is Useful

Most agent logs are useful for debugging but incomplete for learning. A self-improving agent needs to know not only the prompt and completion, but also:

- what the agent observed,
- what tools, memory, retrieval, and MCP resources were available,
- which action was taken,
- what outcome happened,
- what human or programmatic feedback arrived later,
- whether the data is safe and eligible for training,
- whether the event can be replayed,
- which intervention is likely appropriate.

ATDP Data Proxy gives agent builders and researchers a small reference implementation for this learning-data layer. It can help teams build:

- replay tests for failed agent behaviors,
- memory extraction workflows,
- RAG quality datasets,
- tool-failure and schema-review loops,
- human-feedback and reward datasets,
- SFT/DPO/RL data exports,
- control-plane prototypes for self-evolving agents.

## Architecture

```text
Agent / App
  |
  | OpenAI-compatible chat API
  | Boundary events for tools, MCP, RAG, memory, feedback, replay
  v
ATDP Data Proxy
  |-- OpenAI-compatible provider adapter
  |-- ATDP event recorder
  |-- Boundary capture API / Python SDK
  |-- SQLite + JSONL trajectory storage
  |-- Late reward API
  |-- ATDP Step and AReaL-minimal exporters
  |-- Evolution signal API
  |-- Optional OpenTelemetry spans
  v
External model provider
  - host Ollama
  - OpenAI
  - Groq
  - OpenAI-compatible gateway such as LiteLLM
```

## Provider Support

The current provider adapter supports OpenAI-compatible APIs.

| Provider | Status | Notes |
| --- | --- | --- |
| Host Ollama | Supported | Run Ollama on your machine and point the proxy to `http://localhost:11434/v1` locally or `http://host.docker.internal:11434/v1` from Docker. |
| OpenAI | Supported in principle | Use `ATDP_UPSTREAM_BASE_URL=https://api.openai.com/v1` and `ATDP_UPSTREAM_API_KEY`. |
| Groq | Supported in principle | Use `ATDP_UPSTREAM_BASE_URL=https://api.groq.com/openai/v1` and `ATDP_UPSTREAM_API_KEY`. |
| Other OpenAI-compatible APIs | Supported in principle | Works when the provider implements `/v1/chat/completions` and `/v1/models` compatibly. |
| AWS Bedrock | Via gateway today | Use an OpenAI-compatible gateway such as LiteLLM in front of Bedrock. A native Bedrock adapter is a good future contribution. |
| Anthropic/Gemini native APIs | Via gateway today | Use an OpenAI-compatible gateway or add a native provider adapter. |

Provider-specific limitations still apply. For example, many local or OpenAI-compatible providers do not return token IDs or logprobs, so exports mark token data as unavailable.

## Quick Start With Docker

Docker runs only the ATDP proxy and optional helper services. It does not run Ollama in a container.

### Host Ollama

Start Ollama on your machine first:

```bash
ollama pull llama3.2:1b
ollama serve
```

In another terminal:

```bash
docker compose up --build atdp-proxy
```

The default Compose upstream is `http://host.docker.internal:11434/v1`, which points from the proxy container to Ollama on the host.

Run a request through the proxy:

```bash
SESSION_ID="local-$(date +%s)"

curl -sS http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-ATDP-Session-ID: $SESSION_ID" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [{"role":"user","content":"Explain ATDP in one sentence."}],
    "temperature": 0.2
  }' | jq

curl -sS "http://localhost:8080/v1/atdp/sessions/$SESSION_ID/trajectory" | jq
```

Run the example agent:

```bash
ATDP_MODEL=llama3.2:1b \
docker compose --profile agent run --rm example-agent
```

### OpenAI

```bash
ATDP_UPSTREAM_BASE_URL=https://api.openai.com/v1 \
ATDP_UPSTREAM_API_KEY="$OPENAI_API_KEY" \
docker compose up --build atdp-proxy
```

Then call the proxy with an OpenAI model:

```bash
SESSION_ID="openai-$(date +%s)"

curl -sS http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-ATDP-Session-ID: $SESSION_ID" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role":"user","content":"Explain why trajectory data helps agent evaluation."}]
  }' | jq
```

### Groq

```bash
ATDP_UPSTREAM_BASE_URL=https://api.groq.com/openai/v1 \
ATDP_UPSTREAM_API_KEY="$GROQ_API_KEY" \
docker compose up --build atdp-proxy
```

Then use any Groq chat model available to your account.

### AWS Bedrock Through LiteLLM

Run LiteLLM or another OpenAI-compatible gateway against Bedrock, then point ATDP Data Proxy to that gateway:

```bash
ATDP_UPSTREAM_BASE_URL=http://host.docker.internal:4000/v1 \
ATDP_UPSTREAM_API_KEY="$LITELLM_API_KEY" \
docker compose up --build atdp-proxy
```

Native Bedrock support is not implemented yet.

## Run Locally Without Docker

With `uv`:

```bash
uv sync --extra dev

ATDP_UPSTREAM_BASE_URL=http://localhost:11434/v1 \
ATDP_DATA_DIR=./data \
uv run uvicorn atdp_proxy.app:create_app --factory --reload --port 8080
```

With `venv` and `pip`:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"

ATDP_UPSTREAM_BASE_URL=http://localhost:11434/v1 \
ATDP_DATA_DIR=./data \
uvicorn atdp_proxy.app:create_app --factory --reload --port 8080
```

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `ATDP_PROVIDER` | `openai-compatible` | Provider adapter name. |
| `ATDP_UPSTREAM_BASE_URL` | `http://localhost:11434/v1` locally, `http://host.docker.internal:11434/v1` in Compose | Upstream OpenAI-compatible API base URL. |
| `ATDP_UPSTREAM_API_KEY` | empty | API key sent to upstream as `Authorization: Bearer ...`. |
| `ATDP_FORWARD_AUTHORIZATION` | `false` | Forward caller `Authorization` header when no upstream API key is configured. |
| `ATDP_DATA_DIR` | `./data` | Directory for SQLite and JSONL storage. |
| `ATDP_SQLITE_PATH` | unset | Optional explicit SQLite path. |
| `ATDP_JSONL_PATH` | unset | Optional explicit JSONL path. |
| `ATDP_REQUEST_TIMEOUT_SECONDS` | `120` | Upstream request timeout. |
| `ATDP_DEFAULT_TENANT_ID` | `default` | Default tenant ID for governance metadata. |
| `ATDP_REDACTION_MODE` | `basic` | `basic` or `off`. |
| `ATDP_OTEL_ENABLED` | `false` | Enable optional OpenTelemetry spans. |
| `ATDP_OTEL_EXPORTER_OTLP_ENDPOINT` | unset | OTLP HTTP endpoint. |

## API

OpenAI-compatible proxy:

- `POST /v1/chat/completions`
- `POST /chat/completions`
- `GET /v1/models`

ATDP APIs:

- `POST /v1/atdp/events` records explicit agent events.
- `POST /v1/atdp/boundary-events` records stable execution boundary events.
- `POST /v1/atdp/events/{event_id}/reward` attaches late reward or critique to an immutable event.
- `GET /v1/atdp/events/{event_id}` returns one event with latest reward overlay.
- `GET /v1/atdp/sessions` lists sessions.
- `GET /v1/atdp/sessions/{session_id}/trajectory` returns events for one session.
- `GET /v1/atdp/export/jsonl` exports raw ATDP events as JSONL.
- `GET /v1/atdp/datasets/steps` exports materialized ATDP Step JSONL.
- `GET /v1/atdp/datasets/export?format=atdp_step_jsonl` exports ATDP Step JSONL.
- `GET /v1/atdp/datasets/export?format=areal_minimal` exports minimal AReaL-oriented JSONL.
- `GET /v1/atdp/evolution/signals` returns read-only learning signals and recommendations.

For chat calls, pass a stable session ID in `X-ATDP-Session-ID`. If omitted, the proxy creates one and returns it in `X-ATDP-Session-ID`.

## Boundary Capture Example

```bash
TOOL_CALL_ID="$(
  curl -sS http://localhost:8080/v1/atdp/boundary-events \
    -H "Content-Type: application/json" \
    -H "X-ATDP-Session-ID: session-123" \
    -d '{
      "boundary": "tool",
      "phase": "start",
      "name": "calculator",
      "version": "1.0",
      "input": {"expr": "2+2"},
      "replay": {"mode": "deterministic", "reason": "Pure local function."}
    }' | jq -r .id
)"

curl -sS http://localhost:8080/v1/atdp/boundary-events \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"session-123\",
    \"boundary\": \"tool\",
    \"phase\": \"end\",
    \"name\": \"calculator\",
    \"parent_event_id\": \"$TOOL_CALL_ID\",
    \"input\": {\"expr\":\"2+2\"},
    \"output\": {\"result\":4},
    \"replay\": {\"mode\":\"deterministic\",\"reason\":\"Input and output captured.\"}
  }" | jq
```

## Python Boundary SDK

```python
from atdp_proxy.ingress import ATDPClient

client = ATDPClient(base_url="http://localhost:8080/v1", session_id="session-123")

client.capture_retrieval(
    "docs",
    query={"q": "ATDP"},
    results={"documents": [{"id": "doc-1", "score": 0.91}]},
)

client.capture_memory_write(
    "company_glossary",
    input={"key": "LNL", "value": "Lorem Norway Limited, a company"},
    output={"approved": True},
)

client.capture_replay_result(
    replay_of_session_id="session-123",
    replay_passed=True,
    output={"answer_contains": "Lorem Norway Limited"},
)
```

## LNL Learning-Data Example

The proxy can represent this common self-evolution loop:

1. User asks: `What is LNL?`
2. The model gives a wrong answer.
3. User says: `Wrong. LNL means Lorem Norway Limited, a company.`
4. The app records `human.feedback`.
5. A negative reward is attached to the wrong `llm.response`.
6. An evolution worker can classify this as missing domain knowledge.
7. The app or worker records a `memory.write`.
8. A replay attempt records `replay.result`.
9. Dataset exports and evolution signals preserve the full causal path.

The proxy does not update memory by itself. It captures the data needed for a controlled worker to do that safely.

## Dataset Exports

ATDP Step JSONL:

```bash
curl -sS "http://localhost:8080/v1/atdp/datasets/steps?session_id=session-123"
```

Minimal AReaL-style JSONL:

```bash
curl -sS "http://localhost:8080/v1/atdp/datasets/export?format=areal_minimal&session_id=session-123"
```

Training-ineligible events and reward updates are excluded by default. Use `include_ineligible=true` for debugging exports.

## Testing

```bash
uv run pytest -q
uv run python -m compileall -q atdp_proxy tests examples
docker compose config >/tmp/atdp-compose.yml
docker compose -f docker-compose.host-ollama.yml --profile agent config >/tmp/atdp-host-compose.yml
```

See [how-test.md](how-test.md) for a full local manual verification flow.

## Optional Observability

Enable span export:

```bash
ATDP_OTEL_ENABLED=true \
ATDP_OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318/v1/traces \
docker compose --profile observability up --build
```

Jaeger UI is available at `http://localhost:16686`.

OpenTelemetry is derived from the ATDP learning layer. It is not the canonical trajectory store.

## Current Limitations

- Only the OpenAI-compatible provider adapter is implemented.
- Native Bedrock, Anthropic, and Gemini adapters are not implemented yet.
- SQLite/JSONL are appropriate for local development and MVP deployments, not high-volume multi-node production.
- There is no authn/authz layer yet.
- There is no built-in trainer, online RL loop, or model registry.
- Replay is recorded as metadata and events; there is not yet a full replay executor.

## Roadmap Ideas

- Native provider adapters for Bedrock, Anthropic, Gemini, and Vertex AI.
- Provider registry and adapter contract tests.
- MCP framework interceptors.
- LangChain/LangGraph/OpenAI Agents SDK/CrewAI integration wrappers.
- Replay executor and replay reports.
- Dataset manifests and export provenance hashes.
- Training job/control-plane APIs.
- Production storage backend.
- Auth, tenant isolation, retention, and policy enforcement.

## License

MIT License with explicit attribution notice. You may use, copy, modify, and distribute the project for free, but you must preserve the copyright notice and license text so the original project and contributors receive credit. See [LICENSE](LICENSE).
