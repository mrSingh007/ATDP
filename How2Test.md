# How To Test ATDP Locally

This guide tests the ATDP Learning Infrastructure MVP end to end: proxying, streaming, boundary capture, rewards, redaction, dataset generation, evolution signals, and persistence.

Run commands from the repo root:

```bash
cd /Users/harvindersingh/Desktop/temp/atdp
```

## 1. Prerequisites

Required local tools:

- Docker Desktop
- `uv`
- `curl`
- `jq`
- macOS Ollama running locally

Check them:

```bash
docker --version
docker compose version
uv --version
jq --version
curl http://localhost:11434/api/version
```

Pull the test model:

```bash
ollama pull gemma4:e2b
ollama list | grep 'gemma4:e2b'
```

## 2. Automated Verification

Run the unit/integration tests:

```bash
uv run pytest -q
```

Expected:

```text
12 passed
```

Compile-check the package:

```bash
uv run python -m compileall -q atdp_proxy tests examples
```

Validate Docker Compose configs:

```bash
docker compose config >/tmp/atdp-compose.yml
docker compose -f docker-compose.host-ollama.yml --profile agent config >/tmp/atdp-host-compose.yml
```

## 3. Start Proxy With macOS Ollama

This runs the ATDP proxy in Docker while using Ollama installed on your Mac:

```bash
docker compose -f docker-compose.host-ollama.yml up --build atdp-proxy
```

Leave this terminal running.

In a second terminal:

```bash
curl -sS http://localhost:8080/healthz | jq
```

Expected fields:

```json
{
  "status": "ok",
  "provider": "openai-compatible",
  "redaction_mode": "basic"
}
```

## 4. Test Non-Streaming LLM Capture

```bash
SESSION_ID="atdp-lorem-$(date +%s)"

curl -sS -D /tmp/atdp-chat.headers -o /tmp/atdp-chat.json \
  http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-ATDP-Session-ID: $SESSION_ID" \
  -d '{
    "model": "gemma4:e2b",
    "messages": [
      {
        "role": "user",
        "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Summarize this in one short sentence."
      }
    ],
    "temperature": 0.2,
    "metadata": {
      "governance": {
        "tenant_id": "local-test",
        "training_eligible": true
      }
    }
  }'

cat /tmp/atdp-chat.json | jq
```

Capture event IDs from response headers:

```bash
REQUEST_EVENT_ID="$(awk -F': ' 'tolower($1)=="x-atdp-request-event-id" {gsub("\r","",$2); print $2}' /tmp/atdp-chat.headers)"
RESPONSE_EVENT_ID="$(awk -F': ' 'tolower($1)=="x-atdp-response-event-id" {gsub("\r","",$2); print $2}' /tmp/atdp-chat.headers)"

echo "$SESSION_ID"
echo "$REQUEST_EVENT_ID"
echo "$RESPONSE_EVENT_ID"
```

Inspect the trajectory:

```bash
curl -sS "http://localhost:8080/v1/atdp/sessions/$SESSION_ID/trajectory" | jq
```

Expected:

- `event_count` is at least `2`
- event types include `llm.request` and `llm.response`
- events contain `event_hash`
- response event `previous_event_hash` matches request event hash
- governance `tenant_id` is `local-test`

Quick check:

```bash
curl -sS "http://localhost:8080/v1/atdp/sessions/$SESSION_ID/trajectory" \
  | jq -r '.events[].type'
```

Expected:

```text
llm.request
llm.response
```

## 5. Test Streaming LLM Capture

```bash
STREAM_SESSION_ID="atdp-stream-$(date +%s)"

curl -N -sS http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-ATDP-Session-ID: $STREAM_SESSION_ID" \
  -d '{
    "model": "gemma4:e2b",
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": "Stream a one sentence answer about why ATDP creates training samples."
      }
    ],
    "temperature": 0.2
  }'
```

Inspect recorded stream trajectory:

```bash
curl -sS "http://localhost:8080/v1/atdp/sessions/$STREAM_SESSION_ID/trajectory" | jq
```

Expected:

- event types include `llm.request` and `llm.response`
- response event outcome contains `stream_capture`
- response event contains reconstructed assistant content

## 6. Test Boundary Capture

Record a tool call/result pair:

```bash
TOOL_CALL_ID="$(
  curl -sS http://localhost:8080/v1/atdp/boundary-events \
    -H "Content-Type: application/json" \
    -H "X-ATDP-Session-ID: $SESSION_ID" \
    -d '{
      "boundary": "tool",
      "phase": "start",
      "name": "local_word_count",
      "version": "0.1.0",
      "input": {"text": "Lorem ipsum dolor sit amet."},
      "replay": {"mode": "deterministic", "reason": "Captured full input."}
    }' | jq -r .id
)"

TOOL_RESULT_ID="$(
  curl -sS http://localhost:8080/v1/atdp/boundary-events \
    -H "Content-Type: application/json" \
    -d "{
      \"session_id\": \"$SESSION_ID\",
      \"boundary\": \"tool\",
      \"phase\": \"end\",
      \"name\": \"local_word_count\",
      \"version\": \"0.1.0\",
      \"parent_event_id\": \"$TOOL_CALL_ID\",
      \"input\": {\"text\": \"Lorem ipsum dolor sit amet.\"},
      \"output\": {\"word_count\": 5},
      \"replay\": {\"mode\": \"deterministic\", \"reason\": \"Captured full input and output.\"}
    }" | jq -r .id
)"

echo "$TOOL_CALL_ID"
echo "$TOOL_RESULT_ID"
```

Record a retrieval query/result pair with replay snapshot refs:

```bash
RETRIEVAL_QUERY_ID="$(
  curl -sS http://localhost:8080/v1/atdp/boundary-events \
    -H "Content-Type: application/json" \
    -H "X-ATDP-Session-ID: $SESSION_ID" \
    -d '{
      "boundary": "retrieval",
      "phase": "start",
      "name": "local_docs",
      "version": "0.1.0",
      "input": {"query": "ATDP learning infrastructure"},
      "replay": {
        "mode": "approximate",
        "reason": "Retriever can be replayed against the captured corpus snapshot.",
        "snapshot_refs": {"corpus": "local-docs@v1", "chunk_ids": ["chunk-1"]}
      }
    }' | jq -r .id
)"

RETRIEVAL_RESULT_ID="$(
  curl -sS http://localhost:8080/v1/atdp/boundary-events \
    -H "Content-Type: application/json" \
    -d "{
      \"session_id\": \"$SESSION_ID\",
      \"boundary\": \"retrieval\",
      \"phase\": \"end\",
      \"name\": \"local_docs\",
      \"version\": \"0.1.0\",
      \"parent_event_id\": \"$RETRIEVAL_QUERY_ID\",
      \"input\": {\"query\": \"ATDP learning infrastructure\"},
      \"output\": {
        \"documents\": [
          {\"id\": \"chunk-1\", \"text\": \"ATDP records execution boundaries as RL-grade trajectory data.\"}
        ]
      },
      \"replay\": {
        \"mode\": \"approximate\",
        \"reason\": \"Result depends on the captured corpus snapshot.\",
        \"snapshot_refs\": {\"corpus\": \"local-docs@v1\", \"chunk_ids\": [\"chunk-1\"]}
      }
    }" | jq -r .id
)"

echo "$RETRIEVAL_QUERY_ID"
echo "$RETRIEVAL_RESULT_ID"
```

Record other boundary types:

```bash
for payload in \
  '{"boundary":"memory","phase":"start","name":"short_term","output":{"items":[]}}' \
  '{"boundary":"memory","phase":"end","name":"short_term","input":{"item":"ATDP records training samples."}}' \
  '{"boundary":"file","phase":"event","name":"read","input":{"path":"README.md"},"output":{"ok":true}}' \
  '{"boundary":"browser","phase":"event","name":"click","input":{"selector":"#submit"},"output":{"ok":true}}' \
  '{"boundary":"human_feedback","phase":"event","name":"approval","output":{"approved":true}}' \
  '{"boundary":"guardrail","phase":"event","name":"pii_filter","input":{"text":"hello"},"output":{"allowed":true}}' \
  '{"boundary":"replay","phase":"event","name":"manual_replay","replay_of_session_id":"example-original","replay_passed":true,"output":{"passed":true}}' \
  '{"boundary":"final_outcome","phase":"event","name":"task_complete","output":{"success":true}}'
do
  curl -sS http://localhost:8080/v1/atdp/boundary-events \
    -H "Content-Type: application/json" \
    -H "X-ATDP-Session-ID: $SESSION_ID" \
    -d "$payload" | jq -r '.type + " " + .id'
done
```

Check event types:

```bash
curl -sS "http://localhost:8080/v1/atdp/sessions/$SESSION_ID/trajectory" \
  | jq -r '.events[].type'
```

Expected includes:

```text
llm.request
llm.response
tool.call
tool.result
retrieval.query
retrieval.result
memory.read
memory.write
file.action
browser.action
human.feedback
guardrail.decision
replay.result
session.end
```

Record an MCP tool call/result pair:

```bash
MCP_CALL_ID="$(
  curl -sS http://localhost:8080/v1/atdp/boundary-events \
    -H "Content-Type: application/json" \
    -H "X-ATDP-Session-ID: $SESSION_ID" \
    -d '{
      "boundary": "mcp",
      "phase": "start",
      "name": "filesystem.read_file",
      "server_name": "filesystem",
      "server_version": "2026.7.0",
      "input_schema_version": "schema-v1",
      "correlation_id": "mcp-local-1",
      "input": {"path": "README.md"},
      "resources": [{"uri": "file://README.md"}],
      "replay": {"mode": "deterministic", "reason": "Input and result are captured."}
    }' | jq -r .id
)"

curl -sS http://localhost:8080/v1/atdp/boundary-events \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION_ID\",
    \"boundary\": \"mcp\",
    \"phase\": \"end\",
    \"name\": \"filesystem.read_file\",
    \"parent_event_id\": \"$MCP_CALL_ID\",
    \"server_name\": \"filesystem\",
    \"correlation_id\": \"mcp-local-1\",
    \"input\": {\"path\": \"README.md\"},
    \"output\": {\"content\":\"# ATDP Data Proxy\"},
    \"side_effects_happened\": false,
    \"replay\": {\"mode\":\"deterministic\",\"reason\":\"Input and result are captured.\"}
  }" | jq
```

Confirm paired MCP call/result became one training step:

```bash
curl -sS "http://localhost:8080/v1/atdp/datasets/steps?session_id=$SESSION_ID" \
  | jq -r 'select(.metadata.event_types == ["mcp.tool.call","mcp.tool.result"]) | .decision_context.server_name'
```

Expected:

```text
filesystem
```

## 7. Test Late Reward Attachment

Attach a delayed reward to the LLM response event:

```bash
curl -sS "http://localhost:8080/v1/atdp/events/$RESPONSE_EVENT_ID/reward" \
  -H "Content-Type: application/json" \
  -d '{
    "value": 1.0,
    "label": "accepted",
    "critique": "The answer was concise enough for this local smoke test.",
    "source": "manual_test",
    "evaluator_version": "manual-v1",
    "training_eligible": true
  }' | jq
```

Expected:

- original event `reward` remains unchanged or null
- `latest_reward` is present
- `reward_updates[0].update_hash` is present

Check:

```bash
curl -sS "http://localhost:8080/v1/atdp/events/$RESPONSE_EVENT_ID" \
  | jq '{id, reward, latest_reward, reward_updates}'
```

## 8. Test ATDP Step Dataset Export

Export materialized ATDP steps:

```bash
curl -sS "http://localhost:8080/v1/atdp/datasets/steps?session_id=$SESSION_ID"
```

Pretty-print the first step:

```bash
curl -sS "http://localhost:8080/v1/atdp/datasets/steps?session_id=$SESSION_ID" \
  | head -n 1 \
  | jq
```

Expected fields:

- `step_id`
- `session_id`
- `trajectory_id`
- `source_event_ids`
- `observation`
- `decision_context`
- `action`
- `outcome`
- `reward`
- `governance`
- `replay`
- `metadata.token_data_status`

Repeat the export and confirm `step_id` values are stable across runs:

```bash
curl -sS "http://localhost:8080/v1/atdp/datasets/steps?session_id=$SESSION_ID" \
  | jq -r .step_id > /tmp/atdp-step-ids-1.txt

curl -sS "http://localhost:8080/v1/atdp/datasets/steps?session_id=$SESSION_ID" \
  | jq -r .step_id > /tmp/atdp-step-ids-2.txt

diff /tmp/atdp-step-ids-1.txt /tmp/atdp-step-ids-2.txt && echo "OK: stable step IDs"
```

Confirm paired tool call/result became one training step:

```bash
curl -sS "http://localhost:8080/v1/atdp/datasets/steps?session_id=$SESSION_ID" \
  | jq -r 'select(.metadata.event_types == ["tool.call","tool.result"]) | .outcome.output.word_count'
```

Expected:

```text
5
```

Confirm paired retrieval query/result became one training step with replay snapshot metadata:

```bash
curl -sS "http://localhost:8080/v1/atdp/datasets/steps?session_id=$SESSION_ID" \
  | jq -r 'select(.metadata.event_types == ["retrieval.query","retrieval.result"]) | .replay.snapshot_refs.corpus'
```

Expected:

```text
local-docs@v1
```

## 9. Test Minimal AReaL Export

```bash
curl -sS "http://localhost:8080/v1/atdp/datasets/export?format=areal_minimal&session_id=$SESSION_ID" \
  | head -n 1 \
  | jq
```

Expected fields:

- `interaction_id`
- `parent_id`
- `messages`
- `output_message_list`
- `reward`
- `token_ids`
- `logprobs`
- `metadata.token_data_status`

For Ollama/OpenAI-compatible mode, `token_ids` and `logprobs` may be `null`, and `metadata.token_data_status` should be `unavailable`.

## 10. Test Training Eligibility Filtering

Create an ineligible session:

```bash
INELIGIBLE_SESSION_ID="atdp-ineligible-$(date +%s)"

curl -sS http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-ATDP-Session-ID: $INELIGIBLE_SESSION_ID" \
  -d '{
    "model": "gemma4:e2b",
    "messages": [{"role":"user","content":"Say hello once."}],
    "metadata": {
      "governance": {
        "tenant_id": "local-test",
        "training_eligible": false
      }
    }
  }' | jq
```

Default dataset export should exclude it:

```bash
curl -sS "http://localhost:8080/v1/atdp/datasets/steps?session_id=$INELIGIBLE_SESSION_ID"
```

Expected: empty output.

Explicit override should include it:

```bash
curl -sS "http://localhost:8080/v1/atdp/datasets/steps?session_id=$INELIGIBLE_SESSION_ID&include_ineligible=true" \
  | head -n 1 \
  | jq '.governance.training_eligible'
```

Expected:

```text
false
```

Attach a reward that is itself ineligible and confirm it is excluded by default:

```bash
INELIGIBLE_REWARD_EVENT_ID="$(
  curl -sS http://localhost:8080/v1/atdp/events \
    -H "Content-Type: application/json" \
    -d '{
      "session_id": "atdp-ineligible-reward",
      "type": "tool.result",
      "action": {"name": "debug_lookup"},
      "outcome": {"result": "debug-only"}
    }' | jq -r .id
)"

curl -sS "http://localhost:8080/v1/atdp/events/$INELIGIBLE_REWARD_EVENT_ID/reward" \
  -H "Content-Type: application/json" \
  -d '{
    "value": 1.0,
    "label": "debug_only",
    "source": "manual_test",
    "training_eligible": false
  }' | jq

curl -sS "http://localhost:8080/v1/atdp/datasets/steps?session_id=atdp-ineligible-reward" \
  | jq '.reward'
```

Expected:

```text
null
```

## 11. Test Basic Redaction

```bash
REDACTION_SESSION_ID="atdp-redaction-$(date +%s)"

curl -sS http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-ATDP-Session-ID: $REDACTION_SESSION_ID" \
  -d '{
    "model": "gemma4:e2b",
    "messages": [
      {
        "role": "user",
        "content": "My email is user@example.com and api_key=secret123. Reply with OK."
      }
    ]
  }' | jq
```

Confirm stored trajectory is redacted:

```bash
curl -sS "http://localhost:8080/v1/atdp/sessions/$REDACTION_SESSION_ID/trajectory" \
  | tee /tmp/atdp-redaction.json \
  | jq -r '.events[0].observation.messages[0].content'

grep -q 'user@example.com' /tmp/atdp-redaction.json && echo "FAILED: email leaked" || echo "OK: email redacted"
grep -q 'secret123' /tmp/atdp-redaction.json && echo "FAILED: secret leaked" || echo "OK: secret redacted"
```

Expected:

```text
OK: email redacted
OK: secret redacted
```

Confirm bearer tokens, cookies, and secret URL parameters are redacted:

```bash
SECRET_REDACTION_SESSION_ID="atdp-secret-redaction-$(date +%s)"

curl -sS http://localhost:8080/v1/atdp/events \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SECRET_REDACTION_SESSION_ID\",
    \"type\": \"file.action\",
    \"observation\": {
      \"text\": \"Bearer abcdefghijklmnopqrstuvwxyz Cookie: sessionid=super-secret https://example.test/path?api_key=super-secret&x=1\"
    }
  }" | jq

curl -sS "http://localhost:8080/v1/atdp/sessions/$SECRET_REDACTION_SESSION_ID/trajectory" \
  | tee /tmp/atdp-secret-redaction.json \
  | jq -r '.events[0].observation.text'

grep -q 'abcdefghijklmnopqrstuvwxyz' /tmp/atdp-secret-redaction.json && echo "FAILED: bearer leaked" || echo "OK: bearer redacted"
grep -q 'sessionid=super-secret' /tmp/atdp-secret-redaction.json && echo "FAILED: cookie leaked" || echo "OK: cookie redacted"
grep -q 'api_key=super-secret' /tmp/atdp-secret-redaction.json && echo "FAILED: URL secret leaked" || echo "OK: URL secret redacted"
```

## 12. Test Evolution Signals

```bash
curl -sS "http://localhost:8080/v1/atdp/evolution/signals?session_id=$SESSION_ID" | jq
```

Expected fields:

- `event_count`
- `step_count`
- `rewarded_steps`
- `missing_reward_count`
- `reward_coverage`
- `error_count`
- `tool_failure_clusters`
- `retrieval_failure_clusters`
- `memory_miss_patterns`
- `ineligible_ratio`
- `replayability`
- `replay_gap_count`
- `candidate_training_slices`
- `recommendations`

Typical recommendations during manual testing:

- `needs_reward`
- `candidate_training_slice`
- `replay_gap`

## 13. Test LNL Feedback And Replay Readiness

This validates the expected self-evolution learning-data flow. The proxy should capture the data and recommend a memory update/replay; it is not expected to autonomously learn by itself.

```bash
LNL_SESSION_ID="atdp-lnl-$(date +%s)"

curl -sS -D /tmp/atdp-lnl.headers -o /tmp/atdp-lnl.json \
  http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-ATDP-Session-ID: $LNL_SESSION_ID" \
  -d '{
    "model": "gemma4:e2b",
    "messages": [{"role":"user","content":"What is LNL?"}],
    "temperature": 0.2
  }'

LNL_RESPONSE_EVENT_ID="$(awk -F': ' 'tolower($1)=="x-atdp-response-event-id" {gsub("\r","",$2); print $2}' /tmp/atdp-lnl.headers)"

curl -sS "http://localhost:8080/v1/atdp/events/$LNL_RESPONSE_EVENT_ID/reward" \
  -H "Content-Type: application/json" \
  -d '{
    "value": -1.0,
    "label": "wrong_answer",
    "source": "human",
    "critique": "Model missed a private company glossary entry."
  }' | jq

FEEDBACK_EVENT_ID="$(
  curl -sS http://localhost:8080/v1/atdp/boundary-events \
    -H "Content-Type: application/json" \
    -H "X-ATDP-Session-ID: $LNL_SESSION_ID" \
    -d "{
      \"boundary\": \"human_feedback\",
      \"name\": \"user_correction\",
      \"parent_event_id\": \"$LNL_RESPONSE_EVENT_ID\",
      \"output\": {\"text\":\"Wrong. LNL means Lorem Norway Limited, a company.\"}
    }" | jq -r .id
)"

curl -sS http://localhost:8080/v1/atdp/boundary-events \
  -H "Content-Type: application/json" \
  -H "X-ATDP-Session-ID: $LNL_SESSION_ID" \
  -d "{
    \"boundary\": \"memory\",
    \"phase\": \"end\",
    \"name\": \"company_glossary\",
    \"parent_event_id\": \"$FEEDBACK_EVENT_ID\",
    \"input\": {\"key\":\"LNL\",\"value\":\"Lorem Norway Limited, a company\"},
    \"output\": {\"approved\":true},
    \"replay\": {\"mode\":\"deterministic\",\"reason\":\"Memory key/value and approval are captured.\"}
  }" | jq

curl -sS http://localhost:8080/v1/atdp/boundary-events \
  -H "Content-Type: application/json" \
  -H "X-ATDP-Session-ID: $LNL_SESSION_ID" \
  -d "{
    \"boundary\": \"replay\",
    \"name\": \"lnl_replay\",
    \"replay_of_session_id\": \"$LNL_SESSION_ID\",
    \"replay_of_event_id\": \"$LNL_RESPONSE_EVENT_ID\",
    \"replay_passed\": true,
    \"output\": {\"answer_contains\":\"Lorem Norway Limited\"},
    \"replay\": {\"mode\":\"deterministic\",\"reason\":\"Replay evaluator result is captured.\"}
  }" | jq

curl -sS "http://localhost:8080/v1/atdp/evolution/signals?session_id=$LNL_SESSION_ID" \
  | jq '{memory_miss_patterns, recommendations}'
```

Expected recommendations include:

```text
missing_domain_knowledge
memory_write
replay_needed
```

## 14. Test Raw JSONL And SQLite Persistence

Check files:

```bash
ls -lh data
```

Expected:

```text
atdp.sqlite3
atdp_events.jsonl
```

Export session JSONL:

```bash
curl -sS "http://localhost:8080/v1/atdp/export/jsonl?session_id=$SESSION_ID" | head
```

Check SQLite session/event counts from the host:

```bash
sqlite3 data/atdp.sqlite3 "select count(*) from sessions;"
sqlite3 data/atdp.sqlite3 "select type, count(*) from events group by type order by type;"
```

If `sqlite3` is not installed locally, inspect through Python:

```bash
python3 - <<'PY'
import sqlite3
conn = sqlite3.connect("data/atdp.sqlite3")
print(conn.execute("select count(*) from sessions").fetchone())
print(conn.execute("select type, count(*) from events group by type order by type").fetchall())
PY
```

## 15. Optional: Test Example Agent

```bash
ATDP_MODEL=gemma4:e2b \
docker compose -f docker-compose.host-ollama.yml --profile agent run --rm ollama-agent
```

Then inspect sessions:

```bash
curl -sS http://localhost:8080/v1/atdp/sessions | jq
```

## 16. Stop Services

```bash
docker compose -f docker-compose.host-ollama.yml down
```

To remove local test data:

```bash
rm -rf data
```

## Pass Criteria

The full local test passes when:

- Automated tests show `12 passed`.
- Proxy health returns `status: ok`.
- Non-streaming and streaming chat both create `llm.request` and `llm.response`.
- Boundary events cover MCP, tool, retrieval, memory, file, browser, human feedback, guardrail, replay, and final outcome.
- Late rewards appear as immutable reward updates.
- ATDP Step JSONL contains materialized training steps with stable `step_id` values.
- Minimal AReaL export returns stable JSONL records.
- Ineligible events and ineligible reward updates are excluded by default from dataset export.
- Redaction removes obvious emails, bearer tokens, cookies, URL secrets, and key-value secrets from stored/exported data.
- Evolution signals return learning recommendations, including LNL-style missing-domain-knowledge recommendations.
- `data/atdp.sqlite3` and `data/atdp_events.jsonl` are created.
