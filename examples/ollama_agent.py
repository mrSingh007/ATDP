from __future__ import annotations

import os
from uuid import uuid4

import httpx


BASE_URL = os.getenv("ATDP_BASE_URL", "http://localhost:8080/v1").rstrip("/")
MODEL = os.getenv("ATDP_MODEL", "llama3.2:1b")
PROMPT = os.getenv("ATDP_AGENT_PROMPT", "Explain ATDP in two practical sentences.")


def count_words(text: str) -> dict[str, int]:
    words = [word for word in text.split() if word.strip()]
    return {"word_count": len(words), "character_count": len(text)}


def post_event(client: httpx.Client, session_id: str, event_type: str, **payload) -> dict:
    response = client.post(
        f"{BASE_URL}/atdp/events",
        headers={"X-ATDP-Session-ID": session_id},
        json={
            "session_id": session_id,
            "type": event_type,
            **payload,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    session_id = str(uuid4())
    with httpx.Client(timeout=120) as client:
        post_event(
            client,
            session_id,
            "session.start",
            observation={"agent": "examples/ollama_agent.py"},
            metadata={"model": MODEL},
        )

        tool_args = {"text": PROMPT}
        tool_call = post_event(
            client,
            session_id,
            "tool.call",
            decision_context={
                "tool_name": "count_words",
                "tool_version": "0.1.0",
                "permission_scope": "local",
            },
            action={"name": "count_words", "arguments": tool_args},
            replay={"mode": "deterministic", "reason": "Pure local function."},
        )
        tool_result = count_words(PROMPT)
        post_event(
            client,
            session_id,
            "tool.result",
            parent_event_id=tool_call["id"],
            observation={"tool_call_event_id": tool_call["id"]},
            outcome={"result": tool_result},
            replay={"mode": "deterministic", "reason": "Captured full tool input and output."},
        )

        chat_response = client.post(
            f"{BASE_URL}/chat/completions",
            headers={"X-ATDP-Session-ID": session_id},
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a concise test agent using an ATDP proxy.",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"{PROMPT}\n\n"
                            f"A local count_words tool returned: {tool_result}. "
                            "Include one sentence about why trajectory recording helps RL."
                        ),
                    },
                ],
                "temperature": 0.2,
                "metadata": {
                    "atdp_session_id": session_id,
                    "tenant_id": "local-test",
                    "prompt_template_fingerprint": "example-ollama-agent-v1",
                },
            },
        )
        chat_response.raise_for_status()
        body = chat_response.json()
        print("session_id:", session_id)
        print("response:")
        print(body["choices"][0]["message"]["content"])

        post_event(
            client,
            session_id,
            "session.end",
            outcome={"completed": True},
            reward={"value": 1.0, "label": "manual_smoke_success", "source": "example_agent"},
        )


if __name__ == "__main__":
    main()
