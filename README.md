# Superhero + Science Facts Chatbot

A FastAPI chatbot with a single `POST /ask` endpoint. It uses Groq's
function-calling to decide whether a question is about superheroes, a local
science-facts dataset, or both — calls the right source(s), and answers with
source attribution.

No frameworks (no LangChain/agents) — just the `groq` SDK directly, two
plain Python data sources, and FastAPI.

## How it works

```
POST /ask
  -> validate request (non-empty, length-bounded question)
  -> Groq call #1: model picks tool(s) - search_dataset / search_superhero / neither
  -> run whichever tool(s) were picked
       - search_dataset:   TF-IDF search over data/facts.json
       - search_superhero: call to superheroapi.com
  -> Groq call #2: model writes the final answer using the tool results
  -> response: { answer, tools_used }
```

If the model calls no tools, its own response is returned directly (handles
out-of-scope questions without wasting calls on the dataset/API).

## Setup

```bash
pip install -r requirements.txt
```

Get API keys:
- Groq: https://console.groq.com/ (free, email signup)
- Superhero API: http://superheroapi.com/ (free, sign in with GitHub)

Create `.env` (see `.env.example`):

```bash
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-8b-instant
SUPERHERO_API_TOKEN=your_superhero_token
```

## Run

```bash
uvicorn main:app --reload
```

API at `http://localhost:8000`, interactive docs at `/docs`.

## Example

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How fast does light travel?"}'
```

```json
{
  "answer": "Light travels at approximately 299,792 km/s in a vacuum... Source: local science facts dataset",
  "tools_used": ["search_dataset"]
}
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Covers the core logic: TF-IDF dataset search, the superhero API adapter
(success, not-found, timeout, HTTP errors, circuit breaker), and the
routing/synthesis logic in `groq_client.py` (dataset-only, superhero-only,
both, neither, and failure cases) with the Groq client mocked. No test hits
a real Groq or superheroapi.com endpoint.

## Validation & error handling

- Request validation: question must be non-empty, 2-1000 chars (`models.py`).
- Pre-LLM check: reject empty/oversized questions before calling Groq.
- Pre-tool check: reject empty/oversized arguments the model tries to pass
  to a tool.
- Post-tool check: tool output is capped/truncated before being sent back
  to the model.
- Superhero API: timeouts, HTTP errors, and "not found" are handled
  distinctly; a simple circuit breaker stops hammering the API after 5
  consecutive failures.
- Any unrecoverable failure (Groq unreachable, bad tool-call JSON, etc.)
  raises `LLMError` internally, mapped to a `502` with a safe message. A
  global exception handler catches anything unexpected as a `500`.

## Dataset

`data/facts.json` — ~25 short general-science facts (space, biology,
geology, physics), deliberately unrelated to superheroes so routing is
unambiguous.

## Decisions & tradeoffs

- **TF-IDF, not RAG/embeddings**: good enough for ~25 static facts; avoids
  an embeddings model/vector DB dependency. `DatasetRetriever.search` is a
  plain query-in/ranked-results-out interface, so it could be swapped for
  an embeddings-based implementation later without touching callers.
- **No agent framework**: two Groq chat-completions calls (route, then
  synthesize) using native function-calling. Simpler to read, debug, and
  run than wiring up an agent framework for two tools.
- **Source attribution, double-enforced**: the system prompt requires the
  model to state its source in the answer text, and `tools_used` is tracked
  independently in code from which tools actually ran - so attribution
  doesn't rely solely on the model remembering the instruction.
- **Superhero disambiguation**: exact case-insensitive name match preferred,
  otherwise first result - no clarification round-trip with the user.
- **Stateless**: each `/ask` call is single-turn, no conversation memory.
- **No auth/rate limiting**: out of scope for this exercise.
