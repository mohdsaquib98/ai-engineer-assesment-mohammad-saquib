# Superhero + Science Facts Chatbot

A FastAPI chatbot that answers questions about superheroes (via Superhero API) and a local dataset of science facts. Uses Groq's function-calling to route questions to the right source(s) and cites where information came from.


## About

The chatbot receives a question via `POST /ask` and:
1. **Routes intelligently**: Groq decides which tool(s) to call — `search_dataset` (science facts) or `search_superhero` (superhero API)
2. **Executes tools**: TF-IDF search over `data/facts.json` or HTTP call to superheroapi.com
3. **Synthesizes answer**: Groq combines tool results into a coherent response
4. **Tracks sources**: Returns the answer with a `tools_used` list showing what was consulted


## Sample Screenshots
1. Only Internal Data used.
<img width="1408" height="760" alt="Screenshot 2026-10-10 at 2 22 30 AM" src="https://github.com/user-attachments/assets/cf3e0c74-239e-4b2d-aaf6-8b81138828aa" />
2. Both Internal Data and Superhero API got used.
<img width="1419" height="766" alt="Screenshot 2026-10-10 at 2 21 54 AM" src="https://github.com/user-attachments/assets/c452551f-4cf9-42ac-8b57-a759687ea008" />
3. Only Superhero API got used.
<img width="1412" height="776" alt="Screenshot 2026-10-10 at 2 20 46 AM" src="https://github.com/user-attachments/assets/02421749-ae5e-4395-8731-5f6e4483f406" />


**Validation & error handling**:
- Questions must be non-empty and at least 2 characters
- Pre-LLM and pre-tool checks catch invalid inputs early
- Superhero API handles timeouts, HTTP errors, and "not found" with a circuit breaker
- Errors return structured responses (422 for validation, 502 for upstream failures, 500 for unexpected errors)

## Setup

```bash
pip install -r requirements.txt
```

Get API keys:
- Groq: https://console.groq.com/ (free)
- Superhero API: http://superheroapi.com/ (free, sign in with GitHub)

Create `.env` (see `.env.example`):
```bash
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-20b  # Groq model ID
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

Response:
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
