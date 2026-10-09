# Superhero + Science Facts Chatbot

A FastAPI chatbot that answers questions about superheroes (via Superhero API) and a local dataset of science facts. Uses Groq's function-calling to route questions to the right source(s) and always cites where information came from.

## How it works

The chatbot receives a question via `POST /ask` and:
1. Asks Groq to decide which tool(s) to call: `search_dataset` (science facts) or `search_superhero` (superhero API)
2. Executes the selected tool(s):
   - `search_dataset`: TF-IDF search over `data/facts.json`
   - `search_superhero`: HTTP call to superheroapi.com
3. Sends the tool results back to Groq to synthesize a final answer
4. Returns the answer with a `tools_used` list showing which sources were consulted

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
