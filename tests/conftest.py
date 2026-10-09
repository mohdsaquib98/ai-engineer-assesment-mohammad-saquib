import os

# Config requires these at import time; tests don't hit the real APIs, so
# dummy values are set here before any test module imports `config`.
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("SUPERHERO_API_TOKEN", "test-superhero-token")
