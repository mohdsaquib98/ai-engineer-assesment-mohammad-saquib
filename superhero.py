"""Thin adapter over superheroapi.com."""
from dataclasses import dataclass
from typing import Dict, Optional

import httpx


@dataclass
class SuperheroResult:
    found: bool
    hero: Optional[Dict] = None
    error: Optional[str] = None


class SuperheroAdapter:
    def __init__(self, token: str, timeout: float = 5.0, max_failures: int = 5):
        self.token = token
        self.base_url = "https://superheroapi.com/api"
        self.timeout = timeout
        self.max_failures = max_failures
        self.failure_count = 0

    def search(self, name: str) -> SuperheroResult:
        if not name or not name.strip():
            return SuperheroResult(found=False, error="Name cannot be empty")

        if self.failure_count >= self.max_failures:
            return SuperheroResult(
                found=False,
                error="Superhero API temporarily unavailable (too many recent failures)",
            )

        url = f"{self.base_url}/{self.token}/search/{name.strip()}"

        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()

            if data.get("response") == "error":
                self.failure_count += 1
                return SuperheroResult(found=False, error=data.get("error", "Hero not found"))

            results = data.get("results", [])
            if not results:
                return SuperheroResult(found=False, error="No hero found")

            exact_match = next(
                (h for h in results if h.get("name", "").lower() == name.strip().lower()),
                None,
            )
            hero = exact_match or results[0]

            self.failure_count = 0
            return SuperheroResult(found=True, hero=hero)

        except httpx.TimeoutException:
            self.failure_count += 1
            return SuperheroResult(found=False, error="Request timed out")
        except httpx.HTTPStatusError as e:
            self.failure_count += 1
            return SuperheroResult(found=False, error=f"HTTP error: {e.response.status_code}")
        except Exception as e:
            self.failure_count += 1
            return SuperheroResult(found=False, error=f"Unexpected error: {e}")

    def reset_failure_count(self) -> None:
        self.failure_count = 0
