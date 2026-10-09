"""
Internal text dataset search using TF-IDF cosine similarity over a small
static JSON corpus, loaded once into memory at startup.

Future enhancement: swap this for embeddings + a vector store (e.g. FAISS)
once the dataset grows. `DatasetRetriever.search` is kept as a plain
query-in/ranked-chunks-out interface so callers wouldn't need to change.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Fact:
    id: int
    title: str
    text: str

    @classmethod
    def from_dict(cls, data: dict) -> "Fact":
        if not data.get("text") or not data["text"].strip():
            raise ValueError("Fact text cannot be empty")
        return cls(id=data["id"], title=data.get("title", ""), text=data["text"].strip())


@dataclass
class SearchResult:
    fact: Fact
    score: float


class DatasetRetriever:
    def __init__(self, data_path: str, threshold: float = 0.1):
        self.data_path = Path(data_path)
        self.threshold = threshold
        self.facts: List[Fact] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.fact_vectors = None
        self._load_data()

    def _load_data(self) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")

        with open(self.data_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Dataset must be a list of facts")
        if len(data) < 5:
            raise ValueError("Dataset must contain at least 5 facts")

        self.facts = [Fact.from_dict(item) for item in data]

        documents = [f"{fact.title} {fact.text}" for fact in self.facts]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.fact_vectors = self.vectorizer.fit_transform(documents)

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        if not self.vectorizer or self.fact_vectors is None:
            raise RuntimeError("Dataset not initialized")
        if not query or not query.strip():
            return []

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.fact_vectors)[0]

        indexed_scores = [(i, s) for i, s in enumerate(similarities) if s >= self.threshold]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return [SearchResult(fact=self.facts[i], score=s) for i, s in indexed_scores[:top_k]]
