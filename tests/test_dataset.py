import json

import pytest

from dataset import DatasetRetriever, Fact, SearchResult

SAMPLE_FACTS = [
    {"id": 1, "title": "Light Speed", "text": "Light travels at 299,792 km/s"},
    {"id": 2, "title": "Mars", "text": "Mars has a thin atmosphere"},
    {"id": 3, "title": "Water", "text": "71% of Earth is water"},
    {"id": 4, "title": "DNA", "text": "Human DNA is 99.9% identical"},
    {"id": 5, "title": "Moon", "text": "The Moon formed 4.5 billion years ago"},
]


def _write_facts(tmp_path, facts):
    path = tmp_path / "facts.json"
    path.write_text(json.dumps(facts))
    return str(path)


def test_fact_from_dict():
    fact = Fact.from_dict({"id": 1, "title": "Test", "text": "Test content"})
    assert fact.id == 1
    assert fact.text == "Test content"


def test_fact_from_dict_empty_text():
    with pytest.raises(ValueError, match="Fact text cannot be empty"):
        Fact.from_dict({"id": 1, "title": "Test", "text": ""})


def test_retriever_initialization(tmp_path):
    retriever = DatasetRetriever(data_path=_write_facts(tmp_path, SAMPLE_FACTS))
    assert len(retriever.facts) == 5


def test_retriever_file_not_found():
    with pytest.raises(FileNotFoundError):
        DatasetRetriever(data_path="nonexistent.json")


def test_retriever_not_a_list(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ValueError, match="must be a list"):
        DatasetRetriever(data_path=str(path))


def test_retriever_too_few_facts(tmp_path):
    with pytest.raises(ValueError, match="at least 5 facts"):
        DatasetRetriever(data_path=_write_facts(tmp_path, SAMPLE_FACTS[:2]))


def test_search_relevant_query(tmp_path):
    retriever = DatasetRetriever(data_path=_write_facts(tmp_path, SAMPLE_FACTS))
    results = retriever.search("light speed")
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)
    assert any("Light Speed" in r.fact.title for r in results)


def test_search_empty_query(tmp_path):
    retriever = DatasetRetriever(data_path=_write_facts(tmp_path, SAMPLE_FACTS))
    assert retriever.search("") == []
    assert retriever.search("   ") == []


def test_search_no_matches(tmp_path):
    retriever = DatasetRetriever(data_path=_write_facts(tmp_path, SAMPLE_FACTS))
    assert retriever.search("zzxxqqyy nonsense query") == []


def test_search_top_k_limit(tmp_path):
    retriever = DatasetRetriever(data_path=_write_facts(tmp_path, SAMPLE_FACTS), threshold=0.0)
    results = retriever.search("the", top_k=2)
    assert len(results) <= 2
