import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dataset import DatasetRetriever
from groq_client import LLMError, answer_question
from superhero import SuperheroResult

SAMPLE_FACTS = [
    {"id": 1, "title": "Light Speed", "text": "Light travels at 299,792 km/s"},
    {"id": 2, "title": "Mars", "text": "Mars has a thin atmosphere"},
    {"id": 3, "title": "Water", "text": "71% of Earth is water"},
    {"id": 4, "title": "DNA", "text": "Human DNA is 99.9% identical"},
    {"id": 5, "title": "Moon", "text": "The Moon formed 4.5 billion years ago"},
]


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "facts.json"
    path.write_text(json.dumps(SAMPLE_FACTS))
    return DatasetRetriever(data_path=str(path))


@pytest.fixture
def superhero_adapter():
    adapter = MagicMock()
    adapter.search.return_value = SuperheroResult(
        found=True,
        hero={"id": 70, "name": "Batman", "biography": {"full-name": "Bruce Wayne", "alignment": "good"}, "powerstats": {}},
    )
    return adapter


def _tool_call(call_id: str, name: str, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _completion_with_tool_calls(tool_calls):
    message = SimpleNamespace(content=None, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _completion_with_text(text):
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_answer_question_dataset_only(dataset, superhero_adapter):
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _completion_with_tool_calls([_tool_call("call_1", "search_dataset", {"query": "light speed"})]),
        _completion_with_text("Light travels at 299,792 km/s. Source: local science facts dataset"),
    ]

    answer, tools_used = answer_question(client, "model", "How fast is light?", dataset, superhero_adapter)

    assert "Light travels" in answer
    assert tools_used == ["search_dataset"]
    superhero_adapter.search.assert_not_called()
    assert client.chat.completions.create.call_count == 2


def test_answer_question_superhero_only(dataset, superhero_adapter):
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _completion_with_tool_calls([_tool_call("call_1", "search_superhero", {"name": "Batman"})]),
        _completion_with_text("Batman is a superhero. Source: superheroapi.com"),
    ]

    answer, tools_used = answer_question(client, "model", "Who is Batman?", dataset, superhero_adapter)

    assert "Batman" in answer
    assert tools_used == ["search_superhero"]
    superhero_adapter.search.assert_called_once_with("Batman")


def test_answer_question_both_tools(dataset, superhero_adapter):
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _completion_with_tool_calls([
            _tool_call("call_1", "search_dataset", {"query": "light speed"}),
            _tool_call("call_2", "search_superhero", {"name": "Superman"}),
        ]),
        _completion_with_text("Comparison answer. Source: local science facts dataset, superheroapi.com"),
    ]

    answer, tools_used = answer_question(client, "model", "Compare Superman's speed to light", dataset, superhero_adapter)

    assert set(tools_used) == {"search_dataset", "search_superhero"}


def test_answer_question_no_tools_out_of_scope(dataset, superhero_adapter):
    client = MagicMock()
    client.chat.completions.create.return_value = _completion_with_text(
        "I can only answer questions about superheroes or science facts."
    )

    answer, tools_used = answer_question(client, "model", "What's the weather today?", dataset, superhero_adapter)

    assert "only answer questions" in answer
    assert tools_used == []
    superhero_adapter.search.assert_not_called()
    assert client.chat.completions.create.call_count == 1


def test_answer_question_empty_question_raises(dataset, superhero_adapter):
    client = MagicMock()
    with pytest.raises(LLMError, match="cannot be empty"):
        answer_question(client, "model", "", dataset, superhero_adapter)
    client.chat.completions.create.assert_not_called()


def test_answer_question_routing_call_failure_raises(dataset, superhero_adapter):
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("network error")

    with pytest.raises(LLMError, match="Groq routing call failed"):
        answer_question(client, "model", "How fast is light?", dataset, superhero_adapter)


def test_answer_question_synthesis_call_failure_raises(dataset, superhero_adapter):
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _completion_with_tool_calls([_tool_call("call_1", "search_dataset", {"query": "light"})]),
        Exception("network error"),
    ]

    with pytest.raises(LLMError, match="Groq synthesis call failed"):
        answer_question(client, "model", "How fast is light?", dataset, superhero_adapter)


def test_answer_question_rejects_empty_tool_argument(dataset, superhero_adapter):
    client = MagicMock()
    client.chat.completions.create.return_value = _completion_with_tool_calls(
        [_tool_call("call_1", "search_dataset", {"query": "   "})]
    )

    with pytest.raises(LLMError, match="empty argument"):
        answer_question(client, "model", "How fast is light?", dataset, superhero_adapter)


def test_answer_question_rejects_invalid_json_arguments(dataset, superhero_adapter):
    client = MagicMock()
    bad_call = SimpleNamespace(id="call_1", function=SimpleNamespace(name="search_dataset", arguments="not json"))
    client.chat.completions.create.return_value = _completion_with_tool_calls([bad_call])

    with pytest.raises(LLMError, match="invalid arguments"):
        answer_question(client, "model", "How fast is light?", dataset, superhero_adapter)


def test_answer_question_empty_final_answer_raises(dataset, superhero_adapter):
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _completion_with_tool_calls([_tool_call("call_1", "search_dataset", {"query": "light"})]),
        _completion_with_text(""),
    ]

    with pytest.raises(LLMError, match="empty final answer"):
        answer_question(client, "model", "How fast is light?", dataset, superhero_adapter)
