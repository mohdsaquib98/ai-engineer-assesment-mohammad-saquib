import pytest
from pydantic import ValidationError

from models import AskRequest, AskResponse


def test_ask_request_valid():
    request = AskRequest(question="How fast does light travel?")
    assert request.question == "How fast does light travel?"


def test_ask_request_strips_whitespace():
    request = AskRequest(question="  How fast does light travel?  ")
    assert request.question == "How fast does light travel?"


def test_ask_request_empty():
    with pytest.raises(ValidationError, match="Question cannot be empty"):
        AskRequest(question="")


def test_ask_request_whitespace_only():
    with pytest.raises(ValidationError, match="Question cannot be empty"):
        AskRequest(question="   ")


def test_ask_request_too_short():
    with pytest.raises(ValidationError, match="at least"):
        AskRequest(question="a")


def test_ask_request_too_long():
    with pytest.raises(ValidationError, match="too long"):
        AskRequest(question="a" * 1001)


def test_ask_response_structure():
    response = AskResponse(answer="Light travels fast.", tools_used=["search_dataset"])
    assert response.answer == "Light travels fast."
    assert response.tools_used == ["search_dataset"]


def test_ask_response_empty_tools_used():
    response = AskResponse(answer="I don't know.", tools_used=[])
    assert response.tools_used == []
