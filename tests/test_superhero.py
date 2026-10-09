from unittest.mock import MagicMock, patch

import httpx
import pytest

from superhero import SuperheroAdapter


@pytest.fixture
def adapter():
    return SuperheroAdapter(token="test_token", max_failures=5)


def _mock_client(json_data=None, raise_exc=None):
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    if raise_exc:
        mock_client.get.side_effect = raise_exc
    else:
        mock_response = MagicMock()
        mock_response.json.return_value = json_data
        mock_response.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_response
    return mock_client


def test_search_success(adapter):
    data = {"response": "success", "results": [{"id": 70, "name": "Batman", "powerstats": {}}]}
    with patch("httpx.Client", return_value=_mock_client(json_data=data)):
        result = adapter.search("Batman")
    assert result.found is True
    assert result.hero["name"] == "Batman"


def test_search_not_found(adapter):
    data = {"response": "error", "error": "character with given name not found"}
    with patch("httpx.Client", return_value=_mock_client(json_data=data)):
        result = adapter.search("NonexistentHero")
    assert result.found is False
    assert "not found" in result.error.lower()


def test_search_empty_name(adapter):
    result = adapter.search("")
    assert result.found is False
    assert result.error == "Name cannot be empty"


def test_search_timeout(adapter):
    with patch("httpx.Client", return_value=_mock_client(raise_exc=httpx.TimeoutException("timed out"))):
        result = adapter.search("Batman")
    assert result.found is False
    assert "timed out" in result.error.lower()


def test_search_http_error(adapter):
    mock_response = MagicMock()
    mock_response.status_code = 500
    exc = httpx.HTTPStatusError("Server error", request=MagicMock(), response=mock_response)
    with patch("httpx.Client", return_value=_mock_client(raise_exc=exc)):
        result = adapter.search("Batman")
    assert result.found is False
    assert "HTTP error" in result.error


def test_search_exact_match_preferred(adapter):
    data = {
        "response": "success",
        "results": [{"id": 1, "name": "Batgirl"}, {"id": 2, "name": "Batman"}],
    }
    with patch("httpx.Client", return_value=_mock_client(json_data=data)):
        result = adapter.search("Batman")
    assert result.hero["name"] == "Batman"


def test_circuit_breaker(adapter):
    with patch("httpx.Client", return_value=_mock_client(raise_exc=httpx.TimeoutException("timed out"))):
        for _ in range(5):
            adapter.search("Batman")
        result = adapter.search("Batman")
    assert "temporarily unavailable" in result.error.lower()


def test_reset_failure_count(adapter):
    adapter.failure_count = 3
    adapter.reset_failure_count()
    assert adapter.failure_count == 0


def test_failure_count_resets_on_success(adapter):
    adapter.failure_count = 2
    data = {"response": "success", "results": [{"id": 1, "name": "Batman"}]}
    with patch("httpx.Client", return_value=_mock_client(json_data=data)):
        adapter.search("Batman")
    assert adapter.failure_count == 0
