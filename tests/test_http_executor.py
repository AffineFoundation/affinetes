from unittest.mock import AsyncMock, MagicMock

import pytest

from affinetes.infrastructure.http_executor import HTTPExecutor


@pytest.mark.asyncio
async def test_health_check_requires_only_http_200() -> None:
    executor = object.__new__(HTTPExecutor)
    executor.base_url = "http://environment:8000"
    executor.client = MagicMock()
    response = MagicMock(status_code=200)
    response.json.side_effect = AssertionError("health readiness must not parse JSON")
    executor.client.get = AsyncMock(return_value=response)

    assert await executor.health_check() is True

    executor.client.get.assert_awaited_once_with(
        "http://environment:8000/health",
        timeout=5,
    )
    response.json.assert_not_called()


@pytest.mark.asyncio
async def test_health_check_rejects_non_200_response() -> None:
    executor = object.__new__(HTTPExecutor)
    executor.base_url = "http://environment:8000"
    executor.client = MagicMock()
    executor.client.get = AsyncMock(return_value=MagicMock(status_code=204))

    assert await executor.health_check() is False
