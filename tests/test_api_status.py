import types

import pytest

from affinetes import api


class FakeWrapper:
    def __init__(self, name, ready=True, stats=None):
        self.name = name
        self._ready = ready
        self._stats = stats

    def is_ready(self):
        return self._ready

    def get_stats(self):
        return self._stats


class FakeRegistry:
    def __init__(self, mapping):
        self._mapping = dict(mapping)

    def get(self, env_id):
        return self._mapping.get(env_id)

    def list_all(self):
        return list(self._mapping.keys())


def test_get_environment_status_returns_none_for_missing_env(monkeypatch):
    registry = FakeRegistry({})
    monkeypatch.setattr(api, "get_registry", lambda: registry)

    status = api.get_environment_status("missing-env")
    assert status is None


def test_get_environment_status_single_instance(monkeypatch):
    wrapper = FakeWrapper("env-1", ready=True, stats=None)
    registry = FakeRegistry({"env-1": wrapper})
    monkeypatch.setattr(api, "get_registry", lambda: registry)

    status = api.get_environment_status("env-1")
    assert status is not None
    assert status["id"] == "env-1"
    assert status["name"] == "env-1"
    assert status["ready"] is True
    assert status["is_pool"] is False
    assert "stats" not in status


def test_get_environment_status_includes_stats_for_pools(monkeypatch):
    stats = {
        "total_instances": 3,
        "total_requests": 10,
        "instances": [
            {"host": "h1", "port": 8000, "requests": 3},
            {"host": "h2", "port": 8000, "requests": 7},
        ],
    }
    wrapper = FakeWrapper("pool-1", ready=True, stats=stats)
    registry = FakeRegistry({"pool-1": wrapper})
    monkeypatch.setattr(api, "get_registry", lambda: registry)

    status = api.get_environment_status("pool-1")
    assert status is not None
    assert status["id"] == "pool-1"
    assert status["is_pool"] is True
    assert status["stats"] == stats


def test_get_environment_status_excludes_stats_when_flag_false(monkeypatch):
    stats = {"total_instances": 2, "total_requests": 5, "instances": []}
    wrapper = FakeWrapper("pool-2", ready=False, stats=stats)
    registry = FakeRegistry({"pool-2": wrapper})
    monkeypatch.setattr(api, "get_registry", lambda: registry)

    status = api.get_environment_status("pool-2", include_stats=False)
    assert status is not None
    assert status["ready"] is False
    assert status["is_pool"] is True
    assert "stats" not in status


def test_get_all_environment_statuses_aggregates_multiple_envs(monkeypatch):
    w1 = FakeWrapper("env-1", ready=True, stats=None)
    w2_stats = {"total_instances": 2, "total_requests": 4, "instances": []}
    w2 = FakeWrapper("pool-1", ready=True, stats=w2_stats)

    registry = FakeRegistry({"env-1": w1, "pool-1": w2})
    monkeypatch.setattr(api, "get_registry", lambda: registry)

    statuses = api.get_all_environment_statuses()
    assert len(statuses) == 2

    by_id = {s["id"]: s for s in statuses}
    assert by_id["env-1"]["is_pool"] is False
    assert by_id["pool-1"]["is_pool"] is True
    assert by_id["pool-1"]["stats"] == w2_stats

