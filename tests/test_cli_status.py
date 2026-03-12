import json

from affinetes.cli import commands


def test_show_status_no_environments_logs_message(monkeypatch, capsys):
    monkeypatch.setattr(commands, "get_all_environment_statuses", lambda include_stats=True: [])

    commands.show_status(name=None, output="table", verbose=False)
    captured = capsys.readouterr()

    # The logger writes to stderr, but we want to ensure the user gets feedback.
    assert "No active environments" in captured.err or "No active environments" in captured.out


def test_show_status_specific_missing_environment(monkeypatch, capsys):
    monkeypatch.setattr(commands, "get_environment_status", lambda name, include_stats=True: None)

    commands.show_status(name="missing-env", output="table", verbose=False)
    captured = capsys.readouterr()

    assert "missing-env" in captured.err or "missing-env" in captured.out


def test_show_status_json_output(monkeypatch, capsys):
    sample_statuses = [
        {
            "id": "env-1",
            "name": "env-1",
            "ready": True,
            "is_pool": False,
        },
        {
            "id": "pool-1",
            "name": "pool-1",
            "ready": True,
            "is_pool": True,
            "stats": {
                "total_instances": 2,
                "total_requests": 5,
                "instances": [],
            },
        },
    ]

    monkeypatch.setattr(
        commands,
        "get_all_environment_statuses",
        lambda include_stats=True: sample_statuses,
    )

    commands.show_status(name=None, output="json", verbose=False)
    captured = capsys.readouterr()

    data = json.loads(captured.out)
    assert isinstance(data, list)
    assert {item["id"] for item in data} == {"env-1", "pool-1"}


def test_show_status_table_output_without_tabulate(monkeypatch, capsys):
    # Force ImportError for tabulate so that the fallback branch is exercised
    def fake_import_tabulate(*args, **kwargs):
        raise ImportError("tabulate not available")

    monkeypatch.setattr(commands, "get_all_environment_statuses", lambda include_stats=True: [
        {
            "id": "env-1",
            "name": "env-1",
            "ready": True,
            "is_pool": False,
        },
    ])

    # Monkeypatch the import inside show_status by temporarily removing tabulate
    import builtins

    real_import = builtins.__import__

    def mocked_import(name, *args, **kwargs):  # pragma: no cover - defensive
        if name == "tabulate":
            raise ImportError("tabulate not available")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = mocked_import
    try:
        commands.show_status(name=None, output="table", verbose=False)
    finally:
        builtins.__import__ = real_import

    captured = capsys.readouterr()
    assert "ID" in captured.out
    assert "env-1" in captured.out


def test_show_status_verbose_includes_instance_details(monkeypatch, capsys):
    sample_statuses = [
        {
            "id": "pool-1",
            "name": "pool-1",
            "ready": True,
            "is_pool": True,
            "stats": {
                "total_instances": 2,
                "total_requests": 3,
                "instances": [
                    {"host": "h1", "port": 8000, "requests": 1},
                    {"host": "h2", "port": 8001, "requests": 2},
                ],
            },
        },
    ]

    monkeypatch.setattr(
        commands,
        "get_all_environment_statuses",
        lambda include_stats=True: sample_statuses,
    )

    # We don't care whether tabulate is available here; both branches print the table first.
    commands.show_status(name=None, output="table", verbose=True)
    captured = capsys.readouterr()

    assert "Instance details" in captured.out
    assert "pool-1" in captured.out
    assert "h1:8000" in captured.out
    assert "h2:8001" in captured.out

