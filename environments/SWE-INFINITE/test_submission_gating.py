"""MiniSWEAgent.solve must only accept a patch from a Submitted run.

A run that terminates any other way (LimitsExceeded at step limit, etc.)
must yield an empty patch — no git-diff fallback from the container — so
that unsubmitted work scores 0, matching upstream mini-swe-agent +
SWE-bench semantics.

Run: python3 environments/SWE-INFINITE/test_submission_gating.py
"""
import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "agents"))

from agents import miniswe as miniswe_mod
from agents.miniswe import MiniSWEAgent, MiniSWEConfig

SAMPLE_DIFF = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"


class FakeModel:
    n_calls = 7
    cost = 0.1


class FakeDefaultAgent:
    """Stands in for minisweagent DefaultAgent inside solve()."""

    run_result = ("LimitsExceeded", "")

    def __init__(self, model, env, **kwargs):
        self.model = FakeModel()
        self.messages = []
        self.config = types.SimpleNamespace(**kwargs)

    def run(self, task):
        return type(self).run_result


class FakeDockerEnv:
    def __init__(self, **kwargs):
        pass

    def cleanup(self):
        pass


def run_solve(run_result):
    FakeDefaultAgent.run_result = run_result
    agent = MiniSWEAgent(MiniSWEConfig(model="m", api_base="http://x", api_key="k"))

    import minisweagent.environments.docker as msa_docker

    fake_subprocess_result = types.SimpleNamespace(returncode=0, stdout="", stderr="")
    with (
        # ThinkingAwareAgent subclasses miniswe's module-level DefaultAgent
        # at solve() call time, so patch it there.
        patch.object(miniswe_mod, "DefaultAgent", FakeDefaultAgent),
        patch.object(msa_docker, "DockerEnvironment", FakeDockerEnv),
        patch.object(miniswe_mod.subprocess, "run", return_value=fake_subprocess_result),
        patch.object(miniswe_mod, "is_image_prepared", return_value=True),
    ):
        return asyncio.get_event_loop().run_until_complete(
            agent.solve(problem_statement="p", docker_image="img")
        )


CASES = [
    # (label, run() result, expected patch, expected exit_status)
    ("submitted run returns its patch", ("Submitted", SAMPLE_DIFF), SAMPLE_DIFF, "Submitted"),
    ("step limit yields empty patch", ("LimitsExceeded", ""), "", "LimitsExceeded"),
    # Non-empty terminating message must not be mistaken for a patch
    ("non-submit message is not a patch", ("LimitsExceeded", "some text"), "", "LimitsExceeded"),
]


def main():
    failed = 0
    for label, run_result, want_patch, want_status in CASES:
        res = run_solve(run_result)
        ok = res.error is None and res.patch == want_patch and res.exit_status == want_status
        # env.py maps empty patch + LimitsExceeded to limit_reached_no_submission
        # via getattr, which must also be safe on results lacking the field.
        ok = ok and getattr(object(), "exit_status", None) is None
        print(f"[{'PASS' if ok else 'FAIL'}] {label:40} patch={res.patch[:20]!r} exit_status={res.exit_status}")
        failed += 0 if ok else 1
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
