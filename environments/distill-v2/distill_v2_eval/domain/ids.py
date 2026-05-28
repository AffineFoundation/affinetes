"""Typed identifiers.

These are ``NewType``s so static checks distinguish e.g. a ``StudentName`` from
a free-form ``str`` even though both are string at runtime.
"""

from __future__ import annotations

from typing import NewType
from uuid import UUID

EnvName = NewType("EnvName", str)
TaskId = NewType("TaskId", int)
TeacherName = NewType("TeacherName", str)
StudentName = NewType("StudentName", str)
Revision = NewType("Revision", str)
PairSetName = NewType("PairSetName", str)
SamplingListName = NewType("SamplingListName", str)
DeploymentId = NewType("DeploymentId", str)

RolloutId = NewType("RolloutId", UUID)
PairId = NewType("PairId", UUID)
RunId = NewType("RunId", UUID)


__all__ = [
    "DeploymentId",
    "EnvName",
    "PairId",
    "PairSetName",
    "Revision",
    "RolloutId",
    "RunId",
    "SamplingListName",
    "StudentName",
    "TaskId",
    "TeacherName",
]
