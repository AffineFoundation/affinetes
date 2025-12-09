"""Shared helpers for budgeted query games."""

from __future__ import annotations


class BudgetMixin:
    query_budget: int
    queries_used: int

    def _count(self):
        self.queries_used += 1
        if self.queries_used > self.query_budget:
            raise ValueError("Query budget exceeded")
