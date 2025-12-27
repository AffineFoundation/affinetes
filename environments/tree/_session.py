"""Session management for tree reconstruction.

This module handles the stateful multi-turn interaction between an agent
and the tree reconstruction environment.
"""

import uuid
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from ._hidden_tree import HiddenTree, QueryResult


class SessionStatus(Enum):
    """Status of a tree reconstruction session."""
    ACTIVE = "active"       # Session is active, queries allowed
    SUBMITTED = "submitted"  # Agent has submitted reconstruction
    TIMEOUT = "timeout"      # Session timed out
    ERROR = "error"          # Session encountered an error


@dataclass
class QueryRecord:
    """Record of a single query."""
    query_type: str
    args: Tuple
    result: Any
    bits: float
    query_number: int


@dataclass
class SessionInfo:
    """Information returned when starting a session."""
    session_id: str
    n: int
    root: int
    available_queries: List[str]
    info_lower_bound: float


@dataclass
class SessionState:
    """Complete state of a tree reconstruction session."""
    session_id: str
    n: int
    seed: int
    method: str
    tree: HiddenTree
    query_history: List[QueryRecord] = field(default_factory=list)
    total_bits: float = 0.0
    status: SessionStatus = SessionStatus.ACTIVE
    submitted_parent: Optional[List[int]] = None
    evaluation_result: Optional[dict] = None

    @property
    def query_count(self) -> int:
        return len(self.query_history)

    @property
    def info_lower_bound(self) -> float:
        return self.tree.get_theoretical_lower_bound()

    @property
    def efficiency(self) -> Optional[float]:
        """Query efficiency: lower_bound / bits_used (1.0 = optimal)."""
        if self.total_bits == 0:
            return None
        return self.info_lower_bound / self.total_bits

    def to_dict(self) -> dict:
        """Convert session state to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "n": self.n,
            "seed": self.seed,
            "method": self.method,
            "query_count": self.query_count,
            "total_bits": self.total_bits,
            "info_lower_bound": self.info_lower_bound,
            "efficiency": self.efficiency,
            "status": self.status.value,
            "submitted_parent": self.submitted_parent,
            "evaluation_result": self.evaluation_result,
        }


class SessionManager:
    """Manages multiple tree reconstruction sessions.

    This class handles:
    - Creating new sessions with deterministic tree generation
    - Processing queries for active sessions
    - Evaluating submitted reconstructions
    - Tracking session statistics
    """

    def __init__(self, max_sessions: int = 1000):
        """Initialize session manager.

        Args:
            max_sessions: Maximum number of concurrent sessions
        """
        self.sessions: Dict[str, SessionState] = {}
        self.max_sessions = max_sessions

    def create_session(
        self,
        n: int,
        seed: int,
        method: str = "prufer",
        session_id: Optional[str] = None
    ) -> SessionInfo:
        """Create a new tree reconstruction session.

        Args:
            n: Number of nodes in the tree
            seed: Random seed for deterministic generation
            method: Generation method ("prufer" or "recursive")
            session_id: Optional custom session ID

        Returns:
            SessionInfo with session details
        """
        if len(self.sessions) >= self.max_sessions:
            # Clean up old submitted/error sessions
            self._cleanup_old_sessions()
            if len(self.sessions) >= self.max_sessions:
                raise RuntimeError("Maximum number of sessions reached")

        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Create hidden tree
        tree = HiddenTree(n, seed, method)

        # Create session state
        state = SessionState(
            session_id=session_id,
            n=n,
            seed=seed,
            method=method,
            tree=tree
        )

        self.sessions[session_id] = state

        return SessionInfo(
            session_id=session_id,
            n=n,
            root=0,
            available_queries=tree.available_queries,
            info_lower_bound=tree.get_theoretical_lower_bound()
        )

    def query(
        self,
        session_id: str,
        query_type: str,
        args: List[int]
    ) -> dict:
        """Execute a query on a session.

        Args:
            session_id: Session identifier
            query_type: One of "ANCESTOR", "LCA", "DEPTH"
            args: Query arguments (node indices)

        Returns:
            Dictionary with query result and session statistics
        """
        state = self._get_active_session(session_id)

        query_type = query_type.upper()
        result = state.tree.run_query(query_type, args)

        # Record query
        record = QueryRecord(
            query_type=query_type,
            args=tuple(args),
            result=result.result,
            bits=result.bits,
            query_number=len(state.query_history) + 1
        )
        state.query_history.append(record)
        state.total_bits += result.bits

        return {
            "result": result.result,
            "query_type": query_type,
            "args": args,
            "bits": result.bits,
            "total_queries": state.query_count,
            "total_bits": state.total_bits,
            "info_lower_bound": state.info_lower_bound
        }

    def submit(
        self,
        session_id: str,
        parent_array: List[int]
    ) -> dict:
        """Submit a reconstruction for evaluation.

        Args:
            session_id: Session identifier
            parent_array: Predicted parent array

        Returns:
            Dictionary with evaluation results
        """
        state = self._get_active_session(session_id)

        # Validate parent array
        if len(parent_array) != state.n:
            raise ValueError(f"Expected parent array of length {state.n}, got {len(parent_array)}")

        # Evaluate reconstruction
        eval_result = state.tree.evaluate_reconstruction(parent_array)

        # Update session state
        state.submitted_parent = parent_array
        state.evaluation_result = eval_result
        state.status = SessionStatus.SUBMITTED

        return {
            "score": eval_result["score"],
            "correct": eval_result["correct"],
            "total": eval_result["total"],
            "query_count": state.query_count,
            "total_bits": state.total_bits,
            "info_lower_bound": state.info_lower_bound,
            "efficiency": state.efficiency,
            "status": state.status.value
        }

    def get_session_state(self, session_id: str) -> dict:
        """Get current state of a session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with session state
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")
        return self.sessions[session_id].to_dict()

    def get_query_history(self, session_id: str) -> List[dict]:
        """Get query history for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of query records
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")

        return [
            {
                "query_number": r.query_number,
                "query_type": r.query_type,
                "args": r.args,
                "result": r.result,
                "bits": r.bits
            }
            for r in self.sessions[session_id].query_history
        ]

    def close_session(self, session_id: str):
        """Close and remove a session.

        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            del self.sessions[session_id]

    def _get_active_session(self, session_id: str) -> SessionState:
        """Get an active session, raising error if not found or not active."""
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")

        state = self.sessions[session_id]
        if state.status != SessionStatus.ACTIVE:
            raise ValueError(f"Session is not active (status: {state.status.value})")

        return state

    def _cleanup_old_sessions(self):
        """Remove completed or error sessions to free up space."""
        to_remove = [
            sid for sid, state in self.sessions.items()
            if state.status in (SessionStatus.SUBMITTED, SessionStatus.ERROR, SessionStatus.TIMEOUT)
        ]
        for sid in to_remove[:len(to_remove) // 2]:  # Remove half of old sessions
            del self.sessions[sid]


# Global session manager instance
_session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    return _session_manager
