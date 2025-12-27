"""Tree Reconstruction Task implementation.

Provides the task interface for tree reconstruction following the affinetes
Challenge/Response pattern.
"""

import re
import random
import math
from typing import Optional, List, Tuple
from dataclasses import dataclass

from ._session import SessionManager, get_session_manager


@dataclass
class TreeChallenge:
    """Challenge specification for tree reconstruction.

    Attributes:
        env: Environment name ("tree")
        session_id: Unique session identifier
        n: Number of nodes in the tree
        seed: Random seed used for generation
        prompt: Initial prompt for the agent
        extra: Additional metadata
    """
    env: str
    session_id: str
    n: int
    seed: int
    prompt: str
    extra: dict


class TreeReconstructionTask:
    """Tree Reconstruction Task following affinetes patterns.

    This task implements a multi-turn interactive environment where:
    1. A hidden tree is generated from a seed
    2. The agent queries the tree to gather information
    3. The agent submits its reconstruction
    4. The reconstruction is scored against ground truth

    Interaction Protocol:
    - generate() creates a new session and returns the challenge prompt
    - query() processes agent queries and returns results
    - submit() evaluates the agent's reconstruction

    Batch Queries:
    - Agents can submit multiple QUERY commands in a single message
    - All queries are processed and results returned together
    - This reduces turns while preserving information complexity

    For single-turn evaluation, use evaluate() which runs the full loop.
    """

    # Query type patterns for parsing agent responses (finds ALL matches)
    QUERY_PATTERN = re.compile(
        r'QUERY\s+(ANCESTOR|LCA|DEPTH|CHILDREN|PATH)\s+(\d+)(?:\s+(\d+))?',
        re.IGNORECASE
    )
    SUBMIT_PATTERN = re.compile(
        r'SUBMIT\s+([\d\s,\-]+)',
        re.IGNORECASE
    )
    MAX_TURNS = 100

    def __init__(
        self,
        default_n: int = 20,
        default_method: str = "prufer",
        max_queries: Optional[int] = None,
        allowed_query_types: Optional[List[str]] = None,
        session_manager: Optional[SessionManager] = None
    ):
        """Initialize the task.

        Args:
            default_n: Default number of nodes
            default_method: Default generation method ("prufer" or "recursive")
            max_queries: Optional query limit (None = unlimited)
            allowed_query_types: Restrict available queries (None = all)
        """
        self.default_n = default_n
        self.default_method = default_method
        self.max_queries = max_queries
        self.allowed_query_types = allowed_query_types or ["ANCESTOR", "LCA", "DEPTH", "CHILDREN", "PATH"]
        self.session_manager = session_manager or get_session_manager()

    async def generate(
        self,
        n: Optional[int] = None,
        seed: Optional[int] = None,
        method: Optional[str] = None,
        task_id: Optional[int] = None
    ) -> TreeChallenge:
        """Generate a new tree reconstruction challenge.

        Args:
            n: Number of nodes (default: self.default_n)
            seed: Random seed (default: random)
            method: Generation method (default: self.default_method)
            task_id: Alternative to seed for compatibility with affinetes

        Returns:
            TreeChallenge with session info and initial prompt
        """
        n = n or self.default_n
        method = method or self.default_method

        # Use task_id as seed if provided (for affinetes compatibility)
        if seed is None:
            seed = task_id if task_id is not None else random.randint(0, 2**63 - 1)

        # Create session
        session_info = self.session_manager.create_session(n, seed, method)

        # Build challenge prompt
        prompt = self._build_prompt(n, session_info)

        return TreeChallenge(
            env="tree",
            session_id=session_info.session_id,
            n=n,
            seed=seed,
            prompt=prompt,
            extra={
                "method": method,
                "info_lower_bound": session_info.info_lower_bound,
                "max_queries": self.max_queries,
                "allowed_query_types": self.allowed_query_types
            }
        )

    def _build_prompt(self, n: int, session_info) -> str:
        """Build the initial challenge prompt."""
        lb = session_info.info_lower_bound

        query_docs = []
        if "ANCESTOR" in self.allowed_query_types:
            query_docs.append("- QUERY ANCESTOR u v → YES if u is an ancestor of v, NO otherwise (1 bit)")
        if "LCA" in self.allowed_query_types:
            query_docs.append(f"- QUERY LCA u v → lowest common ancestor node index (~{math.ceil(math.log2(n))} bits)")
        if "DEPTH" in self.allowed_query_types:
            query_docs.append("- QUERY DEPTH v → depth of node v, root has depth 0")
        if "CHILDREN" in self.allowed_query_types:
            query_docs.append("- QUERY CHILDREN v → list of all children of node v (powerful)")
        if "PATH" in self.allowed_query_types:
            query_docs.append("- QUERY PATH u v → all nodes on path from u to v (very powerful)")

        query_section = "\n".join(query_docs)

        query_limit = f"Query limit: {self.max_queries}" if self.max_queries else "No query limit"

        return f"""You are reconstructing a hidden rooted tree with {n} nodes labeled 0 to {n-1}.
Node 0 is the root. Your goal is to determine the parent of each node from 1 to {n-1}.

Available queries (you may issue MULTIPLE queries per message):
{query_section}

To submit your final answer:
- SUBMIT p1 p2 p3 ... p{{n-1}}
  where p_i is the parent of node i (space or comma separated)
  Example for n=4: SUBMIT 0 0 1 means parent[1]=0, parent[2]=0, parent[3]=1

Rules:
- {query_limit}
- You can issue multiple QUERY commands in a single message (batch queries)
- The root (node 0) has no parent - you only submit parents for nodes 1 to {n-1}

Information theory:
- There are {n}^{n-1} possible trees ≈ {n**(n-1):.0e} trees
- Lower bound: ~{lb:.0f} bits needed to identify the tree
- CHILDREN query on root can reveal significant structure quickly

Strategy hint: Start with QUERY CHILDREN 0 to see the root's children, then explore recursively.

Begin by making queries to learn the tree structure."""

    async def process_response(
        self,
        session_id: str,
        response: str
    ) -> Tuple[str, bool, Optional[dict]]:
        """Process an agent response (query or submission).

        Args:
            session_id: Session identifier
            response: Agent's response text

        Returns:
            Tuple of (result_message, is_complete, evaluation_result)
        """
        # Try to parse as submission first
        submit_match = self.SUBMIT_PATTERN.search(response)
        if submit_match:
            try:
                # Parse parent array
                parent_str = submit_match.group(1)
                parent_values = [int(x) for x in re.findall(r'-?\d+', parent_str)]

                # Get session to know n
                state = self.session_manager.get_session_state(session_id)
                n = state["n"]

                # Construct full parent array (parent[0] = -1 for root)
                if len(parent_values) == n - 1:
                    parent_array = [-1] + parent_values
                elif len(parent_values) == n:
                    parent_array = parent_values
                else:
                    return (
                        f"Error: Expected {n-1} or {n} parent values, got {len(parent_values)}",
                        False,
                        None
                    )

                # Submit reconstruction
                result = self.session_manager.submit(session_id, parent_array)

                efficiency_str = f"{result['efficiency']:.2%}" if result['efficiency'] else "N/A (no queries)"
                msg = f"""Submission evaluated!
Score: {result['score']:.2%} ({result['correct']}/{result['total']} correct)
Queries used: {result['query_count']}
Information bits used: {result['total_bits']:.1f}
Theoretical lower bound: {result['info_lower_bound']:.1f} bits
Efficiency: {efficiency_str}"""

                return (msg, True, result)

            except ValueError as e:
                return (f"Error parsing submission: {e}", False, None)

        # Try to parse as query
        query_matches = list(self.QUERY_PATTERN.finditer(response))
        if query_matches:
            outputs = []
            for match in query_matches:
                query_type = match.group(1).upper()
                if query_type not in self.allowed_query_types:
                    outputs.append(f"Error: Query type {query_type} not allowed. Available: {self.allowed_query_types}")
                    continue

                state = self.session_manager.get_session_state(session_id)
                if self.max_queries and state["query_count"] >= self.max_queries:
                    outputs.append(f"Error: Query limit ({self.max_queries}) reached. You must submit your answer.")
                    break

                args = [int(match.group(2))]
                if match.group(3):
                    args.append(int(match.group(3)))

                try:
                    result = self.session_manager.query(session_id, query_type, args)
                    if query_type == "ANCESTOR":
                        answer = "YES" if result["result"] else "NO"
                    elif query_type in ("CHILDREN", "PATH"):
                        answer = "[" + ", ".join(map(str, result["result"])) + "]"
                    else:
                        answer = str(result["result"])

                    outputs.append(
                        f"Query: {query_type} {' '.join(map(str, args))}\n"
                        f"Result: {answer}\n"
                        f"Total queries: {result['total_queries']}\n"
                        f"Bits used: {result['total_bits']:.1f} / {result['info_lower_bound']:.1f} lower bound"
                    )
                except ValueError as e:
                    outputs.append(f"Error: {e}")

            return ("\n\n".join(outputs), False, None)

        # Could not parse response
        return (
            "Could not parse your response. Please use:\n"
            "- QUERY ANCESTOR u v\n"
            "- QUERY LCA u v\n"
            "- QUERY DEPTH v\n"
            "- QUERY CHILDREN v\n"
            "- QUERY PATH u v\n"
            "- SUBMIT p1 p2 ... p{n-1}",
            False,
            None
        )

    async def evaluate(
        self,
        response: str,
        challenge: TreeChallenge
    ) -> float:
        """Evaluate a final response (for single-turn compatibility).

        This extracts the submission from the response and evaluates it.

        Args:
            response: Agent's full response
            challenge: The original challenge

        Returns:
            Score from 0.0 to 1.0
        """
        submit_match = self.SUBMIT_PATTERN.search(response)
        if not submit_match:
            return 0.0

        try:
            parent_str = submit_match.group(1)
            parent_values = [int(x) for x in re.findall(r'-?\d+', parent_str)]

            n = challenge.n
            if len(parent_values) == n - 1:
                parent_array = [-1] + parent_values
            elif len(parent_values) == n:
                parent_array = parent_values
            else:
                return 0.0

            result = self.session_manager.submit(challenge.session_id, parent_array)
            return result["score"]

        except (ValueError, KeyError):
            return 0.0

    def get_ground_truth(self, session_id: str) -> List[int]:
        """Get ground truth parent array for a session (for debugging).

        Args:
            session_id: Session identifier

        Returns:
            Ground truth parent array
        """
        if session_id not in self.session_manager.sessions:
            raise ValueError(f"Session not found: {session_id}")
        return self.session_manager.sessions[session_id].tree.get_ground_truth()
