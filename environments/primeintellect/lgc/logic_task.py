"""Logic task generator using i3_logic implementation"""

import json
import random
from datasets import load_dataset

import sys
sys.path.insert(0, '/app')

from i3_logic.base.data import Data
from i3_logic.task2verifier import verifier_classes


class LogicTask:
    """Logic task generator and evaluator using INTELLECT-3-RL logic dataset"""
    
    def __init__(
        self,
        dataset_name: str = "PrimeIntellect/INTELLECT-3-RL",
        dataset_subset: str = "logic",
        dataset_split: str = "train",
        dataset_shuffle: bool = False,
        difficulty_key: str = "avg@16_qwen3_4b_instruct_2507",
        min_avg_reward: float = 0.0,
        max_avg_reward: float = 1.0,
        tasks_to_skip: list = None,
    ):
        """
        Initialize LogicTask with dataset configuration
        
        Args:
            dataset_name: HuggingFace dataset name
            dataset_subset: Dataset subset to use
            dataset_split: Dataset split (train/test/validation)
            dataset_shuffle: Whether to shuffle the dataset
            difficulty_key: Key for filtering by difficulty
            min_avg_reward: Minimum average reward filter
            max_avg_reward: Maximum average reward filter
            tasks_to_skip: List of task types to skip
        """
        if tasks_to_skip is None:
            tasks_to_skip = ["arc_agi", "arc_agi_2", "buggy_tables"]
        
        # Load and filter dataset
        self.dataset = (
            load_dataset(dataset_name, dataset_subset, split=dataset_split)
            .map(lambda x: {"info": json.loads(x["info"]), "answer": ""})
            .filter(lambda x: x["info"]["task"] not in tasks_to_skip)
            .filter(lambda x: min_avg_reward <= x.get(difficulty_key, 0) <= max_avg_reward)
            .select_columns(["question", "answer", "info"])
        )
        
        if dataset_shuffle:
            self.dataset = self.dataset.shuffle(seed=42)
    
    async def generate(self, task_id: int = None):
        """
        Generate a logic task challenge
        
        Args:
            task_id: Optional task ID for deterministic selection.
                     If provided, used as index into dataset.
                     If not provided, random sample is selected.
        
        Returns:
            Challenge object with prompt and extra info
        """
        from models import Challenge
        
        if task_id is not None:
            idx = task_id % len(self.dataset)
            sample = self.dataset[idx]
        else:
            idx = random.randint(0, len(self.dataset) - 1)
            sample = self.dataset[idx]
        
        return Challenge(
            env="logic",
            prompt=sample["question"],
            extra={
                "info": sample["info"],
                "task_id": task_id,
                "dataset_index": idx,
                "task_type": sample["info"]["task"]
            }
        )
    
    async def evaluate(self, response: str, challenge):
        """
        Evaluate logic response using task-specific verifier
        
        Args:
            response: Model response
            challenge: Original challenge with info
        
        Returns:
            Score (0.0 or 1.0)
        """
        info = challenge.extra.get("info", {})
        game_data = info.get("game_data_str") or info.get("game_data")
        task = info.get("task")
        
        verifier_cls = verifier_classes.get(task)
        if verifier_cls is None:
            raise ValueError(f"Verifier class not found for task: {task}")
        
        verifier = verifier_cls()
        data_obj = Data.from_json_str(game_data)
        
        # Parse response (handle <think> tags)
        parsed_answer = self._parse_response(response)
        
        return float(verifier.verify(data_obj, parsed_answer))
    
    def _parse_response(self, text: str) -> str:
        """Parse response, removing <think> tags if present"""
        if "<think>" in text:
            if "</think>" not in text:
                return ""
            text = text.split("</think>")[-1].strip()
        return text.strip()