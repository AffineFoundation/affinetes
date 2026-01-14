"""Logic Task V2 - Seed-based task generator"""

import sys
sys.path.insert(0, '/app')

from collections import OrderedDict
from base.data import Data
from games.verifiers import verifier_classes


class LogicTaskV2:
    """Seed-based logic task generator supporting dynamic task generation"""

    # Task ID allocation scheme:
    # Each task type gets a range of 100,000,000 task IDs
    # task_id = task_type_id * 100,000,000 + seed
    #
    # Examples:
    #   0-99,999,999: dyck_language
    #   100,000,000-199,999,999: future task type 1
    #   200,000,000-299,999,999: future task type 2

    TASK_ID_RANGE = 100_000_000  # Each task type gets 100M IDs

    # Supported task types with seed-based generation
    # Each task is in games/<task_name>/ with generator.py and verifier.py
    SUPPORTED_TASKS = {
        "dyck_language": {
            "task_type_id": 0,  # Range: 0-99,999,999
            "module": "games.dyck_language.generator",
            "class": "DyckLanguageGenerator",
            "default_config": {
                "n_types": 3,
                "total_length": 0,      # 0 = random (40-60)
                "to_fill_length": 0,    # 0 = random (20%-50% of total)
                "nesting_depth": 0      # 0 = no minimum depth requirement
            }
        },
        "game_of_24": {
            "task_type_id": 1,  # Range: 100,000,000-199,999,999
            "module": "games.game_of_24.generator",
            "class": "GameOf24Generator",
            "default_config": {}  # All parameters derived from seed
        },
        "operation": {
            "task_type_id": 2,  # Range: 200,000,000-299,999,999
            "module": "games.operation.generator",
            "class": "OperationGenerator",
            "default_config": {}  # All parameters derived from seed
        },
        "cryptarithm": {
            "task_type_id": 3,  # Range: 300,000,000-399,999,999
            "module": "games.cryptarithm.generator",
            "class": "CryptarithmGenerator",
            "default_config": {}  # All parameters derived from seed
        },
        "dyck_language2": {
            "task_type_id": [4, 5],  # Range: 400,000,000-599,999,999 (200M seeds)
            "module": "games.dyck_language2.generator",
            "class": "DyckLanguage2Generator",
            "default_config": {}  # n_types and length derived from seed
        },
        "boolean_expressions": {
            "task_type_id": 6,  # Range: 600,000,000-699,999,999
            "module": "games.boolean_expressions.generator",
            "class": "BooleanExpressionsGenerator",
            "default_config": {}  # All parameters derived from seed
        },
        "sudoku": {
            "task_type_id": 7,  # Range: 700,000,000-799,999,999
            "module": "games.sudoku.generator",
            "class": "SudokuGenerator",
            "default_config": {}  # All parameters derived from seed
        },
        "dyck_language_reasoning_errors": {
            "task_type_id": 8,  # Range: 800,000,000-899,999,999
            "module": "games.dyck_language_reasoning_errors.generator",
            "class": "DyckLanguageReasoningErrorsGenerator",
            "default_config": {
                "n_types": 0,       # 0 = random (2-4)
                "total_length": 0,  # 0 = random (10-30)
                "n_errors": 0       # 0 = random (1-5)
            }
        },
    }

    # Reverse mapping: task_type_id -> task_type_name
    TASK_TYPE_BY_ID = {}
    for name, info in SUPPORTED_TASKS.items():
        tid = info["task_type_id"]
        if isinstance(tid, list):
            for t in tid:
                TASK_TYPE_BY_ID[t] = name
        else:
            TASK_TYPE_BY_ID[tid] = name

    def __init__(self, task_configs: dict = None, max_cache_size: int = 1000):
        """
        Initialize LogicTaskV2 with task configurations

        Args:
            task_configs: Dict mapping task names to their configs
                         Format: {"dyck_language": {"n_types": 4, ...}}
                         If None, uses default configs
            max_cache_size: Maximum number of challenges to cache (default: 1000)
                           Set to 0 to disable caching
        """
        self.task_configs = task_configs or {}
        self._task_instances = {}
        self._max_cache_size = max_cache_size
        self._challenge_cache = OrderedDict()

    @staticmethod
    def decode_task_id(task_id: int):
        """
        Decode task_id into (task_type, seed)

        Args:
            task_id: Global task ID

        Returns:
            tuple: (task_type_name, seed)

        Examples:
            500 -> ("dyck_language", 500)
            100_000_500 -> ("game_of_24", 500)
            400_000_500 -> ("dyck_language2", 500)
            500_000_500 -> ("dyck_language2", 100_000_500)
        """
        task_type_id = task_id // LogicTaskV2.TASK_ID_RANGE
        raw_seed = task_id % LogicTaskV2.TASK_ID_RANGE

        if task_type_id not in LogicTaskV2.TASK_TYPE_BY_ID:
            raise ValueError(
                f"Invalid task_id {task_id}: task_type_id {task_type_id} not found. "
                f"Valid task_type_ids: {list(LogicTaskV2.TASK_TYPE_BY_ID.keys())}"
            )

        task_type = LogicTaskV2.TASK_TYPE_BY_ID[task_type_id]

        # Handle multi-range task types (e.g., dyck_language2 with task_type_id=[4,5])
        tid_config = LogicTaskV2.SUPPORTED_TASKS[task_type]["task_type_id"]
        if isinstance(tid_config, list):
            first_tid = min(tid_config)
            offset = (task_type_id - first_tid) * LogicTaskV2.TASK_ID_RANGE
            seed = offset + raw_seed
        else:
            seed = raw_seed

        return task_type, seed

    @staticmethod
    def encode_task_id(task_type: str, seed: int):
        """
        Encode (task_type, seed) into task_id

        Args:
            task_type: Task type name (e.g., "dyck_language")
            seed: Seed value

        Returns:
            int: Global task ID

        Examples:
            ("dyck_language", 500) -> 500
            ("game_of_24", 500) -> 100_000_500
            ("dyck_language2", 500) -> 400_000_500
            ("dyck_language2", 100_000_500) -> 500_000_500
        """
        if task_type not in LogicTaskV2.SUPPORTED_TASKS:
            raise ValueError(f"Unknown task type: {task_type}")

        tid_config = LogicTaskV2.SUPPORTED_TASKS[task_type]["task_type_id"]

        if isinstance(tid_config, list):
            # Multi-range: determine which task_type_id to use
            max_seed = len(tid_config) * LogicTaskV2.TASK_ID_RANGE
            if not 0 <= seed < max_seed:
                raise ValueError(f"Seed must be in range [0, {max_seed})")
            tid_index = seed // LogicTaskV2.TASK_ID_RANGE
            task_type_id = tid_config[tid_index]
            raw_seed = seed % LogicTaskV2.TASK_ID_RANGE
        else:
            if not 0 <= seed < LogicTaskV2.TASK_ID_RANGE:
                raise ValueError(f"Seed must be in range [0, {LogicTaskV2.TASK_ID_RANGE})")
            task_type_id = tid_config
            raw_seed = seed

        return task_type_id * LogicTaskV2.TASK_ID_RANGE + raw_seed

    def _get_task_instance(self, task_type: str):
        """Lazy load task instance"""
        if task_type not in self._task_instances:
            if task_type not in self.SUPPORTED_TASKS:
                raise ValueError(f"Unsupported task type: {task_type}")

            task_info = self.SUPPORTED_TASKS[task_type]
            module_path = task_info["module"]
            class_name = task_info["class"]

            # Dynamic import
            parts = module_path.split('.')
            module = __import__(module_path, fromlist=[class_name])
            task_class = getattr(module, class_name)

            self._task_instances[task_type] = task_class()

        return self._task_instances[task_type]

    async def generate(self, task_id: int):
        """
        Generate a task challenge using task_id

        The task_id encodes both the task type and seed:
        - task_type_id = task_id // 100,000,000
        - seed = task_id % 100,000,000

        Args:
            task_id: Global task ID (encodes task type and seed)

        Returns:
            Challenge object with prompt and metadata
        """
        from models import Challenge

        # Check cache first
        if self._max_cache_size > 0 and task_id in self._challenge_cache:
            # Move to end (mark as recently used)
            self._challenge_cache.move_to_end(task_id)
            return self._challenge_cache[task_id]

        # Decode task_id to get task_type and seed
        task_type, seed = self.decode_task_id(task_id)

        # Get task instance
        task = self._get_task_instance(task_type)

        # Get config (user config or default)
        config = {
            **self.SUPPORTED_TASKS[task_type]["default_config"],
            **self.task_configs.get(task_type, {})
        }

        # Generate single question with seed
        game_data = task.generate(seed=seed, **config)

        challenge = Challenge(
            env="logic-v2",
            prompt=game_data.question,
            extra={
                "task_id": task_id,
                "task_type": task_type,
                "seed": seed,
                "game_data": game_data.to_json(),
                "metadata": game_data.metadata
            }
        )

        # Store in cache (LRU eviction)
        if self._max_cache_size > 0:
            self._challenge_cache[task_id] = challenge
            # Evict oldest if cache is full
            if len(self._challenge_cache) > self._max_cache_size:
                self._challenge_cache.popitem(last=False)

        return challenge

    async def evaluate(self, response: str, challenge):
        """
        Evaluate response using task-specific verifier

        Args:
            response: Model response
            challenge: Original challenge

        Returns:
            Score (0.0 or 1.0)
        """
        task_type = challenge.extra.get("task_type")
        game_data_json = challenge.extra.get("game_data")

        if not task_type or not game_data_json:
            raise ValueError("Challenge missing task_type or game_data")

        # Get verifier class
        verifier_cls = verifier_classes.get(task_type)
        if verifier_cls is None:
            raise ValueError(f"Verifier class not found for task: {task_type}")

        verifier = verifier_cls()
        data_obj = Data.from_json(game_data_json)

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
