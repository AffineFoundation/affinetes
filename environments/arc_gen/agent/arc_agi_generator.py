from typing import List, Dict, Any, Tuple, Optional
import random

import agent.arc_agi2_utils as utils
import agent.task_list as task_list


def _count_non_black(grid: List[List[int]]) -> int:
    return sum(1 for row in grid for v in row if v != 0)


class ARC2Generator:
    """
    Generate ARC-AGI-2 style problems by applying transformation chains to ARC-1 base tasks.
    
    Key approach:
      - Start from a base ARC-1 task (input -> output)
      - Apply a parameterized chain of transforms to the *output*
      - The same chain (with frozen parameters) is reused for all examples
    
    This creates problems requiring compositional reasoning while remaining
    solvable by humans who can infer the transformation pattern.
    """

    def __init__(
        self,
        max_chain_length: int = 4,
        max_grid_size: int = 30,
        task_range: int=100_000_000
    ):
        self.max_chain_length = max_chain_length
        self.max_grid_size = max_grid_size

        # Quality thresholds to keep outputs meaningful
        self.min_distinct_colors = 2
        self.min_non_black_cells = 6
        self.task_range = task_range
    def generate_initial_problem(
        self,
        task_num: Optional[int] = None,
        rng = None,
    ) -> Dict[str, Any]:
        """
        Generate a base ARC style problem from task_list.
        
        Returns:
            Dict with "input", "output", and "task_num" keys
        """
        max_retry = 10
        while max_retry > 0:
            try:
                tmap = task_list.task_list()
                
                keys = list(tmap.keys())
                
                if task_num is None:
                    task_num = random.randint(0 , len(keys) - 1)
                

                _, gen_fn, _ = tmap[keys[task_num]]
                pair = gen_fn(rng = rng)
                if isinstance(pair, dict):
                    inp = pair["input"]
                    out = pair["output"]
                else:
                    inp, out = pair

                if not utils.is_valid_grid(inp) or not utils.is_valid_grid(out):
                    max_retry -= 1
                    continue

                h, w = utils.get_grid_size(out)
                if h > self.max_grid_size or w > self.max_grid_size:
                    max_retry -= 1
                    continue

                return {"input": inp, "output": out, "task_num": task_num}
            except Exception as e:
                continue

        raise ValueError(f"Generation Error")


    def _sample_params(self, name: str, grid: List[List[int]] , rng = None) -> Optional[Dict[str, Any]]:
        """
        Sample parameters for a transformation based on current grid state.
        Returns None if the transform doesn't need parameters or can't be applied.
        """
        colors_present = list(utils.get_colors_in_grid(grid) - {0})
        palette = list(range(1, 10))  # Colors 1-9

        if name == "swap_colors":
            if len(colors_present) >= 2:
                c1, c2 = rng.sample(colors_present, 2)
            elif len(colors_present) == 1:
                c1 = colors_present[0]
                c2 = rng.choice([c for c in palette if c != c1])
            else:
                c1, c2 = 1, 2
            return {"color1": c1, "color2": c2}

        if name == "remove_color":
            if len(colors_present) <= 1:
                return None
            return {"color": rng.choice(colors_present)}

        if name == "highlight_color":
            if not colors_present:
                return None
            return {"color": rng.choice(colors_present)}

        if name == "shift":
            direction = rng.choice(["up", "down", "left", "right"])
            h, w = utils.get_grid_size(grid)
            span = h if direction in ("up", "down") else w
            max_amt = max(1, span - 1)
            amt = rng.randint(1, min(3, max_amt))
            return {"direction": direction, "amount": amt, "wrap": False}

        return None

    def select_transformation_chain(
        self,
        grid: List[List[int]],
        chain_length: Optional[int] = None,
        rng = None
    ) -> List[Dict[str, Any]]:
        """
        Build a parameterized transformation chain.
        
        Each transform is applied to the current grid state to ensure compatibility,
        and parameters are frozen for reuse across examples.
        """
        if chain_length is None:
            chain_length = self.max_chain_length

        result_chain = []
        cur = utils.deep_copy_grid(grid)
        compatible = utils.get_compatible_transformations(cur, max_size=self.max_grid_size)
        ### rotate
        rotate = rng.choice(["rotate_90" , "rotate_180" , "rotate_180" , None])
        if rotate != None:
            result_chain.append({"name": rotate, "params": None})
        
        ### transpose
        if rng.random() > 0.5:
            result_chain.append({"name": "transpose", "params": None})
        
        ### recenter
        if rng.random() > 0.9:
            result_chain.append({"name": "recenter", "params": None})
            
        ### swap colors
        for _ in range(chain_length):
            if rng.random() > 0.7:
                params = self._sample_params("swap_colors", cur , rng)
                new_cur = utils.apply_transformation(cur, "swap_colors", params)
                if not utils.is_valid_grid(new_cur):
                    continue
                result_chain.append({"name": "swap_colors", "params": params})
                cur = new_cur
            else:
                break
            
        ### zoom                        
        if rng.random() > 0.95:
            if "zoom_3x" in compatible:
                result_chain.append({"name": "zoom_3x", "params": None})
        elif rng.random() > 0.9:
            if "zoom_2x" in compatible:
                result_chain.append({"name": "zoom_2x", "params": None})

        return result_chain

    def apply_transformation_chain(
        self,
        grid: List[List[int]],
        chain: List[Dict[str, Any]],
    ) -> List[List[int]]:
        """Apply a frozen transformation chain to a grid."""
        out = utils.deep_copy_grid(grid)
        for step in chain:
            name = step["name"]
            params = step.get("params")
            out = utils.apply_transformation(out, name, params)
            if not utils.is_valid_grid(out):
                raise ValueError(f"Invalid grid after transform: {name}")
        return out

    def _non_degenerate(self, grid: List[List[int]]) -> bool:
        """Check if grid has sufficient complexity."""
        distinct_colors = utils.get_colors_in_grid(grid) - {0}
        if len(distinct_colors) < self.min_distinct_colors:
            return False
        if _count_non_black(grid) < self.min_non_black_cells:
            return False
        
        return True

    def generate_problem_set(
        self,
        task_id: Optional[int] = 0,
        num_train: int = 3,
        chain_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate train/test examples with the same transformation chain.
        
        This mirrors the ARC-AGI format where the same underlying rule
        applies across all examples.
        
        Returns:
            {
              "train_examples": [{"input": <grid>, "output": <grid>}, ...],
              "test_input": <grid>,
              "test_output": <grid>,  # For validation
              "metadata": {
                  "task_id": <int>,
                  "transformation_chain": [...],
                  "chain_length": int
              }
            }
        """
        task_num = task_id // self.task_range
        seed = task_id % self.task_range
        
        rng = random.Random(seed)
        base_initial = self.generate_initial_problem(task_num, rng)
        task_num = base_initial["task_num"]
        
        # Generate transformation chain (empty if chain_length is 0)
        if chain_length == 0:
            chain = []
        else:
            chain = self.select_transformation_chain(
                base_initial["output"],
                chain_length=chain_length,
                rng = rng
            )
        
        # Generate training examples
        train_examples = []
        attempts = 0
        max_attempts = num_train * 300
        
        is_completed = False
        limit_degenerate = 0
        while attempts < max_attempts:
            attempts += 1
            try:
                base = self.generate_initial_problem(task_num=task_num, rng = rng)
        
                if chain:
                    output = self.apply_transformation_chain(base["output"], chain)
                else:
                    output = base["output"]
                
                if limit_degenerate > 20 or self._non_degenerate(output):
                    if len(train_examples) < num_train:
                        train_examples.append({
                            "input": base["input"],
                            "output": output
                        })
                    else:
                        test_base = base
                        test_output = output
                        is_completed = True
                        break
                else:
                    limit_degenerate += 1
                    
            except Exception as e:
                if type(e).__name__ != "ValueError":
                    import traceback
                    error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                    print("Fail to generate\n", error)
        
        if not is_completed:
            raise ValueError(f"Fail to generate problem.")
        # Generate test example
        
        return {
            "train_examples": train_examples,
            "test_input": test_base["input"],
            "test_output": test_output,
            "metadata": {
                "task_id": task_id,
                "transformation_chain": chain,
                "chain_length": len(chain),
            }
        }