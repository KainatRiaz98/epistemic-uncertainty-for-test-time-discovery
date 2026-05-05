from types import SimpleNamespace
from typing import Any

from ug_ttt.utils import logtree

from tasks.erdos_min_overlap.task import ErdosMinOverlapTask
from tasks.erdos_min_overlap.prompt import SYSTEM_PROMPT
from ug_ttt.recipes.ttt.state import ErdosState
from ug_ttt.recipes.ttt.env_ttt import BaseTTTEnv
from ug_ttt.recipes.ttt.dataset_builder import DatasetConfig


def dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    return d


default_config = {
    "ttt_rm": {
        "num_cpus_per_task": 2,
        "rew_type": "linear",
        "fail_score": 0.0,
        "eval_timeout": 1100,
        "worst_perf_log": -10000,
        "n_item": 200,
    }
}


def verify_erdos(
    generation: str,
    step: int,
    num_cpus_per_task: int = 2,
    eval_timeout: int = 1000,
    log_path: str = "",
    state: ErdosState = None,
) -> dict:
    config = default_config.copy()
    config["ttt_rm"] = default_config["ttt_rm"].copy()
    config["ttt_rm"]["num_cpus_per_task"] = num_cpus_per_task
    config["ttt_rm"]["eval_timeout"] = eval_timeout
    config_ns = dict_to_ns(config)

    logtree.log_text(f"Starting gen, {config}")

    task = ErdosMinOverlapTask(config_ns, log_path)
    out = task.compute_score(generation, step=step, state=state)
    
    construction = None
    actual_c5 = None
    if out["correctness"] > 0 and "result_construction" in out and out["result_construction"] is not None:
        result = out["result_construction"]
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            construction = list(result[0]) if hasattr(result[0], '__iter__') else None
            actual_c5 = float(result[1])  # c5_bound is at index 1
    
    return {
        "score": out["score"],
        "msg": out["msg"],
        "correctness": out["correctness"],
        "performance": -actual_c5 if actual_c5 is not None else None,
        "c5_bound": actual_c5,
        "construction": construction,
        "stdout": out.get("stdout", ""),
    }


def _is_entropic_adv(adv_estimator: str | None) -> bool:
    return adv_estimator in ("entropic", "entropic_adaptive_beta")


def build_erdos_prompt(
    state: ErdosState,
    budget_s: int = 1000,
    num_cpus_per_task: int = 2,
    problem_idx: str = "200",
) -> str:
    """Build improvement prompt for Erdos tasks. Standalone version of ErdosMinOverlapEnv._get_improvement_prompt()."""
    hide_code = "state_only" in str(problem_idx)
    has_code = state.code and state.code.strip() and not hide_code

    if state.parent_values and state.value is not None:
        before_bound = -state.parent_values[0]
        after_bound = -state.value
        value_ctx = f"\nHere are the C₅ bounds before and after running the code above (lower is better): {before_bound:.6f} -> {after_bound:.6f}"
        value_ctx += f"\nOur target is to make the C₅ bound tighter, just as a reference, lower it to at least 0.3808."
    elif state.value is not None:
        value_ctx = f"\nCurrent C₅ bound (lower is better): {-state.value:.6f}"
        value_ctx += f"\nOur target is to make the C₅ bound tighter, just as a reference, lower it to at least 0.3808."
    elif state.c5_bound is not None:
        value_ctx = f"\nCurrent C₅ bound (lower is better): {state.c5_bound:.6f}"
        value_ctx += f"\nOur target is to make the C₅ bound tighter, just as a reference, lower it to at least 0.3808."
    else:
        value_ctx = "\nOur target is to make the C₅ bound tighter, just as a reference, lower it to at least 0.3808."

    if state.observation and state.observation.strip():
        stdout = state.observation.strip()
        if len(stdout) > 500:
            stdout = "...(truncated)\n" + stdout[-500:]
        value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"

    prompt = SYSTEM_PROMPT
    prompt = prompt.replace("<<<BUDGET_S>>>", str(budget_s))
    prompt = prompt.replace("<<<CPUS>>>", str(num_cpus_per_task))

    h_values_section = ""
    if hasattr(state, 'construction') and state.construction is not None and len(state.construction) > 0:
        h_values_section = f"""
You may want to start your search from the current construction, which you can access through the `initial_h_values` global variable (n={len(state.construction)} samples).
You are encouraged to explore solutions that use other starting points to prevent getting stuck in a local optimum.
"""

    if has_code:
        clean_code = state.code.strip()
        if clean_code.startswith("```python"):
            clean_code = clean_code[len("```python"):].strip()
        if clean_code.startswith("```"):
            clean_code = clean_code[3:].strip()
        if clean_code.endswith("```"):
            clean_code = clean_code[:-3].strip()
        code_section = f"""
Here is the last code we ran:
```python
{clean_code}
```

You are iteratively optimizing constructions.{value_ctx}

Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc.
Unless you make a meaningful improvement, you will not be rewarded.
"""
    else:
        code_section = f"""
{value_ctx}

Write code to optimize this construction.
"""

    return f"""{prompt}
{h_values_section}{code_section}"""


class ErdosMinOverlapEnv(BaseTTTEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _truncate_code(self, code: str, max_lines: int = 200) -> str:
        """Keep top and bottom lines, truncate middle."""
        lines = code.split('\n')
        if len(lines) <= max_lines:
            return code
        half = max_lines // 2
        return '\n'.join(lines[:half]) + "\n# ...(middle truncated)...\n" + '\n'.join(lines[-half:])

    def _get_improvement_prompt(self, state: ErdosState) -> str:
        """Build contextual improvement prompt with before/after values and stdout."""
        # state_only mode: show construction but not code (forces model to write fresh code)
        hide_code = "state_only" in self.problem_idx
        has_code = state.code and state.code.strip() and not hide_code
        
        # Value context: show before/after if we have parent values
        # state.value is -c5_bound (higher=better for RL), so negate to get c5_bound (lower=better)
        if state.parent_values and state.value is not None:
            before_bound = -state.parent_values[0]
            after_bound = -state.value
            value_ctx = f"\nHere are the C₅ bounds before and after running the code above (lower is better): {before_bound:.6f} -> {after_bound:.6f}"
            value_ctx += f"\nOur target is to make the C₅ bound tighter, just as a reference, lower it to at least 0.3808."
        elif state.value is not None:
            value_ctx = f"\nCurrent C₅ bound (lower is better): {-state.value:.6f}"
            value_ctx += f"\nOur target is to make the C₅ bound tighter, just as a reference, lower it to at least 0.3808."
        elif state.c5_bound is not None:
            value_ctx = f"\nCurrent C₅ bound (lower is better): {state.c5_bound:.6f}"
            value_ctx += f"\nOur target is to make the C₅ bound tighter, just as a reference, lower it to at least 0.3808."
        else:
            value_ctx = "\nOur target is to make the C₅ bound tighter, just as a reference, lower it to at least 0.3808."
        
        if state.observation and state.observation.strip():
            stdout = state.observation.strip()
            if len(stdout) > 500:
                stdout = "...(truncated)\n" + stdout[-500:]
            value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"
        
        prompt = SYSTEM_PROMPT
        prompt = prompt.replace("<<<BUDGET_S>>>", str(self.budget_s))
        prompt = prompt.replace("<<<CPUS>>>", str(self.num_cpus_per_task))
        
        h_values_section = ""
        if hasattr(state, 'construction') and state.construction is not None and len(state.construction) > 0:
            h_values_section = f"""
You may want to start your search from the current construction, which you can access through the `initial_h_values` global variable (n={len(state.construction)} samples).
You are encouraged to explore solutions that use other starting points to prevent getting stuck in a local optimum.
"""
        
        # Handle code section
        if has_code:
            clean_code = state.code.strip()
            if clean_code.startswith("```python"):
                clean_code = clean_code[len("```python"):].strip()
            if clean_code.startswith("```"):
                clean_code = clean_code[3:].strip()
            if clean_code.endswith("```"):
                clean_code = clean_code[:-3].strip()
            code_section = f"""
Here is the last code we ran:
```python
{clean_code}
```

You are iteratively optimizing constructions.{value_ctx}

Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc. 
Unless you make a meaningful improvement, you will not be rewarded.
"""
        else:
            code_section = f"""
{value_ctx}

Write code to optimize this construction.
"""
        
        return f"""{prompt}
{h_values_section}{code_section}"""

    def _verify_code(
        self,
        generation: str,
        step: int,
        num_cpus_per_task: int = 2,
        eval_timeout: int = 1000,
        log_path: str = "",
        state: ErdosState = None,
        **kwargs
    ) -> dict[str, Any]:
        return verify_erdos(generation, step, num_cpus_per_task, eval_timeout, log_path, state)

    def _get_verify_kwargs(self) -> dict[str, Any]:
        return {
            "num_cpus_per_task": self.num_cpus_per_task,
            "eval_timeout": self.eval_timeout,
            "log_path": self.log_path,
            "state": self.state,
        }

    def _get_timeout_response(self) -> dict[str, Any]:
        return {
            "score": 0.0,
            "msg": "Timeout grading",
            "correctness": 0.0,
            "performance": 0.0,
            "c5_bound": None,
            "construction": None,
            "stdout": "",
        }

    def _get_error_response(self, error_msg: str) -> dict[str, Any]:
        return {
            "score": 0.0,
            "msg": f"Error grading: {error_msg}",
            "correctness": 0.0,
            "performance": 0.0,
            "c5_bound": None,
            "construction": None,
            "stdout": "",
        }

    def _compute_reward(self, outs: dict[str, Any], correctness: float) -> float:
        performance = outs.get("performance")
        if _is_entropic_adv(self.adv_estimator):
            # Erdos: lower c5_bound is better, target=0.38092
            # performance = -c5_bound (actual bound)
            current_bound = -performance if performance is not None else float('inf')
            return 1/current_bound if (correctness > 0 and current_bound > 0) else 0.0
        else:
            return outs["score"]

    def _create_next_state(
        self,
        step_idx: int,
        parsed_code: str,
        outs: dict[str, Any],
    ) -> ErdosState:
        performance = outs.get("performance")
        if performance is None:
            return None
        parent_state = self.initial_state
        parent_values = [parent_state.value] + parent_state.parent_values if parent_state.value is not None else []
        return ErdosState(
            timestep=step_idx,
            code=parsed_code,
            value=performance,
            c5_bound=outs.get("c5_bound"),
            construction=outs.get("construction"),
            parent_values=parent_values,
            observation=outs.get("stdout", ""),
        )

    def _build_metrics(
        self,
        outs: dict[str, Any],
        correct_format: bool,
        message: dict,
        parsed_code: str,
    ) -> dict[str, Any]:
        score = outs["score"]
        correctness = outs["correctness"]
        c5_bound = outs.get("c5_bound")
        best_bound = c5_bound if correctness > 0 else None
        return {
            "format": correct_format,
            "score": score,
            "correctness": correctness,
            "correct": correctness,
            "c5_bound": c5_bound,
            "performance": best_bound,
            "performance/best": best_bound,
            "initial_performance": -self.initial_state.value if self.initial_state.value is not None else None,
            "msg": outs.get("msg", ""),
            "predicted_grid": None,
            "prompt": self.get_question(),
            "response": message['content'],
            "ref": outs.get("msg", ""),
        }


