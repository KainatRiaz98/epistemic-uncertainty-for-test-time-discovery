import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from ug_ttt import renderers
from ug_ttt.rl.problem_env import ProblemEnv
from ug_ttt.rl.types import Action, StepResult
from ug_ttt.utils import logtree

from ug_ttt.recipes.ttt.state import State
from ug_ttt.recipes.ttt.sampler import StateSampler
from ug_ttt.recipes.ttt.dataset_builder import DatasetConfig


# Shared ThreadPoolExecutor for all environments
SAFE_GRADE_MAX_WORKERS = 4096
SAFE_GRADE_EXECUTOR = ThreadPoolExecutor(max_workers=SAFE_GRADE_MAX_WORKERS)

logger = logging.getLogger(__name__)


def last_codeblock_postprocess(input_text, codeblock_seps=['python', 'cpp', 'java', 'cuda'], last_response_strict=True, keep_separators=True):
    """Extract the last code block from input text.
    
    Args:
        input_text: Text to parse
        codeblock_seps: List of language identifiers to look for
        last_response_strict: If True, return empty string for invalid code; otherwise return original text
        keep_separators: If True, return code with ```language wrapper; if False, return code only
    """
    languages_pattern = '|'.join(map(re.escape, codeblock_seps))
    codeblock_start = f'```({languages_pattern})'
    pattern = re.compile(codeblock_start + r'\n(?!```)(.*?)(?:\n```)?(?=\n```|$)', re.DOTALL)
    matches = list(pattern.finditer(input_text))

    if matches:
        last_match = matches[-1]
        language = last_match.group(1)
        code_content = last_match.group(2).rstrip()
        
        # Check if content is empty
        if not code_content or code_content.strip() == '':
            if last_response_strict:
                return ''
            else:
                return input_text
        
        if keep_separators:
            return f'```{language}\n{code_content}\n```'
        else:
            return code_content
    else:
        if last_response_strict:
            return ''
        else:
            return input_text


class BaseTTTEnv(ProblemEnv, ABC):
    """Abstract base class for TTT (Test-Time Training) environments.
    
    This class provides common functionality for all TTT environments, including:
    - Common initialization pattern
    - Code parsing and validation
    - Async grading infrastructure
    - Step processing with state updates
    - Common logging patterns
    """
    
    def __init__(
        self,
        renderer: renderers.Renderer,
        initial_state: State,
        sampler: StateSampler,
        config: DatasetConfig,
    ):
        super().__init__(renderer, convo_prefix=config.convo_prefix)
        
        if initial_state is None:
            raise ValueError("initial_state is required and cannot be None")
        if sampler is None:
            raise ValueError("sampler is required and cannot be None")
        
        self.config = config
        self.timeout = config.timeout
        self.num_cpus_per_task = config.num_cpus_per_task
        self.eval_timeout = config.eval_timeout
        self.log_path = config.log_path
        self.initial_state = initial_state
        self.sampler = sampler
        self.state = initial_state
        self.problem_idx = config.problem_idx
        self.adv_estimator = getattr(config, 'adv_estimator', None)
        self.budget_s = getattr(config, 'budget_s', None)
    
    @abstractmethod
    def _get_improvement_prompt(self, state: State) -> str:
        """Build the improvement prompt for the given state.
        
        Args:
            state: Current state containing code, value, and other context
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def _verify_code(
        self,
        generation: str,
        step: int,
        **kwargs
    ) -> dict[str, Any]:
        """Verify/evaluate the generated code.
        
        Args:
            generation: The code to verify
            step: Current step number
            **kwargs: Environment-specific parameters
            
        Returns:
            Dictionary with at least: score, msg, correctness, performance
            May also include: stdout, result_construction, and other env-specific fields
        """
        pass
    
    @abstractmethod
    def _compute_reward(self, outs: dict[str, Any], correctness: float) -> float:
        """Compute reward from verification outputs.
        
        Args:
            outs: Output dictionary from _verify_code
            correctness: Correctness score (0.0 to 1.0)
            
        Returns:
            Reward value
        """
        pass
    
    @abstractmethod
    def _create_next_state(
        self,
        step_idx: int,
        parsed_code: str,
        outs: dict[str, Any],
    ) -> State:
        """Create the next state from the current step.
        
        Args:
            step_idx: Current step index
            parsed_code: Parsed code from response
            outs: Output dictionary from _verify_code
            
        Returns:
            New State object
        """
        pass
    
    @abstractmethod
    def _build_metrics(
        self,
        outs: dict[str, Any],
        correct_format: bool,
        message: dict,
        parsed_code: str,
    ) -> dict[str, Any]:
        """Build metrics dictionary for StepResult.
        
        Args:
            outs: Output dictionary from _verify_code
            correct_format: Whether the code format was valid
            message: Parsed message from renderer
            parsed_code: Parsed code string
            
        Returns:
            Metrics dictionary
        """
        pass
    
    def _get_code_languages(self) -> list[str]:
        """Return list of code block languages to parse. Override if needed."""
        return ["python"]
    
    def _should_keep_code_separators(self) -> bool:
        """Whether to keep ```language separators in parsed code. Override if needed."""
        return True
    
    def _get_verify_kwargs(self) -> dict[str, Any]:
        """Get keyword arguments to pass to _verify_code. Override if needed."""
        return {
            "num_cpus_per_task": self.num_cpus_per_task,
            "eval_timeout": self.eval_timeout,
            "log_path": self.log_path,
            "state": self.state,
        }
    
    def get_question(self) -> str:
        """Build prompt from template, injecting previous code from state."""
        return self._get_improvement_prompt(self.initial_state)
    
    def check_format(self, parsed_code: str) -> bool:
        """Check if parsed code has valid format."""
        if (parsed_code is None) or (parsed_code.strip() == ''):
            return False
        return True
    
    async def check_answer(self, parsed_code: str, step: int) -> dict[str, Any]:
        """Check answer asynchronously with timeout."""
        if (parsed_code is None) or (parsed_code.strip() == ''):
            return {
                "sum_radii": 0.0,
                "score": 0.0,
                "msg": "Invalid code",
                "correctness": 0.0,
                "performance": 0.0,
            }
        
        return await self._safe_grade(parsed_code, step)
    
    async def _safe_grade(self, given_answer: str, step: int) -> dict[str, Any]:
        """Async grader: runs _verify_code in a background thread with asyncio timeout."""
        loop = asyncio.get_running_loop()
        start_time = time.time()
        verify_kwargs = self._get_verify_kwargs()
        
        try:
            out = await asyncio.wait_for(
                loop.run_in_executor(
                    SAFE_GRADE_EXECUTOR,
                    partial(
                        self._verify_code,
                        given_answer,
                        step,
                        **verify_kwargs
                    )
                ),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.warning(f"Timeout grading: took {elapsed:.1f}s, limit was {self.timeout:.1f}s")
            return self._get_timeout_response()
        except Exception as e:
            logger.warning(f"Exception while grading: {e}")
            return self._get_error_response(str(e))
        
        return out
    
    def _get_timeout_response(self) -> dict[str, Any]:
        """Get default timeout response. Override if needed."""
        return {
            "score": 0.0,
            "msg": "Timeout grading",
            "correctness": 0.0,
            "performance": 0.0,
        }
    
    def _get_error_response(self, error_msg: str) -> dict[str, Any]:
        """Get default error response. Override if needed."""
        return {
            "score": 0.0,
            "msg": f"Error grading: {error_msg}",
            "correctness": 0.0,
            "performance": 0.0,
        }
    
    async def step(self, action: Action, step_idx: int) -> StepResult:
        """Process a step: parse response, verify code, compute reward, update state."""
        message, parse_success = self.renderer.parse_response(action)
        response = message["content"]
        
        # Parse code based on environment-specific settings
        languages = self._get_code_languages()
        keep_separators = self._should_keep_code_separators()
        parsed_code = last_codeblock_postprocess(
            response,
            codeblock_seps=languages,
            keep_separators=keep_separators
        )
        correct_format = float(parse_success) and float(self.check_format(parsed_code))
        
        # Verify code
        outs = await self.check_answer(parsed_code, step_idx)
        score = outs.get("score", 0.0)
        correctness = outs.get("correctness", 0.0)
        performance = outs.get("performance")
        msg = outs.get("msg", "")
        
        # Compute reward
        reward = self._compute_reward(outs, correctness)
        
        # Logging
        logtree.log_text(f"Problem: {self.get_question()[:200]}...")
        logtree.log_text(f"Response: {message['content']}")
        logtree.log_text(
            f"Format Valid: {'✓' if correct_format else '✗'}, "
            f"Score: {score:.4f}, Performance: {performance}, Reward: {reward:.4f}, Msg: {msg}"
        )
        
        # Build metrics
        metrics = self._build_metrics(outs, correct_format, message, parsed_code)
        
        # Create step result
        step_result = StepResult(
            reward=reward,
            episode_done=True,
            next_observation=None,
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )

        # Update sampler with new state if we have valid result
        if correctness > 0:
            try:
                next_state = self._create_next_state(step_idx, parsed_code, outs)
                self.sampler.update_states([next_state], [self.initial_state], save=False)
            except Exception as e:
                logger.warning(f"Failed to create next state: {e}")
                if hasattr(self.sampler, 'record_failed_rollout'):
                    self.sampler.record_failed_rollout(self.initial_state)
        elif hasattr(self.sampler, 'record_failed_rollout'):
            # Record that we tried this parent but got no valid child (for PUCT visit counts)
            self.sampler.record_failed_rollout(self.initial_state)
        
        return step_result
    
    def get_reference_answer(self) -> str:
        """Return the reference answer for logging purposes."""
        raise NotImplementedError("Reference answer not available for TTT environments.")
