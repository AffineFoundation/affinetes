"""Number Guessing interactive environment"""

import os
import time
import httpx
import openai
import random
import re
from typing import Optional


class Actor:
    """Interactive Number Guessing game environment"""
    
    MIN_RANGE = 1
    MAX_RANGE = 1000
    MAX_ATTEMPTS = 10
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize actor with API key"""
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
    
    async def _llm_chat(self, messages, model, base_url, timeout, temperature, api_key, seed=None):
        """Call LLM API"""
        os.environ.pop('SSL_CERT_FILE', None)
        os.environ.pop('REQUESTS_CA_BUNDLE', None)
        
        client = openai.AsyncOpenAI(
            base_url=base_url.rstrip('/'),
            api_key=api_key,
            timeout=httpx.Timeout(timeout),
            max_retries=0
        )
        
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }
        
        if seed is not None:
            params["seed"] = seed
        
        response = await client.chat.completions.create(**params)

        if not response.choices:
            # Return None for empty choices (will result in 0 score)
            return None

        content = response.choices[0].message.content
        if content is None:
            # Return None for empty content (e.g., token limit exhausted during reasoning)
            # This will result in 0 score rather than raising an error
            return None

        return content.strip()
    
    def _parse_guess(self, response: str) -> Optional[int]:
        """Parse guess from LLM response"""
        numbers = re.findall(r'-?\d+', response)
        
        if numbers:
            try:
                return int(numbers[0])
            except ValueError:
                pass
        
        return None
    
    
    async def evaluate(
        self,
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        task_id: Optional[int] = None,
        timeout: int = 600,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """Play number guessing game interactively"""
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        current_api_key = api_key or self.api_key
        start = time.time()
        
        # Generate target number
        random.seed(task_id if task_id is not None else random.randint(0, 2**31 - 1))
        target = random.randint(self.MIN_RANGE, self.MAX_RANGE)
        
        # Initial prompt
        initial_prompt = f"""You are playing a number guessing game.

Rules:
- I have chosen a secret number between {self.MIN_RANGE} and {self.MAX_RANGE} (inclusive)
- You have {self.MAX_ATTEMPTS} attempts to guess the number
- After each guess, I will tell you if the secret number is higher or lower
- Try to find the number in as few attempts as possible

To make a guess, respond with just the number.
Example: "500"

What is your first guess?"""
        
        conversation = [{"role": "user", "content": initial_prompt}]
        
        attempts_used = 0
        success = False
        
        for attempt in range(self.MAX_ATTEMPTS):
            try:
                response = await self._llm_chat(
                    conversation, model, base_url, timeout, temperature, current_api_key, seed
                )
                conversation.append({"role": "assistant", "content": response})
                
                guess = self._parse_guess(response)
                if guess is None:
                    feedback = "Cannot parse your guess. Please respond with just a number.\n\nWhat is your guess?"
                    conversation.append({"role": "user", "content": feedback})
                    continue
                
                attempts_used += 1
                attempts_left = self.MAX_ATTEMPTS - attempts_used
                
                # Validate guess and generate feedback
                if guess == target:
                    success = True
                    feedback = f"Correct! You found the secret number {guess} in {attempts_used} attempts!"
                    conversation.append({"role": "user", "content": feedback})
                    break
                
                if attempts_left == 0:
                    feedback = f"Game over! You've used all {attempts_used} attempts.\nThe secret number was {target}."
                    conversation.append({"role": "user", "content": feedback})
                    break
                
                hint = "higher" if guess < target else "lower"
                feedback = f"""Your guess: {guess}
Result: The secret number is {hint} than {guess}.

Attempts remaining: {attempts_left}

What is your next guess?"""
                conversation.append({"role": "user", "content": feedback})
            
            except Exception as e:
                error_msg = f"Error in attempt {attempt + 1}: {str(e)}"
                conversation.append({"role": "user", "content": error_msg})
                break
        
        result = {
            "task_name": "game:number_guess",
            "score": 1.0 if success else 0.0,
            "success": success,
            "time_taken": time.time() - start,
            "extra": {
                "conversation": conversation,
                "seed": seed,
                "task_id": task_id
            }
        }
        
        return result