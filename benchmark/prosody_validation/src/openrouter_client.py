"""
Async OpenRouter Client for LLM API calls
"""

import asyncio
import base64
import json
import logging
from typing import Optional, Dict, Any, List, Type, TypeVar
from pathlib import Path

import aiohttp
from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .config import get_config

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Async client for OpenRouter API with support for text and multimodal calls."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key (defaults to env var)
            base_url: Base URL for OpenRouter API
        """
        config = get_config()
        self.api_key = api_key or config.OPENROUTER_API_KEY
        self.base_url = base_url or config.OPENROUTER_BASE_URL

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable."
            )

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        # Semaphore for concurrency control
        self.semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_LLM)

    async def close(self):
        """Close the async client."""
        await self.client.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def chat_text(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Send a text-only prompt to the LLM.

        Args:
            prompt: The user prompt
            model: Model name to use
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        async with self.semaphore:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=60.0,
                )

                content = response.choices[0].message.content
                logger.debug(f"LLM response: {content[:100]}...")
                return content

            except Exception as e:
                logger.error(f"Error in chat_text: {e}")
                raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def chat_with_audio(
        self,
        audio_path: Path,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Send an audio file with prompt to the multimodal LLM.

        Args:
            audio_path: Path to the audio file
            prompt: The user prompt
            model: Model name to use (must support audio)
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        async with self.semaphore:
            # Read and encode audio file
            with open(audio_path, "rb") as f:
                audio_data = f.read()
                audio_base64 = base64.b64encode(audio_data).decode("utf-8")

            # Determine mime type based on extension
            mime_type = "audio/wav"
            if audio_path.suffix.lower() == ".mp3":
                mime_type = "audio/mp3"
            elif audio_path.suffix.lower() == ".flac":
                mime_type = "audio/flac"
            elif audio_path.suffix.lower() == ".ogg":
                mime_type = "audio/ogg"

            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_base64,
                                "format": mime_type.split("/")[1],
                            },
                        },
                    ],
                }
            )

            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=120.0,  # Longer timeout for multimodal
                )

                content = response.choices[0].message.content
                logger.debug(f"Multimodal LLM response: {content[:100]}...")
                return content

            except Exception as e:
                logger.error(f"Error in chat_with_audio: {e}")
                raise

    async def chat_batch(
        self,
        prompts: List[Dict[str, Any]],
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> List[str]:
        """Process multiple prompts concurrently.

        Args:
            prompts: List of prompt dictionaries with 'prompt' and optional 'id' key
            model: Model name to use
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            List of responses in the same order as inputs
        """
        tasks = []
        for item in prompts:
            task = self.chat_text(
                prompt=item["prompt"],
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Error processing prompt {i}: {response}")
                results.append(f"ERROR: {str(response)}")
            else:
                results.append(response)

        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def chat_structured(
        self,
        prompt: str,
        model: str,
        response_format: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> T:
        """Send a text prompt and get structured output using JSON schema.

        Args:
            prompt: The user prompt
            model: Model name to use (must support structured output)
            response_format: Pydantic model class defining the output structure
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Parsed Pydantic model instance
        """
        async with self.semaphore:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            try:
                # Get JSON schema from Pydantic model
                schema = response_format.model_json_schema()

                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=60.0,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_format.__name__,
                            "schema": schema,
                            "strict": True,
                        },
                    },
                )

                content = response.choices[0].message.content
                logger.debug(f"Structured LLM response: {content[:100]}...")

                # Parse JSON and validate with Pydantic
                parsed_data = json.loads(content)
                return response_format(**parsed_data)

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                raise
            except Exception as e:
                logger.error(f"Error in chat_structured: {e}")
                raise


async def create_client() -> OpenRouterClient:
    """Factory function to create an OpenRouter client."""
    return OpenRouterClient()


async def test_client():
    """Test the OpenRouter client with a simple prompt."""
    client = await create_client()

    try:
        response = await client.chat_text(
            prompt="Say 'Hello, OpenRouter!' in exactly these words.",
            model="openai/gpt-4o-mini",
        )
        print(f"Test response: {response}")
        return response
    finally:
        await client.close()


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    result = asyncio.run(test_client())
    sys.exit(0 if result else 1)
