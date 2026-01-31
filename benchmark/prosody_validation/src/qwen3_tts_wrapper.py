"""
Qwen3-TTS Wrapper for speech generation with prosody control
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)


class Qwen3TTSWrapper:
    """Wrapper for Qwen3-TTS speech generation with prosody guidance."""

    def __init__(
        self,
        repo_path: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 10,
    ):
        """Initialize the Qwen3-TTS wrapper.

        Args:
            repo_path: Path to the Qwen3-TTS repository (defaults to config)
            device: Device to run inference on ('cuda' or 'cpu')
            batch_size: Number of texts to process in a batch
        """
        self.repo_path = repo_path
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.processor = None

        # Try to import Qwen3-TTS dependencies
        self._check_dependencies()

    def _check_dependencies(self) -> bool:
        """Check if Qwen3-TTS dependencies are available.

        Returns:
            True if dependencies are available, False otherwise
        """
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForTextToWaveform

            logger.info("Qwen3-TTS dependencies found")
            return True
        except ImportError as e:
            logger.warning(f"Qwen3-TTS dependencies not available: {e}")
            logger.info("Install with: pip install torch transformers")
            return False

    async def load_model(self) -> bool:
        """Load the Qwen3-TTS model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not self._check_dependencies():
                logger.error("Qwen3-TTS dependencies not available")
                return False

            import torch
            from transformers import AutoProcessor, AutoModelForTextToWaveform

            logger.info(f"Loading Qwen3-TTS model on {self.device}...")

            # Use default model if not specified
            model_id = self.repo_path or "Qwen/Qwen3-TTS"

            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForTextToWaveform.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device if self.device == "cuda" else None,
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            self.model.eval()
            logger.info("Qwen3-TTS model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load Qwen3-TTS model: {e}")
            return False

    def generate_prosody_instruction(
        self, emotion: str, speech_act: str, intent: str
    ) -> str:
        """Generate a prosody instruction based on metadata.

        Args:
            emotion: Emotion label (e.g., "angry", "happy", "sad")
            speech_act: Speech act type (e.g., "question", "statement", "command")
            intent: Intent label (e.g., "request", "complaint", "praise")

        Returns:
            Prosody instruction string for TTS
        """
        # Normalize emotion for prosody mapping
        emotion_lower = emotion.lower().strip() if emotion else "neutral"

        # Prosody mappings based on emotion
        prosody_map = {
            "angry": "firm and agitated tone",
            "frustrated": "impatient and irritated tone",
            "annoyed": "slightly irritated tone",
            "excited": "energetic and enthusiastic tone",
            "happy": "cheerful and warm tone",
            "sad": "soft and melancholic tone",
            "anxious": "nervous and rushed tone",
            "neutral": "calm and neutral tone",
            "positive": "warm and friendly tone",
            "negative": "cold and distant tone",
        }

        prosody = prosody_map.get(emotion_lower, "calm and neutral tone")

        # Build instruction
        instruction = (
            f"Speak with a {prosody}. This is a {speech_act} expressing {intent}."
        )

        return instruction

    async def generate_speech(
        self, text: str, prosody_instruction: str, output_path: Path
    ) -> bool:
        """Generate speech from text with prosody guidance.

        Args:
            text: Text to synthesize
            prosody_instruction: Prosody guidance instruction
            output_path: Path to save the generated audio

        Returns:
            True if successful, False otherwise
        """
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForTextToWaveform

            # Prepare input
            messages = [{"role": "user", "content": f"{prosody_instruction}\n\n{text}"}]

            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text_input],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate speech
            with torch.no_grad():
                audio = self.model.generate(**inputs)

            # Normalize and save audio
            audio = audio.float()  # Convert to float if needed
            audio = audio / max(audio.abs().max(), 1e-8)  # Normalize

            # Save as WAV
            import soundfile as sf

            sf.write(str(output_path), audio.cpu().numpy(), 22050)

            logger.debug(f"Generated speech: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return False

    async def generate_batch(
        self,
        texts: List[Dict[str, Any]],
        output_dir: Path,
        progress_desc: str = "Generating speech",
    ) -> List[Dict[str, Any]]:
        """Generate speech for a batch of texts.

        Args:
            texts: List of dictionaries containing 'dialog_id', 'text', 'emotion', 'speech_act', 'intent'
            output_dir: Directory to save generated audio files
            progress_desc: Description for progress bar

        Returns:
            List of results with dialog_id, audio_path, and success status
        """
        if not self.model:
            if not await self.load_model():
                logger.error("Failed to load model")
                return []

        results = []
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create progress bar
        progress_bar = tqdm_asyncio(total=len(texts), desc=progress_desc, unit="audio")

        for item in texts:
            dialog_id = item["dialog_id"]
            text = item["text"]
            emotion = item.get("emotion", "neutral")
            speech_act = item.get("speech_act", "statement")
            intent = item.get("intent", "inform")

            output_path = output_dir / f"{dialog_id}.wav"

            # Skip if audio already exists
            if output_path.exists():
                logger.debug(f"Audio already exists: {output_path}")
                results.append(
                    {
                        "dialog_id": dialog_id,
                        "audio_path": str(output_path),
                        "success": True,
                    }
                )
                progress_bar.update(1)
                continue

            # Generate prosody instruction
            prosody_instruction = self.generate_prosody_instruction(
                emotion, speech_act, intent
            )

            # Generate speech
            success = await self.generate_speech(text, prosody_instruction, output_path)

            results.append(
                {
                    "dialog_id": dialog_id,
                    "audio_path": str(output_path) if success else None,
                    "success": success,
                }
            )

            progress_bar.update(1)

        progress_bar.close()

        # Log summary
        success_count = sum(1 for r in results if r["success"])
        logger.info(
            f"Generated {success_count}/{len(results)} audio files successfully"
        )

        return results

    async def generate_from_manifest(
        self, manifest_path: Path, output_dir: Path
    ) -> List[Dict[str, Any]]:
        """Generate speech from a manifest CSV file.

        Args:
            manifest_path: Path to the input manifest CSV
            output_dir: Directory to save generated audio files

        Returns:
            List of results
        """
        from .utils import read_csv

        # Read manifest
        manifest = read_csv(manifest_path)
        logger.info(f"Loaded {len(manifest)} items from manifest")

        # Prepare items for batch processing
        items = []
        for row in manifest:
            items.append(
                {
                    "dialog_id": row["dialog_id"],
                    "text": row["rewritten_text"],
                    "emotion": row.get("emotion", "neutral"),
                    "speech_act": row.get("speech_act", "statement"),
                    "intent": row.get("intent", "inform"),
                }
            )

        return await self.generate_batch(
            items, output_dir, "Generating prosody-guided speech"
        )

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model:
            import torch

            del self.model
            del self.processor
            self.model = None
            self.processor = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Qwen3-TTS model unloaded")


async def create_tts_wrapper() -> Qwen3TTSWrapper:
    """Factory function to create a Qwen3-TTS wrapper."""
    from .config import get_config

    config = get_config()
    wrapper = Qwen3TTSWrapper(
        repo_path=config.QWEN3_TTS_REPO_PATH,
        device=config.QWEN3_TTS_DEVICE,
        batch_size=config.BATCH_SIZE_TTS,
    )
    return wrapper


if __name__ == "__main__":
    import sys

    async def test_wrapper():
        wrapper = await create_tts_wrapper()
        if await wrapper.load_model():
            print("Qwen3-TTS wrapper initialized successfully")
            wrapper.unload_model()
            return True
        else:
            print("Failed to initialize Qwen3-TTS wrapper")
            return False

    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(test_wrapper())
    sys.exit(0 if result else 1)
