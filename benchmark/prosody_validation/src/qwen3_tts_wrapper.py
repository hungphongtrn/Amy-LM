"""
Qwen3-TTS Wrapper for speech generation with prosody control using qwen-tts library
"""

import asyncio
import logging
import random
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm.asyncio import tqdm_asyncio

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

logger = logging.getLogger(__name__)

# Available speakers
SPEAKERS = ["Ryan", "Aiden"]


class Qwen3TTSWrapper:
    """Wrapper for Qwen3-TTS speech generation with prosody guidance using qwen-tts library."""

    def __init__(
        self,
        repo_path: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 10,
    ):
        """Initialize the Qwen3-TTS wrapper.

        Args:
            repo_path: Path to the Qwen3-TTS model (defaults to Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
            device: Device to run inference on ('cuda' or 'cpu')
            batch_size: Number of texts to process in a batch
        """
        self.repo_path = repo_path or "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        self.device = device
        self.batch_size = batch_size
        self.model = None

    async def load_model(self) -> bool:
        """Load the Qwen3-TTS model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            logger.info(
                f"Loading Qwen3-TTS model from {self.repo_path} on {self.device}..."
            )

            # Map device string to device_map format
            device_map = self.device if self.device.startswith("cuda") else "cpu"

            self.model = Qwen3TTSModel.from_pretrained(
                self.repo_path,
                device_map=device_map,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            )

            logger.info("Qwen3-TTS model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load Qwen3-TTS model: {e}")
            import traceback

            traceback.print_exc()
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
            Prosody instruction string for TTS in English
        """
        # Normalize emotion for prosody mapping
        emotion_lower = emotion.lower().strip() if emotion else "neutral"

        # Prosody mappings based on emotion (original English)
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

        # Build instruction in English (original format)
        instruction = (
            f"Speak with a {prosody}. This is a {speech_act} expressing {intent}."
        )

        return instruction

    def generate_speech_batch(
        self,
        items: List[Dict[str, Any]],
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        """Generate speech for a batch of items using batched inference.

        Args:
            items: List of dictionaries containing 'dialog_id', 'text', 'emotion', 'speech_act', 'intent'
            output_dir: Directory to save generated audio files

        Returns:
            List of results with dialog_id, audio_path, and success status
        """
        try:
            # Prepare batched inputs
            texts = []
            instructs = []
            speakers = []
            output_paths = []
            dialog_ids = []
            existing_results = []

            for item in items:
                dialog_id = item["dialog_id"]
                text = item["text"]
                emotion = item.get("emotion", "neutral")
                speech_act = item.get("speech_act", "statement")
                intent = item.get("intent", "inform")

                output_path = output_dir / f"{dialog_id}.wav"

                # Skip if audio already exists
                if output_path.exists():
                    logger.debug(f"Audio already exists: {output_path}")
                    existing_results.append(
                        {
                            "dialog_id": dialog_id,
                            "audio_path": str(output_path),
                            "success": True,
                        }
                    )
                    continue

                output_paths.append(output_path)
                dialog_ids.append(dialog_id)

                # Generate prosody instruction
                prosody_instruction = self.generate_prosody_instruction(
                    emotion, speech_act, intent
                )

                texts.append(text)
                instructs.append(prosody_instruction)
                # Random speaker selection
                speakers.append(random.choice(SPEAKERS))

            # If all files exist, return early
            if not texts:
                return existing_results

            # Generate speech in batch
            wavs, sr = self.model.generate_custom_voice(
                text=texts,
                language="Auto",
                speaker=speakers,
                instruct=instructs,
            )

            # Save each audio file
            results = existing_results
            for i, (dialog_id, output_path) in enumerate(zip(dialog_ids, output_paths)):
                if i < len(wavs):
                    sf.write(str(output_path), wavs[i], sr)
                    results.append(
                        {
                            "dialog_id": dialog_id,
                            "audio_path": str(output_path),
                            "success": True,
                        }
                    )
                    logger.debug(f"Generated speech: {output_path}")
                else:
                    results.append(
                        {
                            "dialog_id": dialog_id,
                            "audio_path": None,
                            "success": False,
                        }
                    )

            return results

        except Exception as e:
            logger.error(f"Error generating speech batch: {e}")
            import traceback

            traceback.print_exc()
            return [
                {
                    "dialog_id": item["dialog_id"],
                    "audio_path": None,
                    "success": False,
                }
                for item in items
            ]

    async def generate_batch(
        self,
        texts: List[Dict[str, Any]],
        output_dir: Path,
        progress_desc: str = "Generating speech",
    ) -> List[Dict[str, Any]]:
        """Generate speech for all texts with batching.

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

        # Process in batches sequentially
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_results = self.generate_speech_batch(batch, output_dir)
            results.extend(batch_results)
            progress_bar.update(len(batch))

        progress_bar.close()

        # Log summary
        success_count = sum(1 for r in results if r.get("success"))
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
            del self.model
            self.model = None

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
