"""
Qwen3-TTS Wrapper for speech generation with prosody control using qwen-tts library
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm.asyncio import tqdm_asyncio

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

logger = logging.getLogger(__name__)


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
            Prosody instruction string for TTS
        """
        # Normalize emotion for prosody mapping
        emotion_lower = emotion.lower().strip() if emotion else "neutral"

        # Prosody mappings based on emotion
        prosody_map = {
            "angry": "用愤怒的语气说",
            "frustrated": "用沮丧的语气说",
            "annoyed": "用恼怒的语气说",
            "excited": "用兴奋的语气说",
            "happy": "用开心的语气说",
            "sad": "用悲伤的语气说",
            "anxious": "用焦虑的语气说",
            "neutral": "用中性的语气说",
            "positive": "用积极的语气说",
            "negative": "用消极的语气说",
        }

        prosody = prosody_map.get(emotion_lower, "用中性的语气说")

        # Build instruction in Chinese for better compatibility with Qwen3-TTS
        instruction = f"{prosody}。这是一个{speech_act}，表达{intent}的意图。"

        return instruction

    async def generate_speech(
        self,
        text: str,
        prosody_instruction: str,
        output_path: Path,
        language: str = "Auto",
    ) -> bool:
        """Generate speech from text with prosody guidance.

        Args:
            text: Text to synthesize
            prosody_instruction: Prosody guidance instruction
            output_path: Path to save the generated audio
            language: Language code (default: "Auto" for auto-detection)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Use default speaker and apply prosody instruction
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language=language,
                speaker="Vivian",  # Default speaker
                instruct=prosody_instruction if prosody_instruction else None,
            )

            # Save audio
            sf.write(str(output_path), wavs[0], sr)

            logger.debug(f"Generated speech: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            import traceback

            traceback.print_exc()
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
