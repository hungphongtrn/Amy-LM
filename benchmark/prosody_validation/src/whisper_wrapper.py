"""
Whisper ASR Wrapper for local speech transcription
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)


class WhisperWrapper:
    """Wrapper for OpenAI Whisper speech recognition."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        language: Optional[str] = None,
        batch_size: int = 20,
    ):
        """Initialize the Whisper wrapper.

        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run inference on ('cpu' or 'cuda')
            language: Language code for transcription (e.g., 'en' for English)
            batch_size: Number of audio files to process in a batch
        """
        self.model_size = model_size
        self.device = device
        self.language = language
        self.batch_size = batch_size
        self.model = None

        # Try to import whisper
        self._check_dependencies()

    def _check_dependencies(self) -> bool:
        """Check if whisper dependencies are available.

        Returns:
            True if dependencies are available, False otherwise
        """
        try:
            import whisper
            import torch

            logger.info("Whisper dependencies found")
            return True
        except ImportError as e:
            logger.warning(f"Whisper dependencies not available: {e}")
            logger.info("Install with: pip install openai-whisper torch")
            return False

    def load_model(self) -> bool:
        """Load the Whisper model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not self._check_dependencies():
                logger.error("Whisper dependencies not available")
                return False

            import whisper

            logger.info(
                f"Loading Whisper '{self.model_size}' model on {self.device}..."
            )

            self.model = whisper.load_model(self.model_size, device=self.device)

            logger.info("Whisper model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            return False

    def transcribe_audio(
        self, audio_path: Path, verbose: bool = False
    ) -> Dict[str, Any]:
        """Transcribe a single audio file.

        Args:
            audio_path: Path to the audio file
            verbose: Whether to print verbose output

        Returns:
            Dictionary containing transcription results
        """
        try:
            import whisper
            import torch

            if not self.model:
                if not self.load_model():
                    return {
                        "success": False,
                        "error": "Model not loaded",
                        "audio_path": str(audio_path),
                    }

            # Ensure audio path exists
            if not audio_path.exists():
                return {
                    "success": False,
                    "error": "Audio file not found",
                    "audio_path": str(audio_path),
                }

            # Transcribe audio
            options = {
                "language": self.language,
                "verbose": verbose,
                "fp16": torch.cuda.is_available() and self.device == "cuda",
            }

            result = self.model.transcribe(str(audio_path), **options)

            return {
                "success": True,
                "text": result.get("text", "").strip(),
                "audio_path": str(audio_path),
                "language": result.get("language", self.language),
                "duration": result.get("duration", 0),
            }

        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {e}")
            return {"success": False, "error": str(e), "audio_path": str(audio_path)}

    async def transcribe_batch(
        self, audio_paths: List[Path], progress_desc: str = "Transcribing audio"
    ) -> List[Dict[str, Any]]:
        """Transcribe a batch of audio files.

        Args:
            audio_paths: List of paths to audio files
            progress_desc: Description for progress bar

        Returns:
            List of transcription results
        """
        if not self.model:
            if not self.load_model():
                logger.error("Failed to load Whisper model")
                return []

        results = []

        # Create progress bar
        progress_bar = tqdm_asyncio(
            total=len(audio_paths), desc=progress_desc, unit="files"
        )

        for audio_path in audio_paths:
            result = self.transcribe_audio(audio_path)
            results.append(result)

            if result["success"]:
                logger.debug(f"Transcribed: {audio_path} -> {result['text'][:50]}...")
            else:
                logger.warning(f"Failed to transcribe: {audio_path}")

            progress_bar.update(1)

        progress_bar.close()

        # Log summary
        success_count = sum(1 for r in results if r["success"])
        logger.info(
            f"Transcribed {success_count}/{len(results)} audio files successfully"
        )

        return results

    async def transcribe_from_manifest(
        self, manifest_path: Path
    ) -> List[Dict[str, Any]]:
        """Transcribe audio files from a manifest CSV file.

        Args:
            manifest_path: Path to the manifest CSV with 'audio_path' column

        Returns:
            List of transcription results
        """
        from .utils import read_csv

        # Read manifest
        manifest = read_csv(manifest_path)
        logger.info(f"Loaded {len(manifest)} items from manifest")

        # Extract audio paths
        audio_paths = [Path(row["audio_path"]) for row in manifest]

        # Filter existing files
        existing_paths = [p for p in audio_paths if p.exists()]
        missing_paths = [p for p in audio_paths if not p.exists()]

        if missing_paths:
            logger.warning(f"Missing audio files: {len(missing_paths)}")

        return await self.transcribe_batch(existing_paths, "Transcribing with Whisper")

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model:
            import torch

            del self.model
            self.model = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Whisper model unloaded")


def create_whisper_wrapper(
    model_size: str = "base", device: str = "cpu", language: Optional[str] = None
) -> WhisperWrapper:
    """Factory function to create a Whisper wrapper.

    Args:
        model_size: Whisper model size
        device: Device to run on
        language: Language code

    Returns:
        WhisperWrapper instance
    """
    return WhisperWrapper(model_size=model_size, device=device, language=language)


if __name__ == "__main__":
    import sys

    def test_wrapper():
        wrapper = create_whisper_wrapper()
        if wrapper.load_model():
            print("Whisper wrapper initialized successfully")
            wrapper.unload_model()
            return True
        else:
            print("Failed to initialize Whisper wrapper")
            return False

    logging.basicConfig(level=logging.INFO)
    result = test_wrapper()
    sys.exit(0 if result else 1)
