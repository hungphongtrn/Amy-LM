"""
Whisper ASR Wrapper using HuggingFace Transformers pipeline for ASR batching support.
Refactored from openai-whisper to HuggingFace Transformers for better batching capabilities.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)


class HuggingFaceWhisperWrapper:
    """HuggingFace Transformers-based Whisper wrapper with proper batching support."""

    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3",
        device: str = "cpu",
        language: Optional[str] = None,
        chunk_length_s: int = 30,
        batch_size: int = 16,
        attention_implementation: str = "sdpa",
        use_torch_compile: bool = False,
        torch_dtype=None,
    ):
        """Initialize the HuggingFace Whisper wrapper.

        Args:
            model_id: HuggingFace model ID (e.g., 'openai/whisper-large-v3')
            device: Device to run inference on ('cpu' or 'cuda')
            language: Language code for transcription (e.g., 'en' for English)
            chunk_length_s: Chunk length for long-form transcription (0 to disable chunking)
            batch_size: Number of audio files to process in a batch
            attention_implementation: Attention implementation ('flash_attention_2', 'sdpa', 'default')
            use_torch_compile: Whether to use torch.compile for speed-up (incompatible with chunked algorithm)
            torch_dtype: torch data type for inference (torch.float16 or torch.float32)
        """
        self.model_id = model_id
        self.device = device
        self.language = language
        self.chunk_length_s = chunk_length_s
        self.batch_size = batch_size
        self.attention_implementation = attention_implementation
        self.use_torch_compile = use_torch_compile
        self.torch_dtype = torch_dtype
        self.pipe = None
        self.processor = None

        # Validate attention implementation
        valid_attn = {"flash_attention_2", "sdpa", "default"}
        if self.attention_implementation not in valid_attn:
            logger.warning(
                f"Invalid attention implementation '{self.attention_implementation}', "
                f"using 'default' instead"
            )
            self.attention_implementation = "default"

        # Check dependencies
        self._check_dependencies()

    def _check_dependencies(self) -> bool:
        """Check if HuggingFace dependencies are available.

        Returns:
            True if dependencies are available, False otherwise
        """
        try:
            import transformers
            import torch

            logger.info(f"Transformers version: {transformers.__version__}")
            logger.info(f"PyTorch version: {torch.__version__}")
            return True
        except ImportError as e:
            logger.warning(f"HuggingFace dependencies not available: {e}")
            logger.info("Install with: pip install transformers torch accelerate")
            return False

    def load_model(self) -> bool:
        """Load the Whisper model using HuggingFace Transformers pipeline.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not self._check_dependencies():
                logger.error("HuggingFace dependencies not available")
                return False

            from transformers import (
                AutoModelForSpeechSeq2Seq,
                AutoProcessor,
                pipeline,
            )

            import torch

            logger.info(
                f"Loading Whisper model '{self.model_id}' on {self.device} "
                f"(attention: {self.attention_implementation}, "
                f"compile: {self.use_torch_compile})..."
            )

            # Set up device and dtype
            device = self.device
            torch_dtype = self.torch_dtype or (
                torch.float16
                if torch.cuda.is_available() and device.startswith("cuda")
                else torch.float32
            )

            # Load model with specified attention implementation
            attn_implementation = (
                None
                if self.attention_implementation == "default"
                else self.attention_implementation
            )

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
            )

            # Apply torch.compile if requested and not using chunked algorithm
            if self.use_torch_compile and self.chunk_length_s == 0:
                logger.info("Applying torch.compile for speed-up...")
                model = torch.compile(model, mode="reduce-overhead")

            model.to(device)

            # Load processor
            processor = AutoProcessor.from_pretrained(self.model_id)

            # Create pipeline
            # Note: chunk_length_s > 0 enables chunked long-form transcription
            # chunk_length_s = 0 or None disables chunking (needed for torch.compile)
            chunk_length_s = self.chunk_length_s if self.chunk_length_s > 0 else None

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=chunk_length_s,
                batch_size=self.batch_size,
                torch_dtype=torch_dtype,
                device=device,
            )

            self.processor = processor
            logger.info(
                "Whisper model loaded successfully via HuggingFace Transformers"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return False

    def transcribe_audio(
        self, audio_path: Path, verbose: bool = False
    ) -> Dict[str, Any]:
        """Transcribe a single audio file using the pipeline.

        Args:
            audio_path: Path to the audio file
            verbose: Whether to print verbose output

        Returns:
            Dictionary containing transcription results
        """
        try:
            if not self.pipe:
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

            # Prepare generation kwargs
            generate_kwargs = {}

            if self.language:
                generate_kwargs["language"] = self.language

            if verbose:
                generate_kwargs["verbose"] = True

            # Transcribe audio using pipeline
            result = self.pipe(str(audio_path), generate_kwargs=generate_kwargs)

            # Extract text from result
            # Pipeline returns {'text': ...} or {'chunks': [...]} for chunked mode
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            text = text.strip() if text else ""

            # Try to get language info if available
            language = (
                result.get("language", self.language)
                if isinstance(result, dict)
                else self.language
            )

            return {
                "success": True,
                "text": text,
                "audio_path": str(audio_path),
                "language": language,
            }

        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return {"success": False, "error": str(e), "audio_path": str(audio_path)}

    async def transcribe_batch(
        self, audio_paths: List[Path], progress_desc: str = "Transcribing audio"
    ) -> List[Dict[str, Any]]:
        """Transcribe a batch of audio files using proper pipeline batching.

        Args:
            audio_paths: List of paths to audio files
            progress_desc: Description for progress bar

        Returns:
            List of transcription results
        """
        if not self.pipe:
            if not self.load_model():
                logger.error("Failed to load Whisper model")
                return []

        # Filter existing files
        existing_paths = [p for p in audio_paths if p.exists()]
        missing_paths = [p for p in audio_paths if not p.exists()]

        if missing_paths:
            logger.warning(f"Missing audio files: {len(missing_paths)}")

        if not existing_paths:
            logger.warning("No valid audio files to transcribe")
            # Return failed results for missing files
            return [
                {
                    "success": False,
                    "error": "Audio file not found",
                    "audio_path": str(p),
                }
                for p in missing_paths
            ]

        # Prepare generation kwargs
        generate_kwargs = {}
        if self.language:
            generate_kwargs["language"] = self.language

        try:
            # Convert paths to strings for pipeline
            audio_strs = [str(p) for p in existing_paths]

            # Use pipeline batch transcription
            # The pipeline handles batching internally with the batch_size parameter
            logger.info(
                f"Transcribing {len(audio_strs)} audio files with batch_size={self.batch_size}..."
            )

            # Create progress bar
            progress_bar = tqdm_asyncio(
                total=len(existing_paths), desc=progress_desc, unit="files"
            )

            # Process in batches using the pipeline
            all_results = self.pipe(
                audio_strs,
                batch_size=self.batch_size,
                generate_kwargs=generate_kwargs,
                return_timestamps=False,
            )

            # Convert pipeline results to our format
            results = []
            for i, (audio_path, result) in enumerate(zip(existing_paths, all_results)):
                # Extract text from result
                if isinstance(result, dict):
                    text = result.get("text", "").strip()
                    language = result.get("language", self.language)
                else:
                    text = str(result).strip()
                    language = self.language

                transcription_result = {
                    "success": True,
                    "text": text,
                    "audio_path": str(audio_path),
                    "language": language,
                }

                results.append(transcription_result)

                if text:
                    logger.debug(f"Transcribed: {audio_path} -> {text[:50]}...")
                else:
                    logger.warning(f"Empty transcription for: {audio_path}")

                progress_bar.update(1)

            progress_bar.close()

            # Add failed results for missing files
            results.extend(
                [
                    {
                        "success": False,
                        "error": "Audio file not found",
                        "audio_path": str(p),
                    }
                    for p in missing_paths
                ]
            )

            # Log summary
            success_count = sum(1 for r in results if r["success"])
            logger.info(
                f"Transcribed {success_count}/{len(results)} audio files successfully"
            )

            return results

        except Exception as e:
            logger.error(f"Error in batch transcription: {e}")
            import traceback

            logger.debug(traceback.format_exc())

            # Return failed results for all files
            return [
                {
                    "success": False,
                    "error": str(e),
                    "audio_path": str(p),
                }
                for p in audio_paths
            ]

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

        return await self.transcribe_batch(audio_paths, "Transcribing with Whisper")

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.pipe:
            import torch

            del self.pipe
            self.pipe = None
            self.processor = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Whisper model unloaded")


class WhisperWrapper:
    """Backward-compatible wrapper for OpenAI Whisper speech recognition."""

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
        self.hf_wrapper = None
        self.use_hf = False

    def __create_hf_wrapper(self) -> HuggingFaceWhisperWrapper:
        """Create a HuggingFace wrapper with config settings.

        Returns:
            HuggingFaceWhisperWrapper instance
        """
        try:
            from .config import get_config

            config = get_config()

            import torch

            # Determine torch dtype based on device
            torch_dtype = (
                torch.float16
                if torch.cuda.is_available() and self.device.startswith("cuda")
                else torch.float32
            )

            return HuggingFaceWhisperWrapper(
                model_id=getattr(config, "WHISPER_MODEL_ID", "openai/whisper-large-v3"),
                device=self.device,
                language=self.language,
                chunk_length_s=getattr(config, "WHISPER_CHUNK_LENGTH_S", 30),
                batch_size=getattr(config, "WHISPER_BATCH_SIZE", self.batch_size),
                attention_implementation=getattr(
                    config, "WHISPER_ATTENTION_IMPLEMENTATION", "sdpa"
                ),
                use_torch_compile=getattr(config, "WHISPER_USE_TORCH_COMPILE", False),
                torch_dtype=torch_dtype,
            )
        except ImportError:
            # Fallback if config not available
            import torch

            torch_dtype = (
                torch.float16
                if torch.cuda.is_available() and self.device.startswith("cuda")
                else torch.float32
            )

            return HuggingFaceWhisperWrapper(
                model_id="openai/whisper-large-v3",
                device=self.device,
                language=self.language,
                chunk_length_s=30,
                batch_size=self.batch_size,
                attention_implementation="sdpa",
                use_torch_compile=False,
                torch_dtype=torch_dtype,
            )

    def _check_dependencies(self) -> bool:
        """Check if whisper dependencies are available.

        Returns:
            True if dependencies are available, False otherwise
        """
        try:
            import whisper
            import torch

            logger.info("OpenAI Whisper dependencies found")
            return True
        except ImportError:
            logger.warning(
                "OpenAI Whisper not available, using HuggingFace Transformers"
            )
            return False

    def load_model(self) -> bool:
        """Load the Whisper model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        # Try HuggingFace first (preferred)
        try:
            logger.info("Attempting to load HuggingFace Transformers Whisper...")
            self.hf_wrapper = self.__create_hf_wrapper()
            if self.hf_wrapper.load_model():
                self.use_hf = True
                logger.info("Using HuggingFace Transformers Whisper backend")
                return True
            self.hf_wrapper = None
        except Exception as e:
            logger.warning(f"Failed to load HuggingFace Whisper: {e}")

        # Fallback to openai-whisper if available
        if self._check_dependencies():
            logger.info("Falling back to openai-whisper...")
            return self._load_whisper_model()

        logger.error("No Whisper backend available")
        return False

    def _load_whisper_model(self) -> bool:
        """Load the OpenAI Whisper model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            import whisper

            logger.info(
                f"Loading Whisper '{self.model_size}' model on {self.device}..."
            )

            self.model = whisper.load_model(self.model_size, device=self.device)
            self.use_hf = False

            logger.info("OpenAI Whisper model loaded successfully")
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
        if self.use_hf and self.hf_wrapper:
            return self.hf_wrapper.transcribe_audio(audio_path, verbose)

        # Fallback to openai-whisper
        return self._transcribe_audio_whisper(audio_path, verbose)

    def _transcribe_audio_whisper(
        self, audio_path: Path, verbose: bool = False
    ) -> Dict[str, Any]:
        """Transcribe a single audio file using openai-whisper.

        Args:
            audio_path: Path to the audio file
            verbose: Whether to print verbose output

        Returns:
            Dictionary containing transcription results
        """
        try:
            import whisper
            import torch

            if not getattr(self, "model", None):
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
        if self.use_hf and self.hf_wrapper:
            return await self.hf_wrapper.transcribe_batch(audio_paths, progress_desc)

        # Fallback to openai-whisper
        return await self._transcribe_batch_whisper(audio_paths, progress_desc)

    async def _transcribe_batch_whisper(
        self, audio_paths: List[Path], progress_desc: str = "Transcribing audio"
    ) -> List[Dict[str, Any]]:
        """Transcribe a batch of audio files using openai-whisper.

        Args:
            audio_paths: List of paths to audio files
            progress_desc: Description for progress bar

        Returns:
            List of transcription results
        """
        if not getattr(self, "model", None):
            if not self.load_model():
                logger.error("Failed to load Whisper model")
                return []

        results = []

        # Create progress bar
        progress_bar = tqdm_asyncio(
            total=len(audio_paths), desc=progress_desc, unit="files"
        )

        for audio_path in audio_paths:
            result = self._transcribe_audio_whisper(audio_path)
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
        if self.use_hf and self.hf_wrapper:
            self.hf_wrapper.unload_model()
            self.hf_wrapper = None
        elif getattr(self, "model", None):
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
            if wrapper.use_hf:
                print("Using HuggingFace Transformers backend")
            else:
                print("Using openai-whisper backend")
            wrapper.unload_model()
            return True
        else:
            print("Failed to initialize Whisper wrapper")
            return False

    logging.basicConfig(level=logging.INFO)
    result = test_wrapper()
    sys.exit(0 if result else 1)
