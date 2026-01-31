"""
src/whisper_wrapper.py

Whisper ASR Wrapper using HuggingFace Transformers pipeline with:
- Canonical HF usage for batching:
    pipe(["audio_1.mp3", "audio_2.mp3"], batch_size=2)
- Transcript always extracted from `text` key
- Strict memory-bounded mode: reload & unload model per batch so only <= batch_size audio files are processed at once

This file provides:
- HuggingFaceWhisperWrapper: actual HF pipeline implementation
- WhisperWrapper: compatibility layer used by Step 4
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Iterator

logger = logging.getLogger(__name__)


def _extract_text(out: Any) -> str:
    if isinstance(out, dict) and "text" in out:
        return (out.get("text") or "").strip()
    if isinstance(out, dict) and "chunks" in out and isinstance(out["chunks"], list):
        return " ".join((c.get("text", "") or "").strip() for c in out["chunks"]).strip()
    return (str(out) if out is not None else "").strip()


def _iter_batches(seq: List[Path], batch_size: int) -> Iterator[List[Path]]:
    bs = max(1, int(batch_size))
    for i in range(0, len(seq), bs):
        yield seq[i : i + bs]


class HuggingFaceWhisperWrapper:
    """HuggingFace Transformers-based Whisper wrapper with proper batching support."""

    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3",
        device: str = "cpu",                 # "cpu" or "cuda:0"
        language: Optional[str] = None,      # e.g. "en", "vi"
        task: str = "transcribe",            # "transcribe" or "translate"
        batch_size: int = 2,
        chunk_length_s: int = 0,             # 0 disables chunking
        attention_implementation: str = "default",  # "default", "sdpa", "flash_attention_2"
        use_torch_compile: bool = False,     # only safe when chunk_length_s == 0
        torch_dtype=None,                    # torch.float16 / torch.float32
    ):
        self.model_id = model_id
        self.device = device
        self.language = language
        self.task = task
        self.batch_size = batch_size
        self.chunk_length_s = chunk_length_s
        self.attention_implementation = attention_implementation
        self.use_torch_compile = use_torch_compile
        self.torch_dtype = torch_dtype

        self.pipe = None
        self.processor = None

        valid_attn = {"default", "sdpa", "flash_attention_2"}
        if self.attention_implementation not in valid_attn:
            logger.warning(
                f"Invalid attention implementation '{self.attention_implementation}', using 'default'"
            )
            self.attention_implementation = "default"

    def _check_dependencies(self) -> bool:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError as e:
            logger.warning(f"HuggingFace dependencies not available: {e}")
            logger.info("Install with: pip install transformers torch accelerate datasets")
            return False

    def _resolve_pipeline_device(self) -> int:
        """
        HF pipeline expects:
          -1 for CPU
           0..N-1 for CUDA device index
        """
        d = str(self.device).lower()
        if d.startswith("cuda"):
            if ":" in d:
                try:
                    return int(d.split(":")[1])
                except Exception:
                    return 0
            return 0
        return -1

    def _build_generate_kwargs(self) -> Dict[str, Any]:
        """
        Robust Whisper language/task control using forced decoder prompt ids.
        Fallbacks included for older transformers versions.
        """
        generate_kwargs: Dict[str, Any] = {}
        if not self.processor:
            return generate_kwargs

        if self.language or self.task:
            try:
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=self.language, task=self.task
                )
                generate_kwargs["forced_decoder_ids"] = forced_decoder_ids
            except Exception:
                if self.language:
                    generate_kwargs["language"] = self.language
                if self.task:
                    generate_kwargs["task"] = self.task

        return generate_kwargs

    def load_model(self) -> bool:
        """Load Whisper model + processor + pipeline."""
        try:
            if not self._check_dependencies():
                return False

            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

            device_idx = self._resolve_pipeline_device()

            # dtype: float16 on GPU, float32 on CPU by default
            torch_dtype = self.torch_dtype
            if torch_dtype is None:
                torch_dtype = (
                    torch.float16
                    if (device_idx >= 0 and torch.cuda.is_available())
                    else torch.float32
                )

            logger.info(
                f"Loading Whisper model '{self.model_id}' on device={device_idx} dtype={torch_dtype}"
            )

            attn_impl = None if self.attention_implementation == "default" else self.attention_implementation

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation=attn_impl,
            )

            if self.use_torch_compile and self.chunk_length_s == 0:
                logger.info("Applying torch.compile(model) ...")
                model = torch.compile(model, mode="reduce-overhead")

            # Put model on correct torch device
            if device_idx >= 0 and torch.cuda.is_available():
                model.to(f"cuda:{device_idx}")
            else:
                model.to("cpu")

            processor = AutoProcessor.from_pretrained(self.model_id)

            pipe_kwargs: Dict[str, Any] = dict(
                task="automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device_idx,  # IMPORTANT
            )

            if self.chunk_length_s and self.chunk_length_s > 0:
                pipe_kwargs["chunk_length_s"] = self.chunk_length_s

            self.pipe = pipeline(**pipe_kwargs)
            self.processor = processor
            return True

        except Exception as e:
            logger.exception(f"Failed to load Whisper model: {e}")
            self.pipe = None
            self.processor = None
            return False

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

    def transcribe_audio(self, audio_path: Path) -> Dict[str, Any]:
        """Transcribe a single audio file."""
        if not self.pipe and not self.load_model():
            return {"success": False, "error": "Model not loaded", "audio_path": str(audio_path)}

        if not audio_path.exists():
            return {"success": False, "error": "Audio file not found", "audio_path": str(audio_path)}

        try:
            generate_kwargs = self._build_generate_kwargs()
            out = self.pipe(str(audio_path), generate_kwargs=generate_kwargs)
            return {"success": True, "text": _extract_text(out), "audio_path": str(audio_path)}
        except Exception as e:
            logger.exception(f"Error transcribing {audio_path}: {e}")
            return {"success": False, "error": str(e), "audio_path": str(audio_path)}

    def transcribe_batch_strict_reload(
        self,
        audio_paths: List[Path],
        progress_desc: str = "Transcribing (strict batches)",
    ) -> List[Dict[str, Any]]:
        """
        Strict memory-bounded transcription:
        - Only ever passes <= batch_size items to pipeline at once.
        - Reloads & unloads the model every batch (user accepts the cost).
        """
        from tqdm import tqdm

        existing = [p for p in audio_paths if p.exists()]
        missing = [p for p in audio_paths if not p.exists()]

        results: List[Dict[str, Any]] = []
        results.extend(
            [{"success": False, "error": "Audio file not found", "audio_path": str(p)} for p in missing]
        )

        if not existing:
            return results

        pbar = tqdm(total=len(existing), desc=progress_desc, unit="files")

        for batch in _iter_batches(existing, self.batch_size):
            if not self.load_model():
                for p in batch:
                    results.append({"success": False, "error": "Model not loaded", "audio_path": str(p)})
                    pbar.update(1)
                continue

            try:
                generate_kwargs = self._build_generate_kwargs()

                # Canonical batching: only THIS batch list (<= batch_size)
                outs = self.pipe(
                    [str(p) for p in batch],
                    batch_size=self.batch_size,
                    generate_kwargs=generate_kwargs,
                )

                for p, out in zip(batch, outs):
                    results.append({"success": True, "text": _extract_text(out), "audio_path": str(p)})
                    pbar.update(1)

            except Exception as e:
                logger.exception(f"Batch transcription error: {e}")
                for p in batch:
                    results.append({"success": False, "error": str(e), "audio_path": str(p)})
                    pbar.update(1)

            finally:
                self.unload_model()

        pbar.close()
        return results


class WhisperWrapper:
    """
    Compatibility wrapper used by the rest of the codebase.

    Important:
    - Step 4 uses transcribe_batch_strict_reload() to avoid OOM.
    """

    def __init__(
        self,
        model_size: str = "base",  # kept for API compatibility
        device: str = "cpu",
        language: Optional[str] = None,
        batch_size: int = 2,
    ):
        self.model_size = model_size
        self.device = device
        self.language = language
        self.batch_size = batch_size

        self.hf_wrapper: Optional[HuggingFaceWhisperWrapper] = None
        self.use_hf = False

    def __create_hf_wrapper(self) -> HuggingFaceWhisperWrapper:
        """Create HF wrapper using config if available."""
        try:
            from .config import get_config
            import torch

            config = get_config()

            device = getattr(config, "WHISPER_DEVICE", self.device)
            model_id = getattr(config, "WHISPER_MODEL_ID", "openai/whisper-large-v3")
            chunk_length_s = getattr(config, "WHISPER_CHUNK_LENGTH_S", 0)
            attn = getattr(config, "WHISPER_ATTENTION_IMPLEMENTATION", "default")
            use_compile = getattr(config, "WHISPER_USE_TORCH_COMPILE", False)

            # dtype default: float16 on cuda, float32 on cpu
            device_is_cuda = str(device).lower().startswith("cuda")
            torch_dtype = torch.float16 if (device_is_cuda and torch.cuda.is_available()) else torch.float32

            return HuggingFaceWhisperWrapper(
                model_id=model_id,
                device=device,
                language=self.language,
                task=getattr(config, "WHISPER_TASK", "transcribe"),
                batch_size=self.batch_size,
                chunk_length_s=chunk_length_s,
                attention_implementation=attn,
                use_torch_compile=use_compile,
                torch_dtype=torch_dtype,
            )
        except Exception:
            # Safe defaults if config import fails
            import torch
            torch_dtype = (
                torch.float16
                if torch.cuda.is_available() and str(self.device).lower().startswith("cuda")
                else torch.float32
            )
            return HuggingFaceWhisperWrapper(
                model_id="openai/whisper-large-v3",
                device=self.device,
                language=self.language,
                task="transcribe",
                batch_size=self.batch_size,
                chunk_length_s=0,
                attention_implementation="default",
                use_torch_compile=False,
                torch_dtype=torch_dtype,
            )

    def load_model(self) -> bool:
        """
        Keep for backward compatibility; strict batch mode loads per batch anyway.
        """
        self.hf_wrapper = self.__create_hf_wrapper()
        self.use_hf = True
        return True

    def transcribe_audio(self, audio_path: Path, verbose: bool = False) -> Dict[str, Any]:
        if not self.hf_wrapper:
            self.hf_wrapper = self.__create_hf_wrapper()
        return self.hf_wrapper.transcribe_audio(audio_path)

    def transcribe_batch_strict_reload(
        self,
        audio_paths: List[Path],
        progress_desc: str = "Transcribing (strict batches)",
    ) -> List[Dict[str, Any]]:
        if not self.hf_wrapper:
            self.hf_wrapper = self.__create_hf_wrapper()
        return self.hf_wrapper.transcribe_batch_strict_reload(
            audio_paths=audio_paths,
            progress_desc=progress_desc,
        )

    def unload_model(self) -> None:
        if self.hf_wrapper:
            self.hf_wrapper.unload_model()
        self.hf_wrapper = None
        self.use_hf = False
