"""
Step 4: ASR Pipeline - Transcribe audio, then analyze with LLM

Simplified version:
- Removes WhisperWrapper entirely
- Uses Hugging Face transformers pipeline with openai/whisper-large-v3
- Loads Whisper ONCE
- Feeds ALL audio paths at once; batching is handled by pipeline(batch_size=asr_batch_size)
- LLM stage stays async with a concurrency limit
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.utils import setup_logging, read_csv, write_jsonl
from src.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)

ASR_PROMPT = """You heard someone say: "{transcription}"

How would you respond? What do you think they mean?"""


def create_asr_prompt(transcription: str) -> str:
    return ASR_PROMPT.format(transcription=transcription)


def _base_result(item: Dict[str, Any], audio_path: Path) -> Dict[str, Any]:
    return {
        "dialog_id": item.get("dialog_id", ""),
        "audio_path": str(audio_path),
        "original_utterance": item.get("text", ""),
        "rewritten_text": item.get("text", ""),
        "emotion": item.get("emotion", ""),
    }


def _build_results_from_transcriptions(
    transcription_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Map transcription result by audio_path for joining with manifest items."""
    m: Dict[str, Dict[str, Any]] = {}
    for r in transcription_results:
        ap = str(r.get("audio_path", ""))
        if ap:
            m[ap] = r
    return m


def _resolve_whisper_model_id(default: str = "openai/whisper-large-v3") -> str:
    """
    Optional: allow config override if you already have WHISPER_MODEL_ID.
    Falls back to openai/whisper-large-v3.
    """
    cfg = get_config()
    return getattr(cfg, "WHISPER_MODEL_ID", None) or default


def load_whisper_pipe(model_id: str, asr_batch_size: int):
    """
    Load Whisper model + processor + HF pipeline once.
    Batching is handled by pipeline(..., batch_size=asr_batch_size).
    """
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device_str)

    processor = AutoProcessor.from_pretrained(model_id)

    # Most compatible: int GPU index or -1 for CPU
    device_arg = 0 if torch.cuda.is_available() else -1

    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device_arg,
        batch_size=asr_batch_size,
    )
    return asr_pipe


def transcribe_all_audio(
    asr_pipe,
    audio_paths: List[Path],
) -> List[Dict[str, Any]]:
    """
    Transcribe all audio paths in one call.
    HF pipeline handles micro-batching internally via batch_size.
    """
    audio_strs = [str(p) for p in audio_paths]

    # If you want to be strict about existence, uncomment:
    # audio_strs = [p for p in audio_strs if Path(p).exists()]

    try:
        outputs = asr_pipe(audio_strs)

        # transformers returns dict for single input, list for multiple
        if isinstance(outputs, dict):
            outputs = [outputs]
        else:
            outputs = list(outputs)

        results: List[Dict[str, Any]] = []
        for p, out in zip(audio_strs, outputs):
            results.append(
                {
                    "audio_path": p,
                    "success": True,
                    "text": ((out or {}).get("text") or "").strip(),
                    "error": None,
                }
            )

        # In the unlikely case outputs length mismatches, mark remaining as failures
        if len(outputs) < len(audio_strs):
            for p in audio_strs[len(outputs) :]:
                results.append(
                    {
                        "audio_path": p,
                        "success": False,
                        "text": None,
                        "error": "ASR pipeline returned no output for this file",
                    }
                )

        return results

    except Exception as e:
        logger.error(f"ASR failed: {e}")
        return [
            {
                "audio_path": str(p),
                "success": False,
                "text": None,
                "error": str(e),
            }
            for p in audio_paths
        ]


async def llm_stage(
    openrouter_client: OpenRouterClient,
    items: List[Dict[str, Any]],
    transcriptions_by_path: Dict[str, Dict[str, Any]],
    llm_model: str,
    max_concurrency: int = 16,
    progress_desc: str = "LLM responses",
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(max_concurrency)

    async def _one(item: Dict[str, Any]) -> Dict[str, Any]:
        audio_path = Path(item.get("audio_path", ""))
        base = _base_result(item, audio_path)

        tr = transcriptions_by_path.get(str(audio_path))
        if not tr or not tr.get("success"):
            return {
                **base,
                "transcription": None,
                "transcription_success": False,
                "transcription_error": (tr or {}).get("error", "Unknown error"),
                "asr_response": None,
                "asr_success": False,
                "asr_error": "Skipped due to transcription failure",
            }

        transcription = (tr.get("text") or "").strip()
        if not transcription:
            return {
                **base,
                "transcription": "",
                "transcription_success": True,
                "transcription_error": None,
                "asr_response": None,
                "asr_success": False,
                "asr_error": "Empty transcription",
            }

        prompt = create_asr_prompt(transcription)

        try:
            async with sem:
                response = await openrouter_client.chat_text(
                    prompt=prompt,
                    model=llm_model,
                    temperature=0.7,
                    max_tokens=256,
                )

            return {
                **base,
                "transcription": transcription,
                "transcription_success": True,
                "transcription_error": None,
                "asr_response": (response or "").strip(),
                "asr_success": True,
                "asr_error": None,
            }

        except Exception as e:
            logger.error(f"ASR LLM error for {audio_path}: {e}")
            return {
                **base,
                "transcription": transcription,
                "transcription_success": True,
                "transcription_error": None,
                "asr_response": None,
                "asr_success": False,
                "asr_error": str(e),
            }

    tasks = [asyncio.create_task(_one(it)) for it in items]

    results: List[Dict[str, Any]] = []
    for t in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=progress_desc, unit="files"):
        results.append(await t)
    return results


async def _run_llm_and_close(
    openrouter_client: OpenRouterClient,
    items: List[Dict[str, Any]],
    transcriptions_by_path: Dict[str, Dict[str, Any]],
    llm_model: str,
    llm_max_concurrency: int,
) -> List[Dict[str, Any]]:
    try:
        return await llm_stage(
            openrouter_client=openrouter_client,
            items=items,
            transcriptions_by_path=transcriptions_by_path,
            llm_model=llm_model,
            max_concurrency=llm_max_concurrency,
        )
    finally:
        await openrouter_client.close()


def run_step4(
    manifest_path: Path,
    output_path: Path,
    asr_model: str = "base",  # kept for CLI compatibility; HF uses model_id
    llm_model: str = "openai/gpt-4o-mini",
    # ASR knobs
    asr_batch_size: int = 32,  # <- per your request
    # LLM knobs
    llm_max_concurrency: int = 16,
) -> bool:
    logger.info("=" * 60)
    logger.info("STEP 4: ASR Pipeline (HF Whisper batched + LLM)")
    logger.info("=" * 60)

    try:
        logger.info(f"Loading manifest from {manifest_path}")
        manifest = read_csv(manifest_path)
        logger.info(f"Loaded {len(manifest)} items")

        successful_items = [
            item for item in manifest if str(item.get("success", "")).lower() == "true"
        ]
        logger.info(f"Processing {len(successful_items)} successful audio files")

        if not successful_items:
            logger.warning("No successful audio files to process")
            write_jsonl([], output_path)
            return True

        # ---- Stage 1 (SYNC): load Whisper once, transcribe all audio paths (pipeline handles batching) ----
        audio_paths = [Path(it.get("audio_path", "")) for it in successful_items]

        model_id = _resolve_whisper_model_id(default="openai/whisper-large-v3")
        logger.info(
            f"Using Whisper model_id: {model_id} (CLI --asr-model='{asr_model}' kept for compatibility)"
        )
        logger.info(f"ASR pipeline batch_size: {asr_batch_size}")

        asr_pipe = load_whisper_pipe(model_id=model_id, asr_batch_size=asr_batch_size)

        transcription_results = transcribe_all_audio(asr_pipe=asr_pipe, audio_paths=audio_paths)

        # Cleanup to free memory for LLM stage (optional but helpful on smaller GPUs)
        del asr_pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        transcriptions_by_path = _build_results_from_transcriptions(transcription_results)

        # ---- Stage 2 (ASYNC): LLM responses ----
        openrouter_client = OpenRouterClient()
        results = asyncio.run(
            _run_llm_and_close(
                openrouter_client=openrouter_client,
                items=successful_items,
                transcriptions_by_path=transcriptions_by_path,
                llm_model=llm_model,
                llm_max_concurrency=llm_max_concurrency,
            )
        )

        results.sort(key=lambda x: x.get("dialog_id", ""))

        write_jsonl(results, output_path)
        logger.info(f"Saved {len(results)} ASR responses to {output_path}")

        transcription_success = sum(1 for r in results if r.get("transcription_success"))
        asr_success = sum(1 for r in results if r.get("asr_success"))
        logger.info(f"Transcription success: {transcription_success}/{len(results)}")
        logger.info(f"ASR pipeline success: {asr_success}/{len(results)}")

        logger.info("=" * 60)
        logger.info("STEP 4 COMPLETED SUCCESSFULLY")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"Step 4 failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Step 4: ASR Pipeline")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "step2_audio_manifest.csv",
        help="Path to audio manifest CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "outputs" / "responses" / "step4_asr_responses.jsonl",
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--asr-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Kept for CLI compatibility (HF uses WHISPER_MODEL_ID or openai/whisper-large-v3).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="openai/gpt-4o-mini",
        help="LLM model for response",
    )
    parser.add_argument(
        "--asr-batch-size",
        type=int,
        default=4,
        help="HF pipeline batch_size (controls ASR micro-batching).",
    )
    parser.add_argument(
        "--llm-max-concurrency",
        type=int,
        default=16,
        help="Max in-flight LLM requests.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    success = run_step4(
        manifest_path=args.manifest,
        output_path=args.output,
        asr_model=args.asr_model,
        llm_model=args.llm_model,
        asr_batch_size=args.asr_batch_size,
        llm_max_concurrency=args.llm_max_concurrency,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
