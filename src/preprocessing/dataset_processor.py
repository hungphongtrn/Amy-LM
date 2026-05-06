"""Dataset Processor for Amy-LM preprocessing pipeline.

This module provides the DatasetProcessor class which orchestrates loading
Hugging Face datasets and running FACodec encoding on batches of audio samples.

The processor:
1. Loads datasets from Hugging Face Hub (or mocks for testing)
2. Processes audio samples in batches through FACodec encoder
3. Extracts content, prosody, and timbre codebook indices
4. Builds a new HF Dataset with all required columns
5. Handles errors gracefully (per-sample, doesn't crash)
6. Supports saving to disk and pushing to HF Hub

Example:
    >>> from preprocessing.facodec_encoder import FACodecEncoder
    >>> from preprocessing.dataset_processor import DatasetProcessor
    >>> from pathlib import Path
    >>> 
    >>> encoder = FACodecEncoder(device="cpu")
    >>> processor = DatasetProcessor(encoder, Path("./output"), "cpu")
    >>> 
    >>> dataset = processor.process_dataset("user/dataset", split="train")
    >>> processor.save(dataset, "user/processed-dataset")
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from preprocessing.facodec_encoder import FACodecEncoder, FACodecStreams

# Try to import datasets, provide helpful error if not available
try:
    from datasets import load_dataset, Dataset
    from datasets import Audio as HFAudio
    from datasets import Features, Value, Sequence
    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False


class DatasetProcessor:
    """Processes audio datasets through FACodec encoder.
    
    This class orchestrates the preprocessing pipeline:
    - Loads HF datasets (mocked in tests)
    - Encodes audio through FACodec in batches
    - Extracts all FACodec streams: prosody, content, acoustic, timbre vector
    - Handles errors gracefully (logs failures, continues)
    - Supports saving to disk and pushing to HF Hub
    
    Args:
        facodec: FACodecEncoder instance for encoding audio
        output_dir: Directory to save processed datasets
        device: Device to run on ("cpu" or "cuda")
    
    Attributes:
        facodec: The FACodec encoder instance
        output_dir: Output directory path
        device: The device being used
    
    Example:
        >>> encoder = FACodecEncoder(device="cpu")
        >>> processor = DatasetProcessor(encoder, Path("./output"), "cpu")
        >>> dataset = processor.process_dataset("user/audio-dataset", split="train")
        >>> processor.push_to_hub(dataset, "user/processed-audio")
    """
    
    def __init__(self, facodec: FACodecEncoder, output_dir: Path, device: str):
        """Initialize the DatasetProcessor.
        
        Args:
            facodec: FACodecEncoder instance for encoding
            output_dir: Directory path for saving datasets
            device: Device string ("cpu" or "cuda")
        
        Raises:
            ImportError: If datasets library is not available
        """
        if not _DATASETS_AVAILABLE:
            raise ImportError(
                "The 'datasets' library is required. "
                "Install with: pip install datasets"
            )
        
        self.facodec = facodec
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_dataset(
        self, 
        dataset_name: str, 
        split: str, 
        max_samples: Optional[int] = None,
        batch_size: int = 8,
        config: Optional[str] = None,
    ) -> Dataset:
        """Process a dataset through FACodec encoder.
        
        Loads a dataset from Hugging Face Hub, processes audio samples
        through FACodec in batches, and returns a new dataset with codebook indices.
        
        Args:
            dataset_name: Hugging Face dataset name (e.g., "user/dataset")
            split: Dataset split to load (e.g., "train", "validation", "test")
            max_samples: Maximum number of samples to process (None for all)
            batch_size: Number of samples to encode at once through FACodec.
                Larger values increase GPU throughput but also memory usage.
            config: Optional dataset config/subset name (e.g., "emns" for CAMEO).
        
        Returns:
            A HF Dataset with columns:
                - dataset: Source dataset name
                - id: Unique sample ID
                - audio: Audio dict with array and sampling_rate
                - prosody_codebooks_idx: List of prosody indices [T80]
                - content_codebooks_idx: Nested list of content indices [2, T80]
                - acoustic_codebooks_idx: Nested list of acoustic indices [3, T80]
                - timbre_vector: List of 256 float32 values
        
        Raises:
            ValueError: If dataset loading fails
            RuntimeError: If processing fails catastrophically
        
        Note:
            Samples that fail encoding are skipped but processing continues.
            Check the tqdm output for failure count.
        """
        if config:
            source_dataset = load_dataset(dataset_name, config, split=split)
        else:
            source_dataset = load_dataset(dataset_name, split=split)
        
        if max_samples is not None:
            source_dataset = source_dataset.select(range(min(max_samples, len(source_dataset))))
        
        processed_data: List[Dict[str, Any]] = []
        failures: List[tuple] = []
        
        audio_batch: List[torch.Tensor] = []
        pending_entries: List[tuple] = []
        
        for idx, sample in enumerate(tqdm(source_dataset, desc=f"Processing {dataset_name}")):
            try:
                audio_tensor = self._extract_audio_from_sample(sample)
                audio_batch.append(audio_tensor)
                pending_entries.append((sample, dataset_name, idx))
                
                if len(audio_batch) >= batch_size or idx == len(source_dataset) - 1:
                    results = self.facodec.encode_batch(audio_batch)
                    for (samp, ds_name, row_idx), streams in zip(
                        pending_entries, results
                    ):
                        entry = self._build_processed_entry(
                            samp, ds_name, row_idx, streams
                        )
                        processed_data.append(entry)
                    audio_batch.clear()
                    pending_entries.clear()
            except Exception as e:
                sample_id = sample.get("id", f"row_{idx}")
                failures.append((sample_id, str(e)))
        
        if failures:
            print(f"\n  {len(failures)} samples failed to process:")
            for sample_id, error in failures[:10]:
                print(f"  - {sample_id}: {error}")
            if len(failures) > 10:
                print(f"  ... and {len(failures) - 10} more")
        
        if not processed_data:
            raise RuntimeError("No samples were successfully processed")
        
        features = Features({
            "dataset": Value("string"),
            "id": Value("string"),
            "audio": HFAudio(sampling_rate=16000),
            # prosody: single codebook [T80] -> Sequence(int64)
            "prosody_codebooks_idx": Sequence(Value("int64")),
            # content: 2 codebooks [2, T80] -> Sequence(Sequence(int64))
            "content_codebooks_idx": Sequence(Sequence(Value("int64"))),
            # acoustic: 3 codebooks [3, T80] -> Sequence(Sequence(int64))
            "acoustic_codebooks_idx": Sequence(Sequence(Value("int64"))),
            # timbre vector: [256] float32 -> Sequence(float32)
            "timbre_vector": Sequence(Value("float32")),
        })
        return Dataset.from_list(processed_data, features=features)
    
    def _extract_audio_from_sample(self, sample: Dict[str, Any]) -> torch.Tensor:
        """Extract and preprocess audio from a dataset sample.
        
        Returns a 1D float tensor at 16kHz ready for FACodec encoding.
        """
        audio_data = sample.get("audio")
        if audio_data is None:
            raise ValueError("Sample missing 'audio' field")
        
        if isinstance(audio_data, dict):
            audio_array = audio_data.get("array")
            sampling_rate = audio_data.get("sampling_rate", 16000)
        elif isinstance(audio_data, np.ndarray):
            audio_array = audio_data
            sampling_rate = 16000
        else:
            try:
                samples = audio_data.get_all_samples()
                audio_array = samples.data.numpy().squeeze()
                sampling_rate = samples.sample_rate
            except AttributeError:
                raise ValueError(f"Unsupported audio format: {type(audio_data)}")
        
        if isinstance(audio_array, list):
            audio_array = np.array(audio_array, dtype=np.float32)
        
        if audio_array is None or len(audio_array) == 0:
            raise ValueError("Empty audio array")
        
        if sampling_rate != 16000:
            audio_array = self._resample_audio(audio_array, sampling_rate, 16000)
            sampling_rate = 16000
        
        return torch.from_numpy(audio_array).float()

    def _build_processed_entry(
        self,
        sample: Dict[str, Any],
        dataset_name: str,
        row_idx: int,
        streams: FACodecStreams,
    ) -> Dict[str, Any]:
        """Construct a processed sample entry from FACodecStreams.
        
        Converts tensors to nested lists for HF Dataset serialization.
        
        Args:
            sample: Original dataset sample
            dataset_name: Name of the source dataset
            row_idx: Row index in the source dataset
            streams: FACodecStreams with encoded representations
            
        Returns:
            Dict with all fields ready for HF Dataset
        """
        audio_data = sample.get("audio")

        if isinstance(audio_data, dict):
            audio_array = audio_data.get("array")
            sampling_rate = audio_data.get("sampling_rate", 16000)
        else:
            try:
                audio_array = audio_data["array"]
                sampling_rate = audio_data["sampling_rate"]
            except Exception:
                audio_array = None
                sampling_rate = 16000

        if isinstance(audio_array, list):
            audio_array = np.array(audio_array, dtype=np.float32)
        elif isinstance(audio_array, np.ndarray):
            audio_array = audio_array.astype(np.float32)

        sample_id = sample.get("id", f"row_{row_idx}")
        
        # Convert tensors to nested lists
        # prosody: [1, T] -> squeeze to [T] -> list
        prosody_indices = streams.prosody_codebooks_idx.squeeze(0).tolist()
        
        # content: [2, T] -> list of 2 lists
        content_indices = streams.content_codebooks_idx.tolist()
        
        # acoustic: [3, T] -> list of 3 lists
        acoustic_indices = streams.acoustic_codebooks_idx.tolist()
        
        # timbre vector: [256] -> list of 256 floats
        timbre_vector = streams.timbre_vector.tolist()
        
        return {
            "dataset": dataset_name,
            "id": sample_id,
            "audio": {"array": audio_array, "sampling_rate": sampling_rate},
            "prosody_codebooks_idx": prosody_indices,
            "content_codebooks_idx": content_indices,
            "acoustic_codebooks_idx": acoustic_indices,
            "timbre_vector": timbre_vector,
        }
    
    def _resample_audio(
        self, 
        audio: np.ndarray, 
        orig_sr: int, 
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate.
        
        Uses simple linear interpolation for resampling.
        
        Args:
            audio: Audio array
            orig_sr: Original sample rate
            target_sr: Target sample rate
        
        Returns:
            Resampled audio array
        """
        if orig_sr == target_sr:
            return audio
        
        # Calculate new length
        duration = len(audio) / orig_sr
        new_length = int(duration * target_sr)
        
        # Simple resampling using linear interpolation
        indices = np.linspace(0, len(audio) - 1, new_length)
        resampled = np.interp(indices, np.arange(len(audio)), audio)
        
        return resampled.astype(np.float32)
    
    def save(self, dataset: Dataset, repo_id: str, split: str = "data") -> Path:
        """Save a dataset to disk as parquet.
        
        Creates the directory structure output_dir / repo_id and saves
        the dataset as a parquet file.
        
        Args:
            dataset: HF Dataset to save
            repo_id: Repository ID (e.g., "org/dataset-name")
            split: Split name for filename (e.g., "train", "validation")
        
        Returns:
            Path to saved parquet file
        
        Example:
            >>> processor.save(dataset, "my-org/my-dataset", split="train")
            PosixPath('output/my-org/my-dataset/train.parquet')
        """
        # Create output directory for this dataset
        save_dir = self.output_dir / repo_id
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Use split name for filename
        output_path = save_dir / f"{split}.parquet"
        
        # Save to parquet
        dataset.to_parquet(str(output_path))
        
        print(f"Saved dataset to {output_path}")
        
        return output_path
    
    def push_to_hub(self, dataset: Dataset, repo_id: str) -> None:
        """Push a dataset to Hugging Face Hub.
        
        Uploads the dataset to the Hugging Face Hub using push_to_hub().
        Requires authentication via HF_TOKEN environment variable or
        huggingface-cli login.
        
        Args:
            dataset: HF Dataset to push
            repo_id: Repository ID (e.g., "org/dataset-name")
        
        Raises:
            RuntimeError: If push fails (e.g., authentication error)
        
        Example:
            >>> processor.push_to_hub(dataset, "my-org/my-dataset")
        """
        try:
            dataset.push_to_hub(repo_id)
            print(f"Pushed dataset to https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to push to hub: {e}") from e