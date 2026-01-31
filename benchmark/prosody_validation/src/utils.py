"""
Utility functions for Prosody Validation Benchmark
"""

import base64
import csv
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)


def encode_audio_base64(audio_path: Path) -> str:
    """Encode an audio file to base64.

    Args:
        audio_path: Path to the audio file

    Returns:
        Base64 encoded string of the audio file
    """
    with open(audio_path, "rb") as f:
        audio_data = f.read()
        return base64.b64encode(audio_data).decode("utf-8")


def decode_base64_audio(base64_string: str, output_path: Path) -> None:
    """Decode base64 string to an audio file.

    Args:
        base64_string: Base64 encoded audio data
        output_path: Path to save the decoded audio file
    """
    audio_data = base64.b64decode(base64_string)
    with open(output_path, "wb") as f:
        f.write(audio_data)


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file and return a list of dictionaries.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of dictionaries, one per line
    """
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def write_jsonl(data: List[Dict[str, Any]], file_path: Path) -> None:
    """Write a list of dictionaries to a JSONL file.

    Args:
        data: List of dictionaries to write
        file_path: Path to the output file
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(item: Dict[str, Any], file_path: Path) -> None:
    """Append a single dictionary to a JSONL file.

    Args:
        item: Dictionary to append
        file_path: Path to the JSONL file
    """
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_csv(file_path: Path) -> List[Dict[str, str]]:
    """Read a CSV file and return a list of dictionaries.

    Args:
        file_path: Path to the CSV file

    Returns:
        List of dictionaries representing rows
    """
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def write_csv(
    data: List[Dict[str, Any]], file_path: Path, fieldnames: Optional[List[str]] = None
) -> None:
    """Write a list of dictionaries to a CSV file.

    Args:
        data: List of dictionaries to write
        file_path: Path to the output file
        fieldnames: List of column names (auto-derived from first dict if not provided)
    """
    if not data:
        logger.warning("Empty data provided to write_csv")
        return

    if fieldnames is None:
        fieldnames = list(data[0].keys())

    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """Save a dictionary to a JSON file.

    Args:
        data: Dictionary to save
        file_path: Path to the output file
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load a dictionary from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Loaded dictionary
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_file_hash(file_path: Path) -> str:
    """Get the MD5 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def validate_audio_file(file_path: Path) -> bool:
    """Validate that a file is a valid audio file.

    Args:
        file_path: Path to the file

    Returns:
        True if valid audio file, False otherwise
    """
    if not file_path.exists():
        return False

    # Check file extension
    valid_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    if file_path.suffix.lower() not in valid_extensions:
        return False

    # Check file size (minimum 100 bytes)
    if file_path.stat().st_size < 100:
        return False

    return True


class ProgressTracker:
    """Simple progress tracker with logging."""

    def __init__(self, description: str, total: int, unit: str = "items"):
        """Initialize the progress tracker.

        Args:
            description: Description of the task
            total: Total number of items
            unit: Unit of measurement (e.g., "items", "files")
        """
        self.description = description
        self.total = total
        self.unit = unit
        self.progress_bar = None
        self.completed = 0

    def __enter__(self):
        """Enter the context manager."""
        self.progress_bar = tqdm_asyncio(
            total=self.total, desc=self.description, unit=self.unit, dynamic_ncols=True
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        if self.progress_bar:
            self.progress_bar.close()

    def update(self, n: int = 1) -> None:
        """Update the progress bar.

        Args:
            n: Number of items to add
        """
        if self.progress_bar:
            self.progress_bar.update(n)
            self.completed += n

    def set_description(self, desc: str) -> None:
        """Set the description of the progress bar.

        Args:
            desc: New description
        """
        if self.progress_bar:
            self.progress_bar.set_description(desc)


def setup_logging(
    log_level: str = "INFO", log_file: Optional[Path] = None
) -> logging.Logger:
    """Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file

    Returns:
        Configured logger
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure handlers
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=numeric_level, format=log_format, handlers=handlers)

    return logging.getLogger(__name__)


def batch_iterator(data: List[Any], batch_size: int) -> Iterator[List[Any]]:
    """Yield batches from a list.

    Args:
        data: List to batch
        batch_size: Size of each batch

    Yields:
        Batches of the specified size
    """
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def chunk_list(data: List[Any], num_chunks: int) -> List[List[Any]]:
    """Split a list into roughly equal chunks.

    Args:
        data: List to split
        num_chunks: Number of chunks

    Returns:
        List of chunks
    """
    if num_chunks <= 0:
        return [data]

    chunk_size = (len(data) + num_chunks - 1) // num_chunks
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
