"""Summary reporting module for Amy-LM preprocessing pipeline.

This module provides the ProcessingSummary dataclass and generate_report function
for tracking preprocessing statistics and generating JSON reports.

The ProcessingSummary collects:
- Total utterances processed and failed
- Frame counts for content, prosody, and timbre codebooks
- Duration histogram for audio length distribution
- Failed sample IDs with error messages

Example:
    >>> from preprocessing.reporting import ProcessingSummary, generate_report
    >>> from pathlib import Path
    >>> 
    >>> summary = ProcessingSummary()
    >>> summary.add_processed(
    ...     content_frames=25,
    ...     prosody_frames=25,
    ...     timbre_frames=25,
    ...     duration_sec=2.0
    ... )
    >>> summary.add_failed(sample_id="sample_001", error="Encoding failed")
    >>> 
    >>> generate_report(summary, Path("report.json"))
"""

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import List, Dict, Any


@dataclass
class ProcessingSummary:
    """Tracks preprocessing statistics for a batch of audio samples.
    
    This dataclass collects statistics during the preprocessing pipeline:
    - Count of successfully processed and failed samples
    - Frame counts for each codebook type (content, prosody, timbre)
    - Audio duration distribution
    - Details of failed samples with error messages
    
    Attributes:
        total_processed: Number of successfully processed samples
        total_failed: Number of samples that failed processing
        content_frame_counts: List of frame counts for content codebooks
        prosody_frame_counts: List of frame counts for prosody codebooks
        timbre_frame_counts: List of frame counts for timbre codebooks
        durations_seconds: List of audio durations in seconds
        failed_samples: List of dicts with failed sample IDs and errors
    
    Example:
        >>> summary = ProcessingSummary()
        >>> summary.add_processed(content_frames=25, prosody_frames=25, 
        ...                       timbre_frames=25, duration_sec=2.0)
        >>> summary.add_failed(sample_id="bad_sample", error="Empty audio")
        >>> report = summary.to_dict()
    """
    
    total_processed: int = 0
    total_failed: int = 0
    content_frame_counts: List[int] = field(default_factory=list)
    prosody_frame_counts: List[int] = field(default_factory=list)
    timbre_frame_counts: List[int] = field(default_factory=list)
    durations_seconds: List[float] = field(default_factory=list)
    failed_samples: List[Dict[str, str]] = field(default_factory=list)
    
    def add_processed(
        self,
        content_frames: int,
        prosody_frames: int,
        timbre_frames: int,
        duration_sec: float
    ) -> None:
        """Add a successfully processed sample to the summary.
        
        Increments the total_processed counter and appends the frame counts
        and duration to their respective lists.
        
        Args:
            content_frames: Number of content codebook frames
            prosody_frames: Number of prosody codebook frames
            timbre_frames: Number of timbre codebook frames
            duration_sec: Audio duration in seconds
        
        Example:
            >>> summary = ProcessingSummary()
            >>> summary.add_processed(25, 25, 25, 2.0)
            >>> summary.total_processed
            1
        """
        self.total_processed += 1
        self.content_frame_counts.append(content_frames)
        self.prosody_frame_counts.append(prosody_frames)
        self.timbre_frame_counts.append(timbre_frames)
        self.durations_seconds.append(duration_sec)
    
    def add_failed(self, sample_id: str, error: str) -> None:
        """Add a failed sample to the summary.
        
        Increments the total_failed counter and stores the sample ID
        and error message.
        
        Args:
            sample_id: Unique identifier of the failed sample
            error: Error message describing why processing failed
        
        Example:
            >>> summary = ProcessingSummary()
            >>> summary.add_failed("sample_001", "Empty audio array")
            >>> summary.total_failed
            1
        """
        self.total_failed += 1
        self.failed_samples.append({"id": sample_id, "error": error})
    
    def _compute_average(self, values: List[float]) -> float:
        """Compute average of a list, returning 0.0 for empty lists.
        
        Args:
            values: List of numeric values
        
        Returns:
            Average value or 0.0 if list is empty
        """
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def _compute_duration_histogram(self) -> Dict[str, int]:
        """Compute duration histogram with predefined bins.
        
        Bins:
        - 0-1s: 0 to 1 second
        - 1-2s: 1 to 2 seconds
        - 2-5s: 2 to 5 seconds
        - 5-10s: 5 to 10 seconds
        - 10s+: 10 seconds and above
        
        Returns:
            Dictionary mapping bin labels to counts
        """
        bins = {
            "0-1s": 0,
            "1-2s": 0,
            "2-5s": 0,
            "5-10s": 0,
            "10s+": 0
        }
        
        for duration in self.durations_seconds:
            if duration < 1.0:
                bins["0-1s"] += 1
            elif duration < 2.0:
                bins["1-2s"] += 1
            elif duration < 5.0:
                bins["2-5s"] += 1
            elif duration < 10.0:
                bins["5-10s"] += 1
            else:
                bins["10s+"] += 1
        
        return bins
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary with computed statistics.
        
        Computes averages and duration histogram from collected data.
        
        Returns:
            Dictionary containing:
            - total_processed: Total successful samples
            - total_failed: Total failed samples
            - avg_content_frames: Average content frame count
            - avg_prosody_frames: Average prosody frame count
            - avg_timbre_frames: Average timbre frame count
            - avg_duration_sec: Average audio duration
            - duration_histogram: Duration distribution histogram
            - failed_samples: List of failed sample details
        
        Example:
            >>> summary = ProcessingSummary()
            >>> summary.add_processed(25, 25, 25, 2.0)
            >>> summary.to_dict()
            {'total_processed': 1, 'total_failed': 0, ...}
        """
        return {
            "total_processed": self.total_processed,
            "total_failed": self.total_failed,
            "avg_content_frames": self._compute_average(self.content_frame_counts),
            "avg_prosody_frames": self._compute_average(self.prosody_frame_counts),
            "avg_timbre_frames": self._compute_average(self.timbre_frame_counts),
            "avg_duration_sec": self._compute_average(self.durations_seconds),
            "duration_histogram": self._compute_duration_histogram(),
            "failed_samples": self.failed_samples.copy()
        }


def generate_report(summary: ProcessingSummary, output_path: Path) -> Path:
    """Generate a JSON report from a ProcessingSummary.
    
    Writes the summary statistics to a JSON file with indentation
    for readability. Creates parent directories if they don't exist.
    
    Args:
        summary: ProcessingSummary containing statistics
        output_path: Path to write the JSON report
    
    Returns:
        Path to the written report file
    
    Raises:
        IOError: If writing to the file fails
    
    Example:
        >>> summary = ProcessingSummary()
        >>> summary.add_processed(25, 25, 25, 2.0)
        >>> generate_report(summary, Path("output/report.json"))
        PosixPath('output/report.json')
    """
    # Ensure parent directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate the report dictionary
    report_data = summary.to_dict()
    
    # Write to JSON with indentation
    with open(output_path, "w") as f:
        json.dump(report_data, f, indent=2)
    
    return output_path
