"""
Step 0: Merge Annotation and Dialogue data, then sample 200 rows
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.utils import setup_logging, write_csv

logger = logging.getLogger(__name__)


def load_annotation_data(file_path: Path) -> pd.DataFrame:
    """Load the annotation CSV file.

    Args:
        file_path: Path to the annotation CSV

    Returns:
        DataFrame with annotation data
    """
    logger.info(f"Loading annotation data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} annotation rows")
    logger.info(f"Annotation columns: {list(df.columns)}")
    return df


def load_dialogue_data(file_path: Path) -> pd.DataFrame:
    """Load the dialogue TSV file.

    Args:
        file_path: Path to the dialogue TSV

    Returns:
        DataFrame with dialogue data
    """
    logger.info(f"Loading dialogue data from {file_path}")
    df = pd.read_csv(file_path, sep="\t")
    logger.info(f"Loaded {len(df)} dialogue rows")
    logger.info(f"Dialogue columns: {list(df.columns)}")
    return df


def merge_data(annotation_df: pd.DataFrame, dialogue_df: pd.DataFrame) -> pd.DataFrame:
    """Merge annotation and dialogue data on dialog_id.

    Args:
        annotation_df: Annotation DataFrame
        dialogue_df: Dialogue DataFrame

    Returns:
        Merged DataFrame
    """
    # Check for dialog_id columns
    annotation_id_col = "dialog_id" if "dialog_id" in annotation_df.columns else None
    dialogue_id_col = "dialog_id" if "dialog_id" in dialogue_df.columns else None

    if not annotation_id_col or not dialogue_id_col:
        # Try to find the ID column
        possible_ids = ["dialog_id", "id", "Dialog_ID", "ID", "dialog"]
        for pid in possible_ids:
            if pid in annotation_df.columns:
                annotation_id_col = pid
                break
        for pid in possible_ids:
            if pid in dialogue_df.columns:
                dialogue_id_col = pid
                break

    if not annotation_id_col or not dialogue_id_col:
        logger.error("Could not find dialog_id column")
        raise ValueError("Could not find dialog_id column in either file")

    logger.info(
        f"Using '{annotation_id_col}' for annotation and '{dialogue_id_col}' for dialogue"
    )

    # Rename columns to standardize
    annotation_df = annotation_df.rename(columns={annotation_id_col: "dialog_id"})
    dialogue_df = dialogue_df.rename(columns={dialogue_id_col: "dialog_id"})

    # Merge on dialog_id
    merged_df = pd.merge(
        dialogue_df,
        annotation_df,
        on="dialog_id",
        how="inner",
        suffixes=("_dialogue", "_annotation"),
    )

    logger.info(f"Merged data: {len(merged_df)} rows")

    return merged_df


def filter_implicature(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for rows with non-empty implicature_text.

    Args:
        df: Input DataFrame

    Returns:
        Filtered DataFrame
    """
    # Check for implicature_text column
    implicature_col = None
    possible_cols = ["implicature_text", "implicature", "Implicature"]
    for col in possible_cols:
        if col in df.columns:
            implicature_col = col
            break

    if not implicature_col:
        logger.warning("Could not find implicature_text column, skipping filter")
        return df

    logger.info(f"Filtering for non-empty '{implicature_col}'")

    # Filter out empty or NaN implicature_text
    df_filtered = df[df[implicature_col].notna()].copy()
    df_filtered = df_filtered[df_filtered[implicature_col].str.strip() != ""]

    logger.info(f"After filtering: {len(df_filtered)} rows with non-empty implicature")

    return df_filtered


def get_stratification_columns(df: pd.DataFrame) -> List[str]:
    """Get the maxim columns for stratification.

    Args:
        df: Input DataFrame

    Returns:
        List of maxim column names
    """
    possible_cols = [
        "maxim_quality",
        "maxim_quantity",
        "maxim_relevance",
        "maxim_manner",
    ]
    return [col for col in possible_cols if col in df.columns]


def stratify_sample(
    df: pd.DataFrame, n_samples: int = 200, random_state: int = 42
) -> pd.DataFrame:
    """Sample with stratification on maxim columns.

    Args:
        df: Input DataFrame
        n_samples: Number of samples to select
        random_state: Random seed

    Returns:
        Sampled DataFrame
    """
    maxim_cols = get_stratification_columns(df)

    if not maxim_cols:
        logger.warning(
            "No maxim columns found for stratification, using random sampling"
        )
        return df.sample(n=min(n_samples, len(df)), random_state=random_state)

    logger.info(f"Stratifying on columns: {maxim_cols}")

    # Create a combined stratification label
    df["_stratify_key"] = (
        df[maxim_cols].fillna("none").astype(str).agg("_".join, axis=1)
    )

    # Calculate sampling proportions
    stratify_counts = df["_stratify_key"].value_counts()
    logger.info(f"Found {len(stratify_counts)} unique stratification groups")

    # Ensure we have enough samples
    if len(df) < n_samples:
        logger.warning(f"Only {len(df)} rows available, returning all")
        return df.drop(columns=["_stratify_key"])

    # Use train_test_split for stratified sampling
    try:
        _, sampled_df = train_test_split(
            df,
            train_size=n_samples,
            stratify=df["_stratify_key"],
            random_state=random_state,
        )
        logger.info(f"Sampled {len(sampled_df)} rows with stratification")
    except ValueError as e:
        logger.warning(f"Stratified sampling failed: {e}, using random sampling")
        sampled_df = df.sample(n=n_samples, random_state=random_state)

    # Remove temporary column
    sampled_df = sampled_df.drop(columns=["_stratify_key"])

    # Log distribution of maxim columns
    for col in maxim_cols:
        logger.info(f"Distribution of {col}:")
        dist = sampled_df[col].value_counts()
        for val, count in dist.items():
            logger.info(f"  {val}: {count}")

    return sampled_df


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order the required output columns.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with selected columns
    """
    required_cols = [
        "dialog_id",
        "utterance",
        "speech_act",
        "intent",
        "emotion",
        "maxim_quality",
        "maxim_quantity",
        "maxim_relevance",
        "maxim_manner",
        "implicature_text",
        "confidence",
    ]

    # Get available columns
    available_cols = []
    for col in required_cols:
        if col in df.columns:
            available_cols.append(col)
        else:
            logger.warning(f"Column '{col}' not found in data")

    # Also include any additional columns that might be useful
    all_cols = list(df.columns)
    extra_cols = [
        col
        for col in all_cols
        if col not in required_cols and col not in available_cols
    ]

    logger.info(f"Selected {len(available_cols)} required columns")
    if extra_cols:
        logger.info(f"Extra columns available: {extra_cols}")

    return df[available_cols].copy()


def run_step0(
    annotation_path: Path,
    dialogue_path: Path,
    output_path: Path,
    n_samples: int = 200,
    random_state: int = 42,
) -> bool:
    """Execute Step 0: Merge and sample data.

    Args:
        annotation_path: Path to the annotation CSV
        dialogue_path: Path to the dialogue TSV
        output_path: Path to save the sampled data
        n_samples: Number of samples to select
        random_state: Random seed

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info("STEP 0: Merge and Sample Data")
    logger.info("=" * 60)

    try:
        # Load data
        annotation_df = load_annotation_data(annotation_path)
        dialogue_df = load_dialogue_data(dialogue_path)

        # Merge data
        merged_df = merge_data(annotation_df, dialogue_df)

        # Filter for non-empty implicature
        filtered_df = filter_implicature(merged_df)

        # Sample with stratification
        sampled_df = stratify_sample(filtered_df, n_samples, random_state)

        # Select output columns
        output_df = select_output_columns(sampled_df)

        # Save output
        write_csv(output_df.to_dict("records"), output_path, list(output_df.columns))
        logger.info(f"Saved {len(output_df)} sampled rows to {output_path}")

        # Log summary
        logger.info("=" * 60)
        logger.info("STEP 0 COMPLETED SUCCESSFULLY")
        logger.info(f"Output: {output_path}")
        logger.info(f"Total samples: {len(output_df)}")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Step 0 failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point for Step 0."""
    parser = argparse.ArgumentParser(description="Step 0: Merge and Sample Data")
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path(__file__).parent.parent.parent
        / "data"
        / "(2000 samples) merged_output.xlsx - Annotation.csv",
        help="Path to annotation CSV file",
    )
    parser.add_argument(
        "--dialogue",
        type=Path,
        default=Path(__file__).parent.parent.parent
        / "data"
        / "(2000 samples) merged_output.xlsx - Dialogue.tsv",
        help="Path to dialogue TSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "step0_sampled_200.csv",
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--n-samples", type=int, default=200, help="Number of samples to select"
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Run step
    success = run_step0(
        annotation_path=args.annotation,
        dialogue_path=args.dialogue,
        output_path=args.output,
        n_samples=args.n_samples,
        random_state=args.random_state,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
