"""
Step 0: Load merged dataset and sample 200 rows

This step works directly with the pre-merged dataset that contains
both utterances and annotations.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.utils import setup_logging, write_csv

logger = logging.getLogger(__name__)


def load_merged_data(file_path: Path) -> pd.DataFrame:
    """Load the pre-merged dataset CSV.

    Args:
        file_path: Path to the merged dataset CSV

    Returns:
        DataFrame with all data (utterance + annotations)
    """
    logger.info(f"Loading merged dataset from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows")
    logger.info(f"Columns: {list(df.columns)}")
    return df


def filter_implicature(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for rows with non-empty implicature_text.

    Args:
        df: Input DataFrame

    Returns:
        Filtered DataFrame
    """
    implicature_col = "implicature_text"

    if implicature_col not in df.columns:
        logger.warning(f"Column '{implicature_col}' not found, skipping filter")
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
    input_path: Path,
    output_path: Path,
    n_samples: int = 200,
    random_state: int = 42,
) -> bool:
    """Execute Step 0: Load merged data and sample.

    Args:
        input_path: Path to the merged dataset CSV
        output_path: Path to save the sampled data
        n_samples: Number of samples to select
        random_state: Random seed

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info("STEP 0: Load Merged Dataset and Sample")
    logger.info("=" * 60)

    try:
        # Load merged data
        df = load_merged_data(input_path)

        # Filter for non-empty implicature
        filtered_df = filter_implicature(df)

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
        logger.info(f"Input: {input_path}")
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
    parser = argparse.ArgumentParser(
        description="Step 0: Load Merged Dataset and Sample"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent.parent.parent.parent
        / "data"
        / "merged_dataset.csv",
        help="Path to merged dataset CSV file",
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
        input_path=args.input,
        output_path=args.output,
        n_samples=args.n_samples,
        random_state=args.random_state,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
