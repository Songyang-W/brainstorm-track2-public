"""
Download Track 2 datasets from HuggingFace.

The datasets contain pre-generated neural data at four difficulty levels:
- super_easy: Crystal-clear signal, no noise (perfect for getting started)
- easy: Clean signal, minimal noise
- medium: Moderate noise and line interference
- hard: Challenging conditions similar to real-time evaluation

Each dataset contains:
- track2_data.parquet: Neural signals (n_samples × 1024 channels)
- ground_truth.parquet: Cursor kinematics and tuned region positions
- README.md: Detailed description of dataset parameters
"""

from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

# Dataset configuration
DATA_PATH = Path("./data")
DATASET_ID = "PrecisionNeuroscience/BrainStorm2026-track2"

# Available difficulty levels
DIFFICULTY_LEVELS = ["super_easy", "easy", "medium", "hard"]


def _download_file(
    filename: str,
    local_dir: Path | None = None,
    token: str | None = None,
) -> Path:
    """
    Download a file from the HuggingFace dataset.

    Args:
        filename: Path to file within the dataset (e.g., "easy/track2_data.parquet")
        local_dir: Local directory to save files (default: ./data)
        token: HuggingFace token (for private datasets)

    Returns:
        Path to the downloaded file
    """
    if local_dir is None:
        local_dir = DATA_PATH

    local_dir.mkdir(parents=True, exist_ok=True)

    return Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=local_dir,
            token=token,
        )
    )


def download_track2_data(
    difficulty: str | None = None,
    local_dir: Path | None = None,
    token: str | None = None,
) -> None:
    """
    Download Track 2 dataset(s) from HuggingFace.

    By default, downloads ALL difficulty levels. Specify a difficulty to download
    only that dataset.

    Args:
        difficulty: One of "super_easy", "easy", "medium", "hard", or None for all
        local_dir: Local directory to save files (default: ./data)
        token: HuggingFace token (for private datasets)

    Example:
        >>> download_track2_data()  # Download all datasets
        >>> download_track2_data("easy")  # Download only easy dataset
    """
    if local_dir is None:
        local_dir = DATA_PATH

    if difficulty is None:
        # Download all difficulty levels
        for diff in DIFFICULTY_LEVELS:
            print(f"Downloading {diff} dataset...")
            _download_difficulty(diff, local_dir, token)
        print(f"\nAll datasets downloaded to: {local_dir}")
    else:
        if difficulty not in DIFFICULTY_LEVELS:
            raise ValueError(
                f"Invalid difficulty '{difficulty}'. Must be one of: {DIFFICULTY_LEVELS}"
            )
        print(f"Downloading {difficulty} dataset...")
        _download_difficulty(difficulty, local_dir, token)
        print(f"\nDataset downloaded to: {local_dir / difficulty}")


def _download_difficulty(
    difficulty: str,
    local_dir: Path,
    token: str | None = None,
) -> None:
    """Download all files for a specific difficulty level."""
    _download_file(f"{difficulty}/track2_data.parquet", local_dir, token)
    _download_file(f"{difficulty}/ground_truth.parquet", local_dir, token)
    _download_file(f"{difficulty}/README.md", local_dir, token)


def load_track2_data(
    difficulty: str = "easy",
    local_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load previously downloaded Track 2 dataset.

    Args:
        difficulty: One of "super_easy", "easy", "medium", "hard"
        local_dir: Local directory where files are saved (default: ./data)

    Returns:
        Tuple of (neural_data DataFrame, ground_truth DataFrame)

    Example:
        >>> data, ground_truth = load_track2_data("easy")
        >>> print(data.shape)  # (n_samples, 1024)
        >>> print(ground_truth.columns)
    """
    if difficulty not in DIFFICULTY_LEVELS:
        raise ValueError(
            f"Invalid difficulty '{difficulty}'. Must be one of: {DIFFICULTY_LEVELS}"
        )

    if local_dir is None:
        local_dir = DATA_PATH

    data_path = local_dir / difficulty / "track2_data.parquet"
    gt_path = local_dir / difficulty / "ground_truth.parquet"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run download_track2_data('{difficulty}') first."
        )

    data = pd.read_parquet(data_path)
    ground_truth = pd.read_parquet(gt_path)

    return data, ground_truth


def load_all_difficulties(
    local_dir: Path | None = None,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load all difficulty levels of the Track 2 dataset.

    Args:
        local_dir: Local directory where files are saved (default: ./data)

    Returns:
        Dictionary mapping difficulty level to (data, ground_truth) tuples

    Example:
        >>> datasets = load_all_difficulties()
        >>> easy_data, easy_gt = datasets["easy"]
    """
    datasets = {}
    for difficulty in DIFFICULTY_LEVELS:
        datasets[difficulty] = load_track2_data(difficulty, local_dir)

    return datasets


def get_data_path(difficulty: str = "easy") -> Path:
    """
    Get the local path to a downloaded dataset directory.

    Args:
        difficulty: One of "super_easy", "easy", "medium", "hard"

    Returns:
        Path to the dataset directory

    Example:
        >>> path = get_data_path("easy")
        >>> # Use with brainstorm-stream:
        >>> # brainstorm-stream --from-file {path}
    """
    if difficulty not in DIFFICULTY_LEVELS:
        raise ValueError(
            f"Invalid difficulty '{difficulty}'. Must be one of: {DIFFICULTY_LEVELS}"
        )

    return DATA_PATH / difficulty


if __name__ == "__main__":
    # Simple CLI for downloading data
    import sys

    if len(sys.argv) > 1:
        difficulty = sys.argv[1]
        print(f"Downloading Track 2 {difficulty} dataset...")
        download_track2_data(difficulty)
    else:
        print("Downloading all Track 2 datasets...")
        download_track2_data()
