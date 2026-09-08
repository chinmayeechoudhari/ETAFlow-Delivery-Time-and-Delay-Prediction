"""
Data Loader Module for ETAFlow Preprocessing Layer.

Safely loads raw shipment dataset without in-place modification.
Includes optional MD5 checksum verification against DVC tracking metadata.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def compute_file_md5(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute MD5 checksum of a file.

    Args:
        file_path: Path to the target file.
        chunk_size: Byte chunk size for reading large files.

    Returns:
        Hexadecimal MD5 digest string.
    """
    path = Path(file_path)
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def verify_dvc_checksum(
    data_path: Union[str, Path], dvc_path: Optional[Union[str, Path]] = None
) -> bool:
    """Verify that raw data file matches MD5 recorded in .dvc file.

    Args:
        data_path: Path to raw dataset.
        dvc_path: Path to DVC metadata file. If None, appends '.dvc' to data_path.

    Returns:
        True if checksum matches, raises ValueError if mismatch.
    """
    data_p = Path(data_path)
    dvc_p = Path(dvc_path) if dvc_path else data_p.with_name(f"{data_p.name}.dvc")

    if not dvc_p.exists():
        logger.warning("DVC metadata file not found at %s. Skipping checksum check.", dvc_p)
        return True

    with open(dvc_p, "r", encoding="utf-8") as f:
        dvc_meta = yaml.safe_load(f)

    expected_md5 = None
    if "outs" in dvc_meta and len(dvc_meta["outs"]) > 0:
        expected_md5 = dvc_meta["outs"][0].get("md5")

    if not expected_md5:
        logger.warning("No MD5 recorded in DVC metadata file %s.", dvc_p)
        return True

    actual_md5 = compute_file_md5(data_p)
    if actual_md5.lower() != expected_md5.lower():
        raise ValueError(
            f"Dataset integrity mismatch! Expected MD5 {expected_md5}, but found {actual_md5}."
        )

    logger.info("DVC checksum verified successfully (%s)", actual_md5)
    return True


class DataLoader:
    """Data Loader for reading raw shipment data."""

    def __init__(
        self,
        raw_data_path: Union[str, Path] = "data/raw/shipments.csv",
        verify_checksum: bool = True,
    ) -> None:
        """Initialize DataLoader.

        Args:
            raw_data_path: Path to raw CSV file.
            verify_checksum: Whether to verify MD5 against DVC metadata.
        """
        self.raw_data_path = Path(raw_data_path)
        self.verify_checksum = verify_checksum

    def load_data(
        self,
        parse_dates: bool = False,
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load raw shipments dataset.

        Args:
            parse_dates: If True, parses known date columns into datetime objects.
            nrows: Optional number of rows to load for quick testing.

        Returns:
            pd.DataFrame: Loaded dataset.
        """
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw dataset file not found at {self.raw_data_path}")

        if self.verify_checksum:
            verify_dvc_checksum(self.raw_data_path)

        date_cols = [
            "order_date",
            "pickup_datetime",
            "estimated_dispatch_datetime",
            "actual_dispatch_datetime",
            "delivery_datetime",
        ] if parse_dates else False

        logger.info("Loading raw dataset from %s (nrows=%s)...", self.raw_data_path, nrows)
        df = pd.read_csv(
            self.raw_data_path,
            parse_dates=date_cols,
            nrows=nrows,
            low_memory=False,
        )
        logger.info("Successfully loaded %d records and %d columns.", len(df), len(df.columns))
        return df
