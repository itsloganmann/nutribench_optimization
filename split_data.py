"""Download and split the NutriBench dataset into train/validation CSV files.

This script follows the project specification:

* Download the ``NutriBench`` dataset (``dongx1997/NutriBench``) from Hugging Face.
* Save the full CoT-annotated dataset for provenance.
* Create a fixed validation set with at least 1000 examples (default) while preserving
  reproducibility via a deterministic random seed.
* Save ``train.csv`` and ``val.csv`` inside the ``data/`` directory.

Run this script **after** creating and activating the project virtual environment.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, cast

import pandas as pd
import typer
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split


DEFAULT_DATA_DIR = Path("data")
DATASET_NAME = "dongx1997/NutriBench"
DATASET_CONFIG = "v2"
DEFAULT_VAL_SIZE = 1000
DEFAULT_SEED = 42

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _setup_logging(verbose: bool) -> None:
	logging.basicConfig(
		level=logging.INFO if verbose else logging.WARNING,
		format="%(asctime)s | %(levelname)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)


def _download_dataset(cache_dir: Path | None = None, force: bool = False) -> pd.DataFrame:
	"""Download the NutriBench dataset and return it as a :class:`pandas.DataFrame`.

	Parameters
	----------
	cache_dir:
		Optional path where the Hugging Face dataset cache should live.
	force:
		When ``True`` the dataset is re-downloaded even if a cached copy exists.
	"""

	logging.info("Loading dataset '%s' (config=%s)...", DATASET_NAME, DATASET_CONFIG)
	dataset = cast(
		DatasetDict,
		load_dataset(  # type: ignore[call-arg]
			DATASET_NAME,
			DATASET_CONFIG,
			cache_dir=str(cache_dir) if cache_dir else None,
			download_mode="force_redownload" if force else None,
		),
	)
	if "train" not in dataset:
		msg = "Expected a 'train' split in the NutriBench dataset."
		raise ValueError(msg)

	train_split = cast(Dataset, dataset["train"])
	df = cast(pd.DataFrame, train_split.to_pandas())
	logging.info("Dataset loaded with %d rows and %d columns", df.shape[0], df.shape[1])
	expected_columns = {"meal_description", "carb"}
	missing = expected_columns.difference(df.columns)
	if missing:
		msg = f"Dataset is missing required columns: {sorted(missing)}"
		raise ValueError(msg)
	return df


def _write_metadata(path: Path, metadata: dict[str, object]) -> None:
	path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


@app.command()
def main(
	output_dir: Path = typer.Option(
		DEFAULT_DATA_DIR,
		"--output-dir",
		"-o",
		help="Directory where the CSV files will be saved.",
	),
	val_size: int = typer.Option(
		DEFAULT_VAL_SIZE,
		"--val-size",
		help="Number of samples to include in the validation split (minimum 1000).",
	),
	seed: int = typer.Option(
		DEFAULT_SEED,
		"--seed",
		help="Random seed for deterministic splitting.",
	),
	overwrite: bool = typer.Option(
		False,
		"--overwrite/--no-overwrite",
		help="Overwrite existing CSV files if they already exist.",
	),
	force_download: bool = typer.Option(
		False,
		"--force-download/--no-force-download",
		help="Force re-download of the dataset instead of using the cache.",
	),
	verbose: bool = typer.Option(
		False,
		"--verbose/--quiet",
		help="Display detailed log output.",
	),
	cache_dir: Optional[Path] = typer.Option(
		None,
		"--cache-dir",
		help="Optional directory for the Hugging Face dataset cache.",
	),
) -> None:
	"""Entry point for downloading and splitting the NutriBench dataset."""

	if val_size < DEFAULT_VAL_SIZE:
		raise typer.BadParameter(
			f"Validation size must be at least {DEFAULT_VAL_SIZE} samples.",
			param_hint="--val-size",
		)

	_setup_logging(verbose)
	output_dir.mkdir(parents=True, exist_ok=True)

	raw_csv = output_dir / "nutribench_v2_cot.csv"
	train_csv = output_dir / "train.csv"
	val_csv = output_dir / "val.csv"
	metadata_path = output_dir / "metadata.json"

	if not overwrite and train_csv.exists() and val_csv.exists():
		logging.warning(
			"train.csv and val.csv already exist. Use --overwrite to regenerate them."
		)
		raise typer.Exit(code=0)

	df = _download_dataset(cache_dir=cache_dir, force=force_download)

	if not overwrite and raw_csv.exists():
		logging.info("Raw dataset already present at %s; keeping existing file.", raw_csv)
	else:
		logging.info("Saving raw dataset to %s", raw_csv)
		df.to_csv(raw_csv, index=False)

	if len(df) <= val_size:
		raise typer.BadParameter(
			"Validation size must be smaller than the number of available samples.",
			param_hint="--val-size",
		)

	logging.info("Creating deterministic train/validation split (seed=%d)...", seed)
	train_df, val_df = cast(
		tuple[pd.DataFrame, pd.DataFrame],
		train_test_split(
			df,
			test_size=val_size,
			random_state=seed,
			shuffle=True,
			stratify=None,
		),
	)

	logging.info("Saving train split to %s (%d rows)", train_csv, len(train_df))
	train_df.to_csv(train_csv, index=False)

	logging.info("Saving validation split to %s (%d rows)", val_csv, len(val_df))
	val_df.to_csv(val_csv, index=False)

	metadata = {
		"dataset_name": DATASET_NAME,
		"dataset_config": DATASET_CONFIG,
		"seed": seed,
		"val_size": val_size,
		"train_size": len(train_df),
		"total_rows": len(df),
		"carb_stats": {
			"train": {
				"mean": float(train_df["carb"].mean()),
				"std": float(train_df["carb"].std(ddof=0)),
				"min": float(train_df["carb"].min()),
				"max": float(train_df["carb"].max()),
			},
			"val": {
				"mean": float(val_df["carb"].mean()),
				"std": float(val_df["carb"].std(ddof=0)),
				"min": float(val_df["carb"].min()),
				"max": float(val_df["carb"].max()),
			},
		},
	}
	logging.info("Writing metadata to %s", metadata_path)
	_write_metadata(metadata_path, metadata)

	typer.secho(
		"Dataset preparation complete! train.csv and val.csv are ready in the data directory.",
		fg=typer.colors.GREEN,
	)


if __name__ == "__main__":
	app()
