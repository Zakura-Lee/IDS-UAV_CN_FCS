"""UAV-NIDD preprocessing pipeline for the three official datasets.

This script loads the official UAV-NIDD CSV files configured in config/config.py,
cleans them, performs stratified splitting, optionally applies SMOTE on the
training split only, standardizes features with one shared StandardScaler,
and saves processed arrays plus a report.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "configs") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "configs"))

try:
    import configs.config as project_config
except ImportError:  # pragma: no cover
    try:
        from config import config as project_config
    except ImportError:  # pragma: no cover
        from config.config import *  # type: ignore


class DataPreprocessor:
    """Preprocess UAV-NIDD CSV datasets for deep-learning training."""

    def __init__(self, config_module: Any | None = None) -> None:
        self.config = config_module or project_config
        self.raw_data_dir = Path(getattr(self.config, "RAW_DATA_DIR", ROOT_DIR / "data" / "raw"))
        self.processed_data_dir = Path(getattr(self.config, "PROCESSED_DATA_DIR", ROOT_DIR / "data" / "processed"))
        self.train_config = getattr(self.config, "TRAIN_CONFIG", {})
        self.model_config = getattr(self.config, "MODEL_CONFIG", {})
        self.logger = self._setup_logger()
        self.label_column: Optional[str] = None

    def _setup_logger(self) -> logging.Logger:
        log_dir = ROOT_DIR / "models" / "outputs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("uav_preprocessor")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        handler = logging.FileHandler(log_dir / "preprocess.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        return logger

    def _expected_feature_dim(self, scenario_name: str) -> Optional[int]:
        mapping = {"uav_case1": 45, "ap_case2": 51, "gcs_case3": 85}
        return mapping.get(scenario_name)

    def _resolve_csv_path(self, csv_path: str | Path) -> Path:
        path = Path(csv_path)
        candidates = [path]
        if not path.is_absolute():
            candidates.extend([self.raw_data_dir / path, self.raw_data_dir / path.name, ROOT_DIR / path])

        for candidate in candidates:
            if candidate.exists():
                return candidate

        normalized_target = "".join(ch.lower() for ch in path.stem if ch.isalnum())
        for candidate in sorted(self.raw_data_dir.glob("*.csv")):
            normalized_candidate = "".join(ch.lower() for ch in candidate.stem if ch.isalnum())
            if normalized_candidate == normalized_target:
                return candidate
            if normalized_target and normalized_candidate.startswith(normalized_target):
                return candidate
            if normalized_target and (normalized_target in normalized_candidate or normalized_candidate in normalized_target):
                return candidate
            if "case2" in normalized_target and ("case2" in normalized_candidate or "accesspoint" in normalized_candidate or "access" in normalized_candidate):
                return candidate
            if "case3" in normalized_target and ("case3" in normalized_candidate or "gcs" in normalized_candidate):
                return candidate

        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    def load_data(self, csv_path: str | Path) -> pd.DataFrame:
        resolved_path = self._resolve_csv_path(csv_path)
        self.logger.info("Loading dataset from %s", resolved_path)
        try:
            df = pd.read_csv(resolved_path, encoding="latin1", low_memory=False)
        except Exception as exc:  # pragma: no cover
            self.logger.exception("Failed to read %s", resolved_path)
            raise RuntimeError(f"Failed to load dataset {resolved_path}: {exc}") from exc

        print(f"Loaded {resolved_path}")
        print(f"Shape: {df.shape}")
        print("Columns:", list(df.columns))
        print(df.head(3).to_string(index=False))
        return df

    def detect_label_column(self, df: pd.DataFrame) -> Optional[str]:
        for candidate in ("Label", "attack_type", "label"):
            if candidate in df.columns:
                return candidate
        for column in df.columns:
            name = str(column).strip().lower()
            if name in {"label", "attack_type", "attack", "target", "class", "normal"}:
                return column
        if df.shape[1] > 0:
            last_column = df.columns[-1]
            if last_column is not None:
                return last_column
        return None

    def validate_feature_dimension(self, scenario_name: str, df: pd.DataFrame) -> None:
        expected = self._expected_feature_dim(scenario_name)
        if expected is None:
            return
        feature_count = df.shape[1] - 1 if self.label_column else df.shape[1]
        if feature_count != expected:
            self.logger.warning(
                "Feature dimension mismatch for %s: expected %s, got %s",
                scenario_name,
                expected,
                feature_count,
            )
            print(f"Warning: expected {expected} features for {scenario_name}, got {feature_count}")
        else:
            print(f"Feature count matches expected value {expected} for {scenario_name}")

    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        cleaned = df.copy()
        self.label_column = self.detect_label_column(cleaned)
        if self.label_column is None:
            raise ValueError("No label column detected")

        print("Detected label column:", self.label_column)

        # Remove fully redundant columns: all same values or all unique values
        for column in cleaned.columns:
            if column == self.label_column:
                continue
            if cleaned[column].nunique(dropna=True) <= 1 or cleaned[column].is_unique:
                cleaned = cleaned.drop(columns=[column])

        print(f"After removing redundant columns: {cleaned.shape}")

        # Report missing values
        missing_counts = cleaned.isna().sum()
        print("Missing values per column:")
        print(missing_counts[missing_counts > 0].to_string())

        # Replace inf with NaN, fill numeric missing values with median
        for column in cleaned.columns:
            if column == self.label_column:
                continue
            if pd.api.types.is_numeric_dtype(cleaned[column]):
                cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
                cleaned[column] = cleaned[column].replace([np.inf, -np.inf], np.nan)
                cleaned[column] = cleaned[column].fillna(cleaned[column].median())
            else:
                cleaned[column] = cleaned[column].astype("category").cat.codes
                cleaned[column] = cleaned[column].fillna(-1)

        X = cleaned.drop(columns=[self.label_column])
        y = cleaned[self.label_column]

        # Encode labels to integers for model training
        if not pd.api.types.is_numeric_dtype(y):
            y = y.astype("category").cat.codes

        print("Label distribution:")
        print(pd.Series(y).value_counts().to_string())
        return X.astype(float), pd.Series(y, dtype="int64")

    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        random_state = int(self.train_config.get("random_state", 42))
        test_size = float(self.train_config.get("test_size", 0.2))
        val_size = float(self.train_config.get("val_size", 0.2))

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_size,
            random_state=random_state,
            stratify=y_train_val,
        )

        print("Split sizes:")
        print(f"train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def apply_smote(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        use_smote = bool(self.train_config.get("use_smote", True))
        if not use_smote:
            print("SMOTE disabled by config")
            return X_train, y_train

        print("Applying SMOTE to training set only...")
        try:
            class_counts = y_train.value_counts()
            min_class_count = int(class_counts.min())
            if min_class_count <= 1:
                print("SMOTE skipped because a class has too few samples")
                return X_train, y_train

            k_neighbors = min(5, max(1, min_class_count - 1))
            smote = SMOTE(random_state=int(self.train_config.get("random_state", 42)), k_neighbors=k_neighbors)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            print("SMOTE before/after distribution:")
            print(pd.Series(y_train).value_counts().to_string())
            print(pd.Series(y_resampled).value_counts().to_string())
            return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)
        except Exception as exc:
            self.logger.exception("SMOTE failed")
            print(f"Warning: SMOTE failed, continuing without it: {exc}")
            return X_train, y_train

    def standardize_features(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled

    def save_processed_data(
        self,
        scenario_name: str,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        scaler: StandardScaler,
        feature_names: List[str],
    ) -> None:
        save_dir = self.processed_data_dir / scenario_name
        save_dir.mkdir(parents=True, exist_ok=True)

        np.save(save_dir / f"{scenario_name}_X_train.npy", X_train)
        np.save(save_dir / f"{scenario_name}_y_train.npy", y_train)
        np.save(save_dir / f"{scenario_name}_X_val.npy", X_val)
        np.save(save_dir / f"{scenario_name}_y_val.npy", y_val)
        np.save(save_dir / f"{scenario_name}_X_test.npy", X_test)
        np.save(save_dir / f"{scenario_name}_y_test.npy", y_test)

        with open(save_dir / "scaler.pkl", "wb") as handle:
            pickle.dump(scaler, handle)

        with open(save_dir / "feature_names.txt", "w", encoding="utf-8") as handle:
            handle.write("\n".join(feature_names))

        print(f"Saved processed data to {save_dir}")

    def _report_filename(self, scenario_name: str) -> str:
        mapping = {"uav_case1": "case1_report.txt", "ap_case2": "case2_report.txt", "gcs_case3": "case3_report.txt"}
        return mapping.get(scenario_name, f"{scenario_name}_report.txt")

    def write_report(
        self,
        scenario_name: str,
        input_path: Path,
        original_shape: Tuple[int, int],
        cleaned_shape: Tuple[int, int],
        X_train: np.ndarray,
        y_train: np.ndarray,
        y_before: pd.Series,
        y_after: pd.Series,
    ) -> None:
        report_path = self.processed_data_dir / self._report_filename(scenario_name)
        lines = [
            f"Scenario: {scenario_name}",
            f"Processed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Input file: {input_path}",
            f"Original shape: {original_shape}",
            f"Cleaned shape: {cleaned_shape}",
            f"Training shape after SMOTE: {X_train.shape}",
            "",
            "Class distribution before SMOTE:",
            pd.Series(y_before).value_counts().to_string(),
            "",
            "Class distribution after SMOTE:",
            pd.Series(y_after).value_counts().to_string(),
        ]
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Report saved to {report_path}")

    def preprocess_scenario(self, scenario_name: str, csv_path: str | Path) -> None:
        try:
            df = self.load_data(csv_path)
            self.label_column = self.detect_label_column(df)
            self.validate_feature_dimension(scenario_name, df)
            print(f"Feature dimension for {scenario_name}: {df.shape[1] - 1 if self.label_column else df.shape[1]}")

            X, y = self.clean_data(df)
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

            # SMOTE is applied only to the training split.
            X_train_resampled, y_train_resampled = self.apply_smote(X_train, y_train)

            X_train_scaled, X_val_scaled, X_test_scaled = self.standardize_features(
                X_train_resampled,
                X_val,
                X_test,
            )

            if torch is not None:
                X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
                X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
                X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train_resampled.to_numpy(), dtype=torch.long)
                y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.long)
                y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)
                x_train_out = X_train_tensor.numpy()
                x_val_out = X_val_tensor.numpy()
                x_test_out = X_test_tensor.numpy()
                y_train_out = y_train_tensor.numpy()
                y_val_out = y_val_tensor.numpy()
                y_test_out = y_test_tensor.numpy()
            else:
                x_train_out = X_train_scaled
                x_val_out = X_val_scaled
                x_test_out = X_test_scaled
                y_train_out = y_train_resampled.to_numpy()
                y_val_out = y_val.to_numpy()
                y_test_out = y_test.to_numpy()

            scaler = StandardScaler()
            scaler.fit(X_train_resampled)
            self.save_processed_data(
                scenario_name,
                x_train_out,
                x_val_out,
                x_test_out,
                y_train_out,
                y_val_out,
                y_test_out,
                scaler,
                list(X.columns),
            )
            self.write_report(
                scenario_name,
                self._resolve_csv_path(csv_path),
                df.shape,
                X.shape,
                x_train_out,
                y_train_out,
                y_train,
                y_train_resampled,
            )
        except Exception as exc:
            self.logger.exception("Failed to preprocess %s", scenario_name)
            print(f"Error processing {scenario_name}: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess UAV-NIDD datasets")
    parser.add_argument("--scenario", default="all", help="Scenario key to process: uav_case1, ap_case2, gcs_case3, or all")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    preprocessor = DataPreprocessor()
    config_files = getattr(preprocessor.config, "UAV_NIDD_FILES", {})

    official_names = {"uav_case1": "UAV-Case1-Label.csv", "ap_case2": "Access Point Case2 Label.csv", "gcs_case3": "GSC Case3 Label .csv"}
    selected = []
    if args.scenario == "all":
        selected = [(name, config_files[name]) for name in official_names if name in config_files]
    elif args.scenario in config_files:
        selected = [(args.scenario, config_files[args.scenario])]
    else:
        print(f"Unknown scenario {args.scenario}")
        return

    for scenario_name, csv_path in selected:
        print(f"\nProcessing {scenario_name}...")
        preprocessor.preprocess_scenario(scenario_name, csv_path)


if __name__ == "__main__":
    main()
