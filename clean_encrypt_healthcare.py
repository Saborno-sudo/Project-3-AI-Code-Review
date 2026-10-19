#!/usr/bin/env python3

#####################################################
# Project 3 - AI Code Review
# Student: Sabrina Akter
# Date: October 18, 2026
#
# Purpose:
# Clean and validate a healthcare dataset, create
# summary statistics, pseudonymize patient names,
# and encrypt the cleaned dataset.
#
# Improvements:
# 1. Added improved input and error handling.
# 2. Added detailed data-cleaning feedback.
# 3. Improved console output and completion reporting.
#
# Security:
# The program does not write plaintext patient records
# to the output folder.
#####################################################

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import stat
from pathlib import Path

import numpy as np
import pandas as pd
from cryptography.fernet import Fernet


# Columns containing dates
DATE_COLUMNS = [
    "Date of Admission",
    "Discharge Date",
]

# Columns required for the program to work
REQUIRED_COLUMNS = [
    "Name",
    "Age",
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Date of Admission",
    "Doctor",
    "Hospital",
    "Insurance Provider",
    "Billing Amount",
    "Room Number",
    "Admission Type",
    "Discharge Date",
    "Medication",
    "Test Results",
]


def load_or_create_key(key_path: Path) -> bytes:
    """Load an existing encryption key or create a new one."""

    key_path.parent.mkdir(parents=True, exist_ok=True)

    if key_path.exists():
        key = key_path.read_bytes().strip()

        # Make sure the existing key is valid.
        Fernet(key)
        return key

    key = Fernet.generate_key()

    fd = os.open(
        str(key_path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )

    with os.fdopen(fd, "wb") as handle:
        handle.write(key + b"\n")

    try:
        os.chmod(key_path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass

    return key


def clean_data(input_path: Path) -> tuple[pd.DataFrame, dict]:
    """Read and clean the healthcare dataset."""

    try:
        df = pd.read_csv(input_path, low_memory=False)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find the input file: {input_path}"
        )
    except pd.errors.EmptyDataError:
        raise ValueError("The input CSV file is empty.")
    except pd.errors.ParserError:
        raise ValueError("The input CSV file could not be parsed correctly.")

    original_rows = len(df)
    original_columns = list(df.columns)

    # Remove extra whitespace from column names.
    df.columns = [str(column).strip() for column in df.columns]

    # Check that all required columns are present.
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Input file is missing expected columns: "
            + ", ".join(missing_columns)
        )

    # Remove extra whitespace from text fields.
    for column in df.select_dtypes(
        include=["object", "string"]
    ).columns:
        df[column] = df[column].astype("string").str.strip()
        df[column] = df[column].replace("", pd.NA)

    # -------------------------
    # Clean age
    # -------------------------

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    invalid_age = (
        df["Age"].notna()
        & ~df["Age"].between(0, 120)
    )

    df.loc[invalid_age, "Age"] = np.nan
    df["Age"] = df["Age"].round().astype("Int64")

    # -------------------------
    # Clean billing amounts
    # -------------------------

    df["Billing Amount"] = pd.to_numeric(
        df["Billing Amount"],
        errors="coerce",
    )

    invalid_billing = (
        df["Billing Amount"].notna()
        & (
            ~np.isfinite(df["Billing Amount"])
            | (df["Billing Amount"] < 0)
        )
    )

    df.loc[
        invalid_billing,
        "Billing Amount",
    ] = np.nan

    # -------------------------
    # Clean room numbers
    # -------------------------

    df["Room Number"] = pd.to_numeric(
        df["Room Number"],
        errors="coerce",
    )

    invalid_room = (
        df["Room Number"].notna()
        & (
            (df["Room Number"] < 0)
            | (df["Room Number"] % 1 != 0)
        )
    )

    df.loc[
        invalid_room,
        "Room Number",
    ] = np.nan

    df["Room Number"] = (
        df["Room Number"].astype("Int64")
    )

    # -------------------------
    # Clean dates
    # -------------------------

    for column in DATE_COLUMNS:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )

    # Discharge should not occur before admission.
    invalid_date_order = (
        df["Date of Admission"].notna()
        & df["Discharge Date"].notna()
        & (
            df["Discharge Date"]
            < df["Date of Admission"]
        )
    )

    invalid_date_order_count = int(
        invalid_date_order.sum()
    )

    df.loc[
        invalid_date_order,
        "Discharge Date",
    ] = pd.NaT

    # -------------------------
    # Remove duplicate records
    # -------------------------

    rows_before_dedup = len(df)

    df = (
        df.drop_duplicates()
        .reset_index(drop=True)
    )

    duplicates_removed = (
        rows_before_dedup - len(df)
    )

    # Standardize categorical fields.
    category_columns = [
        "Gender",
        "Blood Type",
        "Medical Condition",
        "Insurance Provider",
        "Admission Type",
        "Medication",
        "Test Results",
    ]

    for column in category_columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .str.title()
        )

    # Create a cleaning report.
    report = {
        "input_file": input_path.name,
        "original_rows": int(original_rows),
        "rows_after_cleaning": int(len(df)),
        "duplicates_removed": int(duplicates_removed),
        "invalid_age_values_set_missing": int(
            invalid_age.sum()
        ),
        "invalid_billing_values_set_missing": int(
            invalid_billing.sum()
        ),
        "invalid_room_numbers_set_missing": int(
            invalid_room.sum()
        ),
        "invalid_date_order_discharge_dates_set_missing":
            invalid_date_order_count,
        "columns_in_input": original_columns,
        "missing_values_by_column": {
            str(key): int(value)
            for key, value
            in df.isna().sum().items()
        },
    }

    return df, report


def add_patient_ids(
    df: pd.DataFrame,
    key: bytes,
) -> pd.DataFrame:
    """Replace patient names with pseudonymous IDs."""

    # Validate the key.
    Fernet(key)

    hmac_key = base64.urlsafe_b64decode(key)

    result = df.copy()

    names = (
        result["Name"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    # Create a pseudonymous ID for each record.
    result["Patient_ID"] = [
        hmac.new(
            hmac_key,
            f"{index}|{name.casefold()}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:20]
        for index, name in enumerate(names.tolist())
    ]

    # Remove patient names.
    result = result.drop(columns=["Name"])

    # Put Patient_ID first.
    columns = [
        "Patient_ID"
    ] + [
        column
        for column in result.columns
        if column != "Patient_ID"
    ]

    return result[columns]


def make_statistics(
    df: pd.DataFrame,
    cleaning_report: dict,
) -> dict:
    """Create aggregate statistics without patient-level data."""

    stats = dict(cleaning_report)

    def numeric_summary(series: pd.Series) -> dict:
        """Calculate basic statistics for numeric data."""

        values = (
            pd.to_numeric(series, errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )

        if values.size == 0:
            return {
                "n": 0,
                "mean": None,
                "median": None,
                "std_population": None,
                "min": None,
                "max": None,
                "q1": None,
                "q3": None,
            }

        return {
            "n": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std_population": float(
                np.std(values, ddof=0)
            ),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "q1": float(
                np.percentile(values, 25)
            ),
            "q3": float(
                np.percentile(values, 75)
            ),
        }

    stats["age"] = numeric_summary(df["Age"])

    stats["billing_amount"] = numeric_summary(
        df["Billing Amount"]
    )

    stats["category_counts"] = {}

    category_columns = [
        "Gender",
        "Blood Type",
        "Medical Condition",
        "Insurance Provider",
        "Admission Type",
        "Medication",
        "Test Results",
    ]

    for column in category_columns:
        stats["category_counts"][column] = {
            str(key): int(value)
            for key, value
            in df[column]
            .value_counts(dropna=False)
            .items()
        }

    # Calculate length of hospital stay.
    valid_dates = (
        df["Date of Admission"].notna()
        & df["Discharge Date"].notna()
    )

    if valid_dates.any():
        stay_days = (
            (
                df.loc[
                    valid_dates,
                    "Discharge Date",
                ]
                - df.loc[
                    valid_dates,
                    "Date of Admission",
                ]
            )
            .dt.total_seconds()
            .to_numpy(dtype=float)
            / 86400.0
        )

        stats["length_of_stay_days"] = (
            numeric_summary(
                pd.Series(stay_days)
            )
        )

    else:
        stats["length_of_stay_days"] = (
            numeric_summary(
                pd.Series(dtype=float)
            )
        )

    return stats


def main() -> None:
    """Run the healthcare cleaning and encryption program."""

    parser = argparse.ArgumentParser(
        description=(
            "Clean, pseudonymize, summarize, "
            "and encrypt a healthcare dataset."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("healthcare_dataset.csv"),
        help="Path to the healthcare CSV file.",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("secure_output"),
        help="Directory for encrypted output files.",
    )

    parser.add_argument(
        "--key-file",
        type=Path,
        default=None,
        help=(
            "Path to the Fernet key. "
            "Defaults to OUTDIR/fernet.key."
        ),
    )

    args = parser.parse_args()

    # Improvement: clearer input validation.
    if not args.input.exists():
        print(
            f"ERROR: Input file not found: "
            f"{args.input}"
        )
        return

    if not args.input.is_file():
        print(
            f"ERROR: Input path is not a file: "
            f"{args.input}"
        )
        return

    try:
        args.outdir.mkdir(
            parents=True,
            exist_ok=True,
        )

        key_path = (
            args.key_file
            or (args.outdir / "fernet.key")
        )

        key = load_or_create_key(key_path)

        cleaned, cleaning_report = clean_data(
            args.input
        )

        stats = make_statistics(
            cleaned,
            cleaning_report,
        )

        pseudonymized = add_patient_ids(
            cleaned,
            key,
        )

        # Convert the cleaned dataset to CSV in memory.
        # No plaintext patient CSV is saved to disk.
        plaintext_csv = (
            pseudonymized.to_csv(
                index=False,
                date_format="%Y-%m-%d",
            ).encode("utf-8")
        )

        # Encrypt the complete dataset.
        encrypted_bytes = Fernet(key).encrypt(
            plaintext_csv
        )

        encrypted_path = (
            args.outdir
            / "patient_data.csv.fernet"
        )

        encrypted_path.write_bytes(
            encrypted_bytes
        )

        try:
            os.chmod(
                encrypted_path,
                stat.S_IRUSR | stat.S_IWUSR,
            )
        except OSError:
            pass

        # Save aggregate statistics.
        stats_path = (
            args.outdir
            / "summary_statistics.json"
        )

        stats_path.write_text(
            json.dumps(
                stats,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    except (ValueError, OSError) as error:
        print(f"ERROR: {error}")
        return

    # -------------------------
    # Improved completion report
    # -------------------------

    print("\n--- DATA CLEANING SUMMARY ---")

    print(
        f"Rows read: "
        f"{cleaning_report['original_rows']:,}"
    )

    print(
        f"Rows retained: "
        f"{len(pseudonymized):,}"
    )

    print(
        f"Exact duplicates removed: "
        f"{cleaning_report['duplicates_removed']:,}"
    )

    print(
        f"Invalid ages found: "
        f"{cleaning_report['invalid_age_values_set_missing']:,}"
    )

    print(
        f"Invalid billing amounts found: "
        f"{cleaning_report['invalid_billing_values_set_missing']:,}"
    )

    print(
        f"Invalid room numbers found: "
        f"{cleaning_report['invalid_room_numbers_set_missing']:,}"
    )

    print(
        f"Invalid discharge dates found: "
        f"{cleaning_report[
            'invalid_date_order_discharge_dates_set_missing'
        ]:,}"
    )

    print("\n--- OUTPUT FILES ---")

    print(
        f"Encrypted patient data: "
        f"{encrypted_path}"
    )

    print(
        f"Aggregate statistics: "
        f"{stats_path}"
    )

    print(
        f"Fernet key: "
        f"{key_path}"
    )

    print("\nProcessing completed successfully.")

    print(
        "\nIMPORTANT: Keep the encryption key "
        "separate from the encrypted patient data."
    )

    print(
        "Do not email, commit, or upload the key "
        "alongside the encrypted data."
    )


if __name__ == "__main__":
    main()

