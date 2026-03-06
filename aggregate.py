#!/usr/bin/env python3
"""Aggregate per-model CSVs from a folder.

Expected input: many CSVs inside a directory (default: results_experiments/).
Each CSV typically has one data row like:
 ,Format,Status❔,Size (MB),metrics/mAP50-95(B),Inference time (ms/im),FPS
1,TensorRT,✅,78.8,0.0,58.25,17.17

Outputs:
- combined_results.csv (one row per model)

Usage:
  python aggregate_and_plot_results.py --input results_experiments --out combined_results.csv
"""

from __future__ import annotations

import argparse
import os
import re
import glob
import csv

import pandas as pd
import matplotlib.pyplot as plt


def sanitize_model_name(filename: str) -> str:
    """Derive a model name from a CSV filename.
    Tries to remove common prefixes like 'results_experiments' and file extensions.
    """
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]

    # Remove leading 'results_experiments' and separators if present
    name = re.sub(r"^results_experiments_train[_\-\s]*", "", name, flags=re.IGNORECASE).strip()
    name = re.sub(r"^results_compressed[_\-\s]*", "", name, flags=re.IGNORECASE).strip()
    # remove everything before the last "_yolo" found
    name = re.sub(r".*_yolo", "yolo", name, flags=re.IGNORECASE).strip()

    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name if name else os.path.splitext(base)[0]


def find_col(df: pd.DataFrame, patterns: list[str]) -> str | None:
    """
    Return the first column whose name matches any of the given regex patterns (case-insensitive).
    """
    cols = list(df.columns)
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        for c in cols:
            if rx.search(str(c)):
                return c
    return None


def coerce_numeric(series: pd.Series) -> pd.Series:
    """
    Make a numeric series from messy strings (commas, spaces, etc.).
    """
    s = series.astype(str).str.replace(",", ".", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def find_map(train_name: str) -> float | None:
    """Read experiments/train/<train_name>/results.csv and return the final mAP value.

    Returns None if the file doesn't exist or can't be parsed.
    NOTE: last_line[7] assumes your results.csv has mAP in column index 7.
    Adjust if your CSV schema differs.
    """
    ruta = os.path.join("experiments", "train", train_name, "results.csv")
    try:
        with open(ruta, "r", newline="") as resfile:
            reader = csv.reader(resfile)
            all_lines = list(reader)
            if not all_lines:
                return None

            last_line = all_lines[-1]
            if len(last_line) <= 7:
                return None

            return float(str(last_line[7]).replace(",", ".").strip())
    except Exception:
        print(f"[WARN] No pude leer mAP de: {ruta}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=".", help="Folder containing per-model CSV files")
    ap.add_argument("--out", default="combined_results.csv", help="Output aggregated CSV path")
    args = ap.parse_args()

    input_dir = args.input
    out_csv = args.out

    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")

    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not csv_files:
        raise SystemExit(f"No CSV files found in: {input_dir}")

    rows: list[dict] = []
    for f in csv_files:
        # 1) Saltar archivos vacíos (0 bytes)
        if os.path.getsize(f) == 0:
            print(f"[SKIP] CSV vacío: {f}")
            continue

        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            print(f"[SKIP] Sin columnas / vacío: {f}")
            continue
        except Exception:
            # Fallback con separador ;
            try:
                df = pd.read_csv(f, sep=";")
            except pd.errors.EmptyDataError:
                print(f"[SKIP] Sin columnas / vacío (sep=';'): {f}")
                continue

        if df.empty:
            print(f"[SKIP] DataFrame vacío tras lectura: {f}")
            continue

        # Many of these files include an unnamed first column (index-like). Drop it.
        unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed")]
        if unnamed:
            df = df.drop(columns=unnamed)

        # Keep first row only (typical case)
        r = df.iloc[0].to_dict()

        model = sanitize_model_name(f)

        # # "train_x" or "yolo26_train_x" -> "x"
        # train_name = model.split("train_", 1)[1] if "train_" in model else model
        # map_from_train = find_map(train_name)

        col_format = find_col(df, [r"^format$", r"format"])
        col_size = find_col(df, [r"size", r"\(mb\)"])
        col_map = find_col(df, [r"mAP", r"map50\-?95"])
        col_inf = find_col(df, [r"inference.*ms", r"inference time", r"ms/im"])
        col_fps = find_col(df, [r"^fps$", r"\bfps\b"])

        out_row = {
            "model": model,
            "Format": r.get(col_format) if col_format else None,
            "Size_MB": r.get(col_size) if col_size else None,
            # Prefer mAP from training results.csv; fallback to the aggregated CSV column if present
            "mAP50_95": r.get(col_map) if col_map else None,
            "Inference_ms_im": r.get(col_inf) if col_inf else None,
            "FPS": r.get(col_fps) if col_fps else None,
        }
        rows.append(out_row)

    combined = pd.DataFrame(rows)

    # Coerce numeric columns
    for c in ["Size_MB", "mAP50_95", "Inference_ms_im", "FPS"]:
        if c in combined.columns:
            combined[c] = coerce_numeric(combined[c])

    combined = combined.sort_values(by=["FPS"], ascending=False, na_position="last")
    combined.to_csv(out_csv, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())