# Two-window Cell Painting analysis (48 h vs 8 days)

This repository contains Python code to reproduce the core analysis described in the manuscript:
"A stability-weighted combined map for Cell Painting across regular and prolonged incubation".

## What this code does
1. Loads a single CSV file that contains both exposure windows.
2. Keeps only compounds with at least *N* independent long-incubation iterations (optional filter).
3. Splits metadata vs numeric features.
4. Standardises features and runs PCA.
5. Builds a kNN graph in PCA space and performs Leiden clustering (Scanpy).
6. Computes within-window replicate stability (mean within-compound replicate distance in PCA space).
7. Builds a **Combined dataset** by selecting, for each compound, the exposure window with higher stability.
8. Exports cluster tables and (optionally) UMAP plots for Short / Long / Combined.

Outputs are saved to the folder specified in `out_dir`.

## Input data expectations
The input CSV should contain:
- metadata columns starting with `Metadata_` (e.g., `Metadata_Name`, `Metadata_Source`, `Metadata_MoA`)
- numeric feature columns (CellProfiler features)

By default, the script uses `Metadata_Source` to split Short vs Long using string patterns:
- short: contains `sh`
- long: contains `l`

If your naming differs, update the patterns in the config.

## Notes
- UMAP is used for 2D visualisation.
- The code sets a random seed for reproducibility.

## Repository contents
- `A stability-weighted combined map for Cell Painting.ipynb`
- `env/siriusrequirements.txt`: full pip freeze shared by the supervisor (may include non-Windows packages).
- `env/requirements_windows.txt`: Windows-friendly requirements (same as above but with non-Windows lines removed).
- `environment.yml`: conda environment file that installs Python and then installs `env/requirements_windows.txt`.

