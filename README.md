# Meta-IVLE
This repository presents a complete data pipeline for college success prediction.

## Data
The raw data should be organized by courses, each having a sub-folder under `DATA_DIR` (specified in `config/.config`).

## Usage
High-level steps of using this pipeline:
1. Put the raw data in place;
1. Rename `config_sample` folder into `config`;
1. Modify the content of config files according to your needs;
1. Execute the pipeline in `run-pipeline.ipynb`.