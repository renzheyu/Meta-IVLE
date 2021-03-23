This repository presents a complete data pipeline for college success prediction. [[Paper](https://educationaldatamining.org/files/conferences/EDM2020/papers/paper_194.pdf)][[Video](https://www.youtube.com/watch?v=CZgACA4BJiQ&t=9s)]

## Data
The raw data should be organized by courses, each having a sub-folder under `DATA_DIR` (specified in `config/.config`).

## Usage
High-level steps of using this pipeline:
1. Put the raw data in place;
1. Rename `config_sample` folder into `config`;
1. Modify the content of config files according to your needs;
1. Execute the pipeline in `run-pipeline.ipynb`.
