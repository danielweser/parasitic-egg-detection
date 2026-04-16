# Automated Detection of Soil-Transmitted Helminth (STH) Eggs

An end-to-end computer vision pipeline for the domain-robust detection and classification of parasitic diseases in highly heterogeneous microscopy images. 

## Executive Summary
Diagnosis of STH infections relies heavily on manual microscopic examination, which is bottlenecked by a scarcity of trained microscopists in endemic regions. This project implements a fully automated **Data Engineering and MLOps pipeline** to train YOLO-based detection models on the Chula-ParasiteEgg-11 dataset. 

**Key engineering highlights:**
* **Automated ETL Pipeline:** Programmatic ingestion of raw zip data from HuggingFace, featuring multi-core image processing and dynamic bounding-box normalization.
* **Clinical Data Integrity:** Implemented custom "letterboxing" algorithms to preserve the aspect ratio of highly variable microscopy images without distorting the biological morphology of the eggs.
* **Cloud-Native Training:** Containerized execution designed for RunPod, featuring automated GPU detection, Tensor Core utilization, epoch-time tracking, email notifications, and auto-termination to optimize cloud compute costs.

## Architecture & Workflow

### 1. Data Ingestion & Preprocessing (`src/build_dataset.py`)
Microscopy datasets are notoriously unstructured. This script standardizes the data:
- **Extraction & Cleaning:** Safely extracts nested zips, strips ghost directories (`data/`), and flattens filenames.
- **Multiprocessing Letterboxing:** Uses `ProcessPoolExecutor` across all available CPU cores to pad images to a unified 800x800 resolution while strictly preserving biological aspect ratios.
- **YOLO NDJSON Export:** Dynamically translates COCO-format JSON annotations into YOLO-compatible normalized coordinates, writing to an NDJSON master file for scalable data loading.

### 2. Model Training & MLOps (`src/train.py`)
A production-ready training script utilizing Ultralytics YOLO:
- **Smart Checkpointing:** Automatically scans experiment directories to resume training from the latest `best.pt` weights.
- **Hardware Optimization:** Dynamically enables high precision matrix multiplication if Tensor Cores are available.
- **Cost Management:** Integrates with the RunPod API to automatically terminate the pod upon training completion, accompanied by automated email alerts detailing fitness and epoch metrics.

## Setup & Execution

**1. Environment Setup**
```bash
git clone https://github.com/danielweser/parasitic-egg-detection.git
cd parasitic-egg-detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Build the Dataset**
Downloads the raw data, cleans it, and builds the YOLO directory structure.
```bash
python src/build_dataset.py
```

**3. Run Training (Cloud / RunPod)**
Ensure your `RUNPOD_POD_ID` and `.runpod_key` are set in your environment if utilizing auto-termination.
```bash
python src/train.py
```

## Dataset
The primary dataset is the **Chula-ParasiteEgg-11** challenge dataset (~13,200 microscopy images, 11 species). The pipeline automatically handles downloading this from the HuggingFace Hub. *(Note: Raw and processed data are heavily `.gitignore`d to keep the repository lightweight).*

## Acknowledgements & Attribution
This pipeline was developed as part of a collaborative research effort. 
* **Data Engineering & MLOps Pipeline:** Architected by Daniel Weser (this repository).
* **Original R-CNN Baseline & Model Exploration:** Developed in collaboration with [Darren King](https://github.com/darrenaking/sth-egg-detection.git).

## License
This project is licensed under the **MIT License**.
