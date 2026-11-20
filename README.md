# ATM Cash Demand Prediction — Dehradun

## Project Information

| Field | Details |
|---|---|
| Group ID | G232 |
| Problem ID | MP293 |
| Project | ATM Cash Demand Prediction |
| Location | Dehradun |
| Course | MLOps |
| Faculty | Upasna Joshi |
| Batch | 34 |

## Team Members

| Enrollment Number | Name |
|---|---|
| S24CSEU1525 | Raghav |
| S24CSEU1542 | Arjun Thakur |
| S24CSEU1533 | Akshat Sharma |
| S24CSEU1545 | Aishani Mittal |

## Project Overview

The objective of this project is to develop a machine learning system that predicts ATM cash withdrawal demand in Dehradun.

The project follows an MLOps workflow covering data investigation, data versioning, model development, experiment tracking, API development, containerization, deployment, and model monitoring.

## Dataset Description

The project contains three datasets:

- `data/train.csv` — Historical training data.
- `data/later.csv` — Later-period data used for monitoring and retraining.
- `data/holdout_features.csv` — Holdout features used to generate final predictions.

The prediction target is `withdrawal_thousand_rs`.

## Milestone 1 — Data Investigation and Versioning

Git is used for source-code version control, and DVC is used to track the datasets.

The initial data investigation identified the following:

| Data issue | Finding |
|---|---|
| Leaking column | Not identified |
| Unit mix-up | `distance_to_market_km` in the Highway group; conversion factor 1000 |
| Sentinel value | `avg_ticket_rs` contains -999 in 54 rows |
| Duplicate rows | None identified |
| Stuck sensor | `days_to_salary` remains 22 from 2026-04-10 to 2026-04-26 |
| Target typos | No 10× target typo group identified |
| Spelling variants | `atm_type` contains variants of 3 actual categories |

Detailed findings are recorded in `findings.yaml`.

## Project Structure

```text
G232_MP293/
├── .dvc/
├── data/
│   ├── train.csv
│   ├── later.csv
│   └── holdout_features.csv
├── src/
│   └── mp293/
│       ├── __init__.py
│       └── data_quality.py
├── .gitignore
├── findings.yaml
├── requirements.txt
├── GROUP.md
├── PROBLEM.md
├── STUDENT_HANDBOOK.md
└── README.md
```

## Setup

### Prerequisites

- Python 3.11
- Git
- Internet connection for installing dependencies

### 1. Clone the repository

```bash
git clone https://github.com/raghav20051-art/G232_MP293..git
cd G232_MP293.
```

### 2. Create a Python virtual environment

A separate virtual environment should be created by each team member on their own computer.

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Install all project dependencies using the shared `requirements.txt` file:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify the installation

```bash
python -c "import pandas, yaml, dvc; print('All dependencies imported successfully')"
```

### 5. Access DVC-tracked datasets

The datasets are tracked using DVC.

To download the datasets from a configured and accessible shared DVC remote, run:

```bash
python -m dvc pull
```

**Note:** A shared DVC remote must be configured and accessible before this command can retrieve the datasets.

### 6. Run the data investigation module

Run this command from the project root:

```bash
python -m src.mp293.data_quality
```

## Tools and Technologies

- Python 3.11
- Pandas
- PyYAML
- Git and GitHub
- DVC
- YAML

Additional tools and technologies will be documented as the project progresses.

## Repository

GitHub: https://github.com/raghav20051-art/G232_MP293.
