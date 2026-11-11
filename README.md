# Cloud Workload Forecasting

Forecasts the number of active VMs and the aggregate CPU utilization for the last 2 hours (24 five-minute steps) of the Azure Public Dataset V2 subset.

## Description

Companies that use cloud providers such as Microsoft Azure and Amazon Web Services (AWS) rent computing resources to run their applications. One of these resources is a Virtual Machine (VM), which is a software-based computer that provides processing power for applications and services.

Managing these resources requires balancing computing capacity with demand. Too few available VMs can leave applications without sufficient resources to handle workload demands, potentially affecting performance and service availability. Too many VMs, on the other hand, can result in unused computing capacity and unnecessary costs. 

This project explores how predictable cloud workloads are by forecasting two metrics over a two-hour period: the number of active Virtual Machines (Active VM Count) and their average CPU utilization (Aggregate CPU Utilization). CPU utilization indicates how busy the processors are, expressed as a percentage of their capacity.

The aim is to evaluate whether these workload metrics can be accurately forecast using historical observations.

Full findings are in `reports/final_report.md`.

## Getting Started

### Dependencies

* Python 3.11 or newer
* Libraries are installed from `requirements.txt` (pandas, numpy, scikit-learn, statsmodels)

### Installing

```bash
git clone https://github.com/chynaenye/cloud-workload-forecasting
cd cloud-workload-forecasting
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS:        source .venv/bin/activate
python -m pip install -e ".[dev]"
```

### Dataset location

The data is not in the repo. Download `data-v1.tar.gz` from the [data-v1 release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1) and extract it into a `data/` folder:

```bash
mkdir -p data
tar -xzf data-v1.tar.gz -C data
```

Expected layout:

```text
data/
├── vm_cpu_readings.csv.gz
├── vm_metadata.csv.gz
├── DATA_CARD.md
└── SHA256SUMS
```

The data is not committed to this repository.

### Executing program

Train the final model:

```bash
python -m src.train --data data
```

Generate the required predictions:

```bash
python -m src.predict --data data --output predictions.csv
```

Optional: run the validation comparison and the tests:

```bash
python -m src.validate --data data
python -m pytest -q
```

Output: `predictions.csv` with columns `timestamp,active_vm_count,aggregate_cpu_utilization` (24 rows, no index).

## Project structure

```text
cloud-workload-forecasting/
│
├── data/
│   ├── DATA_CARD.md
│   ├── SHA256SUMS
│   ├── vm_cpu_readings.csv.gz
│   └── vm_metadata.csv.gz
│
├── notebooks/
│   ├── cloud-forecasting.ipynb
│   └── predictions.csv
│
├── reports/
│   └── final_report.md
│
├── src/
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── validate.py
│   └── predict.py
│
├── tests/
│   └── test_basic.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── venv/
```

## Main modeling decisions

* **Forecasting Targets:** `active_vm_count` is the number of distinct VMs with a valid reading at each timestamp. `aggregate_cpu_utilization` is the mean of `avg_cpu` across VMs at each timestamp (in %). The data had no missing values or gaps, so nothing was filled in.
* **Data Preparation:** The data had no missing values or gaps, so nothing was filled in.
* **Data Split:** built the full 134-step series first, then used the first 110 steps for training and the last 24 as the holdout. The holdout was only used once at the end.
* **Validation:** rolling-origin (expanding window) on the training part only, 24-step horizon. No shuffling.
* **Baselines:** persistence (last value) and a 6-step moving average.
* **Other models tried:** simple exponential smoothing and a Ridge model on lagged values.
* **Final model:** Ridge on lagged values. It had the lowest validation error (MAE 0.172 vs 0.280 for persistence).
* **Features:** only past values (lags 1, 2, 3, 6 and a 6-step rolling mean), so nothing from the future is used.

## Help

If you get a file not found error, check that the data files are inside the data/ folder.

## Assumptions and limitations

- Only about 11 hours of history, so daily or weekly seasonality cannot be learned.
- Aggregate CPU varies within a narrow band (about 9.1% to 9.9% in training), so absolute errors are small and differences between models are small.
- The safety margin is estimated from a few overlapping validation windows and may under-cover on the holdout.

## Acknowledgments

* Microsoft Azure Public Dataset V2 (`data-v1`)
* Atomity HQ
