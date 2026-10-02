# 🌍 GreenTravel Intelligence Challenge — High-Carbon Trip Prediction

> *"The greenest trip is the one you optimise before it happens."*

A machine-learning pipeline that predicts, **before a trip is booked**, whether a corporate business trip will be classified as **high-carbon**. Built on ~87,000 synthetic business-trip records (public training set + private test set) using process-mining event logs, trip metadata, and exception attributes.

| 🐍 Language | 🤖 Model | 📈 OOF ROC-AUC | 🎯 OOF F1 | 🔍 Precision / Recall |
|:-----------:|:--------:|:--------------:|:---------:|:---------------------:|
| Python 3.12 | XGBoost | **0.9994** | **0.986** | 0.990 / 0.983 |

---

## 📌 Table of Contents

1. [Project Background](#-project-background)
2. [Problem Statement](#-problem-statement)
3. [Dataset](#-dataset)
4. [Solution Approach](#-solution-approach)
5. [Feature Engineering](#-feature-engineering)
6. [Model & Training](#-model--training)
7. [Results](#-results)
8. [Key Insights](#-key-insights)
9. [Tech Stack](#-tech-stack--frameworks)
10. [Repository Structure](#-repository-structure)
11. [How to Run](#-how-to-run)
12. [Limitations & Future Work](#-limitations--future-work)

---

## 🌱 Project Background

Business travel is one of the largest and hardest-to-decarbonise slices of a company's **Scope 3** emissions. This project is part of a sustainability capstone ("Decarbonizing Travel", *Road to Net Zero by 2030*) that has two tracks:

| Track | Focus | Tooling |
|-------|-------|---------|
| **Part A – Process Analysis** | Process discovery, policy-exception analysis, emissions dashboards and an executive pitch | Celonis (Process & Variant Explorer) |
| **Part B – Predictive ML** *(this repo)* | Predict high-carbon trips before booking so the company can intervene early (alternative mode, routing, or virtual meeting) | Python, XGBoost |

> ⚠️ All data is fictional and provided for educational purposes only.

---

## 🎯 Problem Statement

**Binary classification:** given only information available *before* a trip is undertaken, predict the probability that the trip belongs to the `HighCarbon` class.

- **Target:** `HighCarbon` (1 = high-carbon, 0 = not)
- **Output:** a probability score per trip for the private test set (`submission.csv`)
- **Evaluation (hidden ground truth):** ROC-AUC, F1, Precision, Recall

### 🚫 Banned features (label leakage)

The challenge rules prohibit using any variable that reveals or is derived from the target. These are dropped immediately after loading and never touched:

`Departure_CO2e` · `Return_CO2e` · `Hotel_CO2e` · `Spend_CO2e` · `TotalCO2e` · `HighCarbon`

---

## 📊 Dataset

Six interconnected CSV files, all joined on **`TripID`**:

| File | Rows | Description |
|------|-----:|-------------|
| `public_trip_data.csv` | 65,289 | Labelled training trips: locations, transport mode, purpose, policy flag, business unit, hotel nights, net cost, CO₂e columns, `HighCarbon` label |
| `public_trip_event_attributes.csv` | 65,289 | Exception context: denial/reimbursement reasons, cancellations, new transport/hotel selections, delays, extensions, days pre-approved, process code |
| `public_trip_event_log.csv` | — | Timestamped process steps per trip (Booking → Approval → Departure → Reimbursement) |
| `private_trip_data.csv` | 21,764 | Unlabelled test trips (no CO₂e / label columns) |
| `private_trip_event_attributes.csv` | 21,764 | Attributes for the test trips |
| `private_trip_event_log.csv` | 223,819 | Event log for the test trips |

Supporting files: `sample_submission.csv` (format reference), `Sustainability_Data_Dictionary.xlsx`.

### Class balance

| HighCarbon | Share |
|-----------:|------:|
| 0 | 74.99% |
| 1 | 25.01% |

---

## 🧭 Solution Approach

```mermaid
flowchart LR
    A[6 CSV files] --> B[Drop banned CO₂e/label columns]
    B --> C[EDA]
    C --> D[Event-log pivot + timing features]
    D --> E[Trip + attribute feature engineering]
    E --> F[5-fold Stratified CV · XGBoost]
    F --> G[OOF evaluation + feature importance]
    G --> H[Retrain on 100% public data]
    H --> I[Predict private set → submission.csv]
```

### 1. Exploratory Data Analysis — the key discovery

`HighCarbon` turns out to be a **hard threshold on `TotalCO2e` (≈ 3,471 kg)**. So the real task is to learn which *pre-booking* features (route, cabin class, mode) push a trip across that line.

**High-carbon rate by transport mode**

| Mode | High-carbon rate | Avg TotalCO₂e (kg) |
|------|-----------------:|-------------------:|
| First Class Flight | 62.9% | 6,702 |
| Business Class Flight | 56.4% | 4,865 |
| Economy Flight | 21.4% | 2,555 |
| BMW 3 diesel | 10.7% | 1,664 |
| BMW 3 plug-in hybrid | 10.4% | 1,675 |
| VW Golf petrol | 5.3% | 1,335 |
| VW Golf diesel | 5.2% | 1,322 |
| Fiat 500 electric | 5.1% | 1,066 |
| Train | 0.0% | 445 |

**Policy compliance:** out-of-policy trips are high-carbon **38.5%** of the time vs. **22.0%** for in-policy trips.

### 2. Event-log feature engineering

The event log is pivoted to one row per trip and converted into process-timing features (details below). The same function is applied to both public and private logs, guaranteeing identical features on train and test.

### 3. Trip + attribute feature engineering

Trip data, event attributes and event-log features are merged (`validate='1:1'`) and expanded into **54 model features** (36 numeric + 18 categorical).

### 4. Modelling

Cross-validated **XGBoost** classifier with native categorical support, early stopping on AUC, and a final refit on all labelled data.

---

## 🛠️ Feature Engineering

### Event-log (process) features — 13 total

| Feature | Definition |
|---------|-----------|
| `booking_lead_days` | Departure − *Book Mode of Transportation* (days) |
| `approval_cycle_hours` | *Travel Request Approved* − *Submit Travel Request* |
| `expense_cycle_hours` | *Expense Reimbursement* − *Submit Expense Request* |
| `confirm_to_departure_days` | Departure − *Receive Confirmation* |
| `booking_to_submit_hours` | *Submit Travel Request* − booking time |
| `n_process_steps` | Total events logged for the trip |
| `n_exception_events` | Events *outside* the defined "happy path" |
| `had_manager_preapproved` | Flag: *Manager Preapproved* event present |
| `*_missing` (×5) | Missing-value indicators for each timing feature (missing values are median-imputed) |

> ✅ Sanity check: median booking lead time is **~21 days for in-policy** vs. **~9 days for out-of-policy** trips, consistent with a 14-day booking rule.

### Trip & attribute features

| Group | Features |
|-------|----------|
| **Transport mode** | `is_flight`, `is_train`, `is_car`, `flight_class_rank` (Economy=1, Business=2, First=3), `is_ev_or_hybrid` |
| **Geography** | `route` (departure city → arrival city), `country_pair`, `is_domestic` |
| **Policy & cost** | `out_of_policy_flag`, `cost_per_hotel_night`, `NetCosts`, `HotelNights`, `EntitiyCode` |
| **Exception flags** | `had_transport_cancel`, `had_new_transport`, `had_hotel_change`, `had_delay`, `had_extension`, `had_price_diff`, `had_expense_denial`, `had_expense_reimb_amount` |
| **Exception magnitudes** | `extension_length`, `transport_price_diff`, `expense_reimb_amount`, `days_preapproved` |
| **Categorical (18)** | Shipping type, purpose, business unit, departure/arrival country, route, country pair, process code, and the reason/selection attribute columns (nulls filled with `"None"`) |

Private-set categoricals are re-aligned to the public-set category levels so train and test share an identical encoding.

---

## 🤖 Model & Training

**Algorithm:** `XGBClassifier` (histogram tree method, native categorical support)

| Hyperparameter | Value |
|----------------|-------|
| `n_estimators` | 600 (with early stopping) |
| `max_depth` | 5 |
| `learning_rate` | 0.05 |
| `subsample` | 0.85 |
| `colsample_bytree` | 0.8 |
| `reg_lambda` | 1.5 |
| `min_child_weight` | 5 |
| `eval_metric` | AUC |
| `early_stopping_rounds` | 40 |
| `random_state` | 42 |

**Validation:** 5-fold `StratifiedKFold` (shuffled, seed 42) → out-of-fold (OOF) predictions for unbiased evaluation.

**Final model:** retrained on 100% of the public data with `n_estimators ≈ 1.1 × mean(best_iteration)` from CV (≈ 84 trees), then used to score the private set.

---

## 📈 Results

### Cross-validation (5-fold, out-of-fold)

| Fold | ROC-AUC | Best iteration |
|-----:|--------:|---------------:|
| 0 | 0.99931 | 98 |
| 1 | 0.99923 | 69 |
| 2 | 0.99953 | 105 |
| 3 | 0.99941 | 53 |
| 4 | 0.99934 | 53 |
| **Overall OOF** | **0.99935** | — |
| Fold mean ± std | 0.99936 ± 0.00010 | — |

### Classification metrics (OOF, threshold = 0.5)

| Metric | Score |
|--------|------:|
| **ROC-AUC** | **0.99935** |
| **Precision** | **0.9897** |
| **Recall** | **0.9830** |
| **F1** | **0.9864** |

### Confusion matrix (OOF, n = 65,289)

|  | Predicted 0 | Predicted 1 |
|--|------------:|------------:|
| **Actual 0** | 48,796 | 167 |
| **Actual 1** | 277 | 16,049 |

Only **444 of 65,289** trips (~0.7%) are misclassified.

### Top features by average gain

| Rank | Feature | Gain |
|-----:|---------|-----:|
| 1 | `out_of_policy_flag` | 1237.6 |
| 2 | `country_pair` | 784.0 |
| 3 | `route` | 714.4 |
| 4 | `is_domestic` | 527.5 |
| 5 | `flight_class_rank` | 488.2 |
| 6 | `ArrivalLocationCountry` | 359.4 |
| 7 | `is_train` | 357.3 |
| 8 | `ShippingTypeDescription` | 334.8 |
| 9 | `HotelNights` | 96.2 |
| 10 | `is_car` | 82.9 |

---

## 💡 Key Insights

- **Route + cabin class drive the label.** Distance (captured via `route` / `country_pair` / `is_domestic`) and cabin class are the physical determinants of total emissions, which is why the model is so accurate. These are legitimate pre-booking features, **not** label leakage.
- **First & Business class flights are the biggest hotspots** — ~63% and ~56% of those trips are high-carbon, versus 0% for trains.
- **Out-of-policy trips skew high-carbon** (38.5% vs. 22.0%), making the policy flag the single most informative feature.
- **Honest finding on event-log features:** adding process-timing features moved AUC only from ~0.9993 → ~0.9994, within fold-to-fold noise. Timing tracks *policy compliance* strongly, but `OutOfPolicy` was already available. The features are kept because they use all legitimate data and keep both challenge tracks on the same population.

### 💼 Business takeaways

- Intervene at booking time for **long-haul, premium-cabin** itineraries — suggest Economy, rail alternatives, or virtual meetings.
- Tighten approval workflows for **short-lead-time** bookings, which correlate with out-of-policy behaviour.
- Prioritise **train over short-haul flights** and EVs/hybrids over petrol/diesel cars as quick wins.

---

## 🧰 Tech Stack & Frameworks

| Category | Tools |
|----------|-------|
| Language | Python 3.12 |
| Data wrangling | `pandas`, `numpy` |
| Modelling | `xgboost` (`XGBClassifier`, native categoricals, `hist` tree method) |
| Validation & metrics | `scikit-learn` (`StratifiedKFold`, `roc_auc_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `RocCurveDisplay`) |
| Visualisation | `matplotlib` |
| Environment | Jupyter Notebook |
| Process-analysis track | Celonis (Part A of the capstone) |

---

## 📁 Repository Structure

```
.
├── HighCarbon_Prediction_Script.ipynb   # End-to-end EDA, features, model, submission
├── submission.csv                       # Output: TripID, HighCarbon (probability)
├── sample_submission.csv                # Reference submission format
├── Methodology_HighCarbon_Prediction.md # Full methodology write-up
├── Sustainability_Data_Dictionary.xlsx  # Column definitions
├── docs/
│   ├── Part2.pdf                        # GreenTravel Intelligence Challenge handbook
│   ├── Sustainability_Capstone_Outline__Decarbonizing_Travel.pdf
│   └── README__Sustainability_Capstone__Decarbonizing_Travel.pdf
└── project/                             # Data folder (referenced as DATA_DIR in the notebook)
    ├── public_trip_data.csv
    ├── public_trip_event_attributes.csv
    ├── public_trip_event_log.csv
    ├── private_trip_data.csv
    ├── private_trip_event_attributes.csv
    └── private_trip_event_log.csv
```

> 💡 Consider adding the data files to `.gitignore` (or using Git LFS) — the event logs are large.

---

## ▶️ How to Run

**1. Clone the repo**

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

**2. Install dependencies**

```bash
pip install pandas numpy matplotlib scikit-learn xgboost jupyter
```

**3. Add the data**

Place all six CSV files in a folder named `project/` (or edit `DATA_DIR` at the top of the notebook).

**4. Run the notebook**

```bash
jupyter notebook HighCarbon_Prediction_Script.ipynb
```

Run all cells top to bottom. The notebook will print CV scores, plot ROC and feature-importance charts, and write `submission.csv` to the working directory.

### Submission format

```csv
TripID,HighCarbon
2,0.989360
4,0.017547
5,0.831603
...
```

`HighCarbon` is the **predicted probability** (0–1) of a trip being high-carbon, for all 21,764 private-set trips.

---

## ⚠️ Limitations & Future Work

- **Near-ceiling performance.** Because the label is a deterministic threshold on emissions driven by distance and cabin class, AUC is close to 1.0; the real-world value lies in the *interpretation*, not the headline score.
- **Synthetic data.** Findings may not transfer to real travel data with noisier emission factors.
- **Threshold tuning.** A 0.5 cutoff is used for precision/recall/F1; a business-driven threshold (e.g. favouring recall for intervention) could be tuned.
- **Possible extensions:** explicit great-circle distance features, LightGBM/CatBoost ensembles, SHAP explanations, probability calibration, and a booking-time recommender that suggests lower-carbon alternatives.

---

## 🙌 Acknowledgements

Based on the *GreenTravel Intelligence Challenge* and the *Sustainability Capstone: Decarbonizing Travel* (Summer Analytics 2026). Data is fictional and for educational use only.

---

## 📄 License

Add your preferred license here (e.g. MIT).