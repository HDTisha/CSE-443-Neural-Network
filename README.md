# CSE-443 Neural Network Assignment 

This repository contains two end-to-end machine learning solutions built for CSE-443,
each in its own directory with its own code, generated outputs, and a detailed
`README.md` covering approach, methodology, and findings.

| | Problem Set 01 | Problem Set 02 |
|---|---|---|
| **Task** | Image classification (CNN) | Binary classification (Logistic Regression) |
| **Domain** | Healthcare — paediatric chest X-rays | Banking — marketing campaign response |
| **Directory** | [`Pneumonia Detection/`](./Pneumonia%20Detection) | [`Bank Marketing/`](./Bank%20Marketing) |
| **Goal** | Classify X-rays as NORMAL or PNEUMONIA | Predict term deposit subscription (yes/no) |
| **Dataset size** | 5,863 JPEG X-ray images | 45,211 customer records, 17 columns |
| **Key metric** | ROC-AUC: **0.83**, Recall (Pneumonia): **0.997** | ROC-AUC: **0.77**, Recall (Yes): **0.62** |

---

## Repository Structure

```
CSE-443 Neural Network/
├── Pneumonia Detection/          # Problem Set 01
│   ├── x-ray.ipynb               # EDA + CNN build/train/evaluate
│   ├── output/                   # Generated plots + trained model ( keras)
│   └── README.md                 # Full write-up: methodology, results, findings
│
├── Bank Marketing/               # Problem Set 02
│   ├── bank.py                   # EDA + preprocessing + Logistic Regression + evaluation
│   ├── bank-full.csv             # Source dataset
│   ├── images/                   # Generated EDA + evaluation plots
│ ├── eda_summary.txt             # EDA summary
│ ├── model_report.txt            # Model evaluation results
│   └── README.md                 # Full write-up: methodology, results, findings
│
└── README.md                   
```

---

## 1. Pneumonia Detection — CNN (Problem Set 01)

A convolutional neural network trained from scratch (4 conv blocks with batch
normalization + max pooling, global average pooling, dense head) to classify paediatric
chest X-rays as **NORMAL** or **PNEUMONIA**, with data augmentation and class weighting
to handle the imbalanced dataset.

- **Result:** 66.5% test accuracy, 0.83 ROC-AUC, 99.7% recall on PNEUMONIA but only
  11% recall on NORMAL — the model is biased toward predicting PNEUMONIA, with
  documented causes (tiny 16-image validation split, only 5 training epochs) and
  concrete next steps in the project README.

## 2. Bank Marketing — Logistic Regression (Problem Set 02)

A logistic regression model predicting whether a bank customer will subscribe to a term
deposit, using the UCI Bank Marketing dataset. `duration` is deliberately excluded to
avoid data leakage (it's only known after a call ends), and `class_weight="balanced"`
compensates for the ~88/12 class imbalance.

- **Result:** 76% test accuracy, 0.77 ROC-AUC, 62% recall on the "yes" class. Strongest
  predictors: prior campaign success, contact month, education, and job type. Full
  feature-importance breakdown and business recommendations in the project README.

---

## Setup

Each project has its own dependencies and run instructions in its own README. In brief:

```bash
# Pneumonia Detection
pip install tensorflow numpy matplotlib scikit-learn
export CHEST_XRAY_DATA_DIR=/path/to/chest_xray   # folder with train/ val/ test/
jupyter notebook "Pneumonia Detection/x-ray.ipynb"

# Bank Marketing
pip install pandas numpy matplotlib seaborn scikit-learn
# place bank-full.csv in "Bank Marketing/data/" (gitignored, not included in repo)
python "Bank Marketing/bank.py"
```

Datasets are not committed to this repository (see `.gitignore`) — download them
separately and place them in each project's expected data path as described above.

