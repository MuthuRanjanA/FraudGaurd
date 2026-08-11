# FraudGuard — Easy Credit Card Fraud Detection

FraudGuard is a Flask + machine-learning web application for detecting possible credit-card fraud.

## Important model input

The original credit-card dataset normally contains:

`Time, V1, V2, ..., V28, Amount, Class`

The model therefore needs **30 input features**:

- Transaction time
- V1 to V28 anonymized PCA features
- Transaction amount

`Class` is the target: `0 = genuine`, `1 = fraud`.

Because V1–V28 are anonymized features from the dataset, they cannot honestly be replaced with made-up human fields. The new interface therefore explains this clearly and provides:
1. A beginner-friendly dashboard.
2. A guided 30-feature prediction form.
3. A paste-a-row option for testing a dataset transaction.
4. A REST API.
5. A training script using StandardScaler + SMOTE + Logistic Regression.

## Setup on Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Put the Kaggle `creditcard.csv` at:

```text
data/creditcard.csv
```

Train:

```powershell
python src/train_model.py
```

Run:

```powershell
python src/fraud_api.py
```

Open:

`http://127.0.0.1:5000`

## API

`POST /api/predict`

```json
{
  "features": [0, 0.1, -0.2, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100]
}
```

The API requires exactly 30 numeric features.

## Project structure

```text
FraudGuard/
├── data/
│   └── creditcard.csv
├── model/
│   ├── fraud_model.pkl
│   └── scaler.pkl
├── src/
│   ├── fraud_api.py
│   └── train_model.py
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── predict.html
│   ├── result.html
│   └── about.html
├── static/
│   └── css/style.css
├── requirements.txt
└── README.md
```

## Security note

This is an educational project. Do not use its prediction as the only decision for a real financial transaction.
