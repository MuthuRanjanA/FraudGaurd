from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = BASE_DIR / "data" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# FEATURE ORDER
# IMPORTANT:
# This SAME order must be used in fraud_api.py
# ============================================================

FEATURES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 60)
    print("FraudGuard - Fraud Detection Model Training")
    print("=" * 60)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_FILE}\n\n"
            "Please put the real creditcard.csv file inside:\n"
            f"{BASE_DIR / 'data'}\n"
        )

    print(f"\nLoading dataset:")
    print(DATA_FILE)

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    df = pd.read_csv(DATA_FILE)

    print(f"\nDataset loaded successfully.")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = FEATURES + ["Class"]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "\nMissing required columns:\n"
            f"{missing}\n\n"
            "Your dataset must contain:\n"
            f"{required_columns}\n\n"
            "The real credit card dataset should have "
            "a 'Class' column where:\n"
            "0 = legitimate transaction\n"
            "1 = fraud transaction"
        )

    # --------------------------------------------------------
    # Clean numeric columns
    #
    # Removes characters such as:
    # *2.1234
    # **-3.4567
    #
    # This is useful if your CSV contains those markers.
    # --------------------------------------------------------

    print("\nCleaning dataset...")

    for column in FEATURES:

        df[column] = (
            df[column]
            .astype(str)
            .str.replace("*", "", regex=False)
            .str.strip()
        )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove rows with invalid values

    before = len(df)

    df = df.dropna(
        subset=FEATURES + ["Class"]
    )

    after = len(df)

    if before != after:

        print(
            f"Removed {before - after:,} invalid rows."
        )

    # --------------------------------------------------------
    # Prepare X and y
    # --------------------------------------------------------

    X = df[FEATURES]

    y = pd.to_numeric(
        df["Class"],
        errors="coerce"
    )

    # Remove invalid Class values

    valid_class = y.isin([0, 1])

    X = X.loc[valid_class]
    y = y.loc[valid_class]

    if len(X) == 0:

        raise ValueError(
            "No valid training rows found."
        )

    print("\nClass distribution BEFORE SMOTE:")

    print(y.value_counts())

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    print("\nScaling features...")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------------
    # SMOTE
    # --------------------------------------------------------

    print("Balancing dataset using SMOTE...")

    smote = SMOTE(
        random_state=42
    )

    X_balanced, y_balanced = smote.fit_resample(
        X_scaled,
        y
    )

    print("\nClass distribution AFTER SMOTE:")

    print(
        pd.Series(y_balanced).value_counts()
    )

    # --------------------------------------------------------
    # Train Logistic Regression
    # --------------------------------------------------------

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    model.fit(
        X_balanced,
        y_balanced
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_file = MODEL_DIR / "fraud_model.pkl"
    scaler_file = MODEL_DIR / "scaler.pkl"

    joblib.dump(
        model,
        model_file
    )

    joblib.dump(
        scaler,
        scaler_file
    )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nOriginal rows : {len(X):,}")
    print(f"Balanced rows : {len(X_balanced):,}")

    print("\nModel saved:")
    print(model_file)

    print("\nScaler saved:")
    print(scaler_file)

    print("\nFeature order:")

    for index, feature in enumerate(FEATURES, start=1):

        print(
            f"{index:2}. {feature}"
        )

    print("\nYou can now start the Flask application.")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()