from __future__ import annotations

import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay

from src.data import geocode_city, fetch_hourly_weather
from src.features import make_supervised, FEATURE_COLS

MODEL_PATH = "models/rain_next_hour_model.joblib"


def main() -> None:
    # Train on a default location so the project runs end-to-end quickly.
    # This is also clearly documented in the README.
    location = geocode_city("Stockholm", country_code="SE")
    df = fetch_hourly_weather(location.latitude, location.longitude, past_days=14)

    X, y = make_supervised(df, rain_threshold_mm=0.1)

    # If dataset is very imbalanced, stratify helps (only if both classes exist)
    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, class_weight="balanced")),
        ]
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print("Features:", FEATURE_COLS)
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=3))

    roc_auc = None
    try:
        roc_auc = float(roc_auc_score(y_test, y_prob))
        print("ROC-AUC:", round(roc_auc, 4))
    except Exception:
        print("ROC-AUC could not be computed (likely only one class present in y_test).")

    # Save ROC curve plot (nice for README)
    try:
        os.makedirs("plots", exist_ok=True)
        RocCurveDisplay.from_predictions(y_test, y_prob)
        plt.title("ROC Curve – Rain Next Hour Classifier")
        plt.grid(True)
        plt.savefig("plots/roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved ROC curve to: plots/roc_curve.png")
    except Exception as e:
        print(f"Could not generate ROC curve plot: {e}")

    # Save model bundle
    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_cols": FEATURE_COLS,
            "train_location": {
                "name": location.name,
                "latitude": location.latitude,
                "longitude": location.longitude,
            },
        },
        MODEL_PATH,
    )
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
