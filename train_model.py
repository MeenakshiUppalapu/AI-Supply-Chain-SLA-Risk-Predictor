import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 1. Load your unified dataset
df = pd.read_csv("supply_chain_shipments.csv")

# Separate features (X) and target (y)
X = df.drop(columns=["shipment_id", "sla_delayed"])
y = df["sla_delayed"]

# 2. Define categorical and numerical columns
categorical_cols = [
    "carrier",
    "origin_hub",
    "shipping_mode",
    "weather_severity",
]
numerical_cols = [
    "distance_km",
    "package_weight_kg",
    "warehouse_dwell_hrs",
    "is_peak_season",
]

# 3. Build preprocessing & Random Forest pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols),
    ]
)

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=150, max_depth=8, random_state=42
            ),
        ),
    ]
)

# 4. Split data (80% training, 20% testing) and train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model_pipeline.fit(X_train, y_train)

# 5. Evaluate Model Performance
y_pred = model_pipeline.predict(X_test)
y_prob = model_pipeline.predict_proba(X_test)[:, 1]

print("=== MODEL PERFORMANCE REPORT ===")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.3f}")

# 6. Extract Top Operational Drivers (Feature Importance)
feature_names = model_pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()
importances = model_pipeline.named_steps["classifier"].feature_importances_

feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances,
}).sort_values(by="Importance", ascending=False)

print("\n=== TOP 5 DRIVERS OF SLA DELAYS ===")
print(feature_df.head(5).to_string(index=False))

# 7. Save the trained pipeline for Phase 3 (GenAI Agent & App)
joblib.dump(model_pipeline, "sla_risk_model.pkl")
print("\nModel saved successfully as 'sla_risk_model.pkl'!")
