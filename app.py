import joblib
import pandas as pd
import streamlit as st
from ai_escalation_agent import generate_ai_mitigation_plan

# Page Configuration
st.set_page_config(
    page_title="AI Supply Chain SLA Risk Predictor",
    page_icon="🚚",
    layout="wide",
)

# Load Trained ML Pipeline
@st.cache_resource
def load_model():
    return joblib.load("sla_risk_model.pkl")

model = load_model()

# Header & Context
st.title("🚚 AI Supply Chain SLA Risk & Escalation Engine")
st.markdown(
    "Predict shipment delay probabilities using **Random Forest ML** and trigger automated "
    "**Generative AI mitigation plans** before shipments breach their 90% contractual SLA."
)
st.divider()

# Sidebar: Live Shipment Telemetry Controls
st.sidebar.header("📦 Configure Shipment Telemetry")

shipment_id = st.sidebar.text_input("Shipment ID", value="SHP-109482")
carrier = st.sidebar.selectbox(
    "Assigned 3PL Carrier",
    ["CN Rail", "Canada Post", "Challenger Motor Freight", "FedEx Express"],
)
origin_hub = st.sidebar.selectbox(
    "Origin Warehouse Hub",
    [
        "Toronto_Hub",
        "Mississauga_Hub",
        "Montreal_Hub",
        "Vancouver_Hub",
        "Chicago_CrossBorder_Hub",
    ],
)
shipping_mode = st.sidebar.selectbox(
    "Shipping Mode", ["Intermodal Rail", "Ground", "LTL Freight", "Express"]
)
warehouse_dwell_hrs = st.sidebar.slider(
    "Origin Warehouse Dwell Time (Hours)", min_value=1.0, max_value=48.0, value=22.5, step=0.5
)
distance_km = st.sidebar.slider(
    "Transit Distance (km)", min_value=150, max_value=3500, value=2450, step=50
)
package_weight_kg = st.sidebar.number_input(
    "Package Weight (kg)", min_value=1.0, max_value=500.0, value=42.5
)
weather_severity = st.sidebar.selectbox(
    "Weather Forecast", ["Clear", "Rain/Snow", "Severe Winter Storm"]
)
is_peak_season = st.sidebar.checkbox("Peak Shipping Season", value=True)
risk_threshold = st.sidebar.slider(
    "AI Escalation Alert Threshold (%)", min_value=50, max_value=90, value=65
)

# Assemble Input DataFrame
shipment_data = {
    "shipment_id": shipment_id,
    "carrier": carrier,
    "origin_hub": origin_hub,
    "shipping_mode": shipping_mode,
    "distance_km": distance_km,
    "package_weight_kg": package_weight_kg,
    "warehouse_dwell_hrs": warehouse_dwell_hrs,
    "weather_severity": weather_severity,
    "is_peak_season": 1 if is_peak_season else 0,
}

input_df = pd.DataFrame([shipment_data]).drop(columns=["shipment_id"])
delay_prob = model.predict_proba(input_df)[0][1]

# Main Dashboard Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Predicted SLA Delay Risk",
        value=f"{delay_prob:.1%}",
        delta=f"{(delay_prob - (risk_threshold / 100)):.1%} vs Threshold",
        delta_color="inverse",
    )

with col2:
    st.metric(
        label="Origin Warehouse Dwell",
        value=f"{warehouse_dwell_hrs} hrs",
        delta="Bottleneck Alert (>12h)" if warehouse_dwell_hrs > 12 else "Normal Flow",
        delta_color="inverse" if warehouse_dwell_hrs > 12 else "normal",
    )

with col3:
    status_text = "🚨 HIGH RISK" if delay_prob >= (risk_threshold / 100) else "✅ ON-TIME TRACK"
    st.metric(label="Automated Dispatch Status", value=status_text)

st.divider()

# Trigger AI Agent Output
if delay_prob >= (risk_threshold / 100):
    st.error(
        f"**ALERT:** Shipment **{shipment_id}** exceeds the {risk_threshold}% risk threshold. "
        "Generative AI Operations Agent triggered below:"
    )
    ai_report = generate_ai_mitigation_plan(shipment_data, delay_prob)
    st.code(ai_report, language="text")
else:
    st.success(
        f"**CLEARED:** Shipment **{shipment_id}** ({carrier}) is operating within acceptable "
        "risk parameters for standard dispatch."
    )
