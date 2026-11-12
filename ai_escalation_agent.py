import os
import joblib
import pandas as pd

# Load your trained Phase 2 ML Pipeline
model = joblib.load("sla_risk_model.pkl")


def generate_ai_mitigation_plan(shipment_data: dict, delay_prob: float) -> str:
    """
    Uses an LLM (OpenAI/Gemini) if an API key is set, or falls back to a
    structured diagnostic template so the pipeline always runs cleanly.
    """
    prompt = f"""
    You are a Senior Supply Chain AI Operations Agent. A shipment has been flagged with a {delay_prob:.1%} probability of missing its 90% SLA target.

    Shipment Telemetry:
    - Shipment ID: {shipment_data['shipment_id']}
    - Assigned Carrier: {shipment_data['carrier']}
    - Origin Hub: {shipment_data['origin_hub']}
    - Mode: {shipment_data['shipping_mode']} | Distance: {shipment_data['distance_km']} km
    - Warehouse Dwell Time: {shipment_data['warehouse_dwell_hrs']} hours
    - Weather Condition: {shipment_data['weather_severity']}
    - Peak Season: {'Yes' if shipment_data['is_peak_season'] == 1 else 'No'}

    Provide:
    1. Root Cause Diagnosis (2 sentences max, referencing whether the issue is internal dock dwell time vs external transit).
    2. Recommended Operational Action (e.g., expedited cross-dock release or lane diversion to FedEx Express).
    3. A concise, professional 3-sentence escalation email to the {shipment_data['origin_hub']} Dock Supervisor.
    """

    # Option 1: Use OpenAI if OPENAI_API_KEY is available in environment variables
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content

    # Option 2: Use Gemini if GEMINI_API_KEY is available in environment variables
    if os.getenv("GEMINI_API_KEY"):
        from google import genai

        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text

    # Option 3: Zero-Setup Local Diagnostic Engine (Runs without API keys)
    primary_driver = (
        f"excessive origin warehouse dwell time ({shipment_data['warehouse_dwell_hrs']} hrs)"
        if shipment_data["warehouse_dwell_hrs"] > 12
        else f"carrier baseline constraints ({shipment_data['carrier']}) and {shipment_data['weather_severity']} conditions"
    )
    reroute_advice = (
        "Divert lane to FedEx Express (66.4% network reliability) and prioritize immediate bay clearance."
        if shipment_data["carrier"] != "FedEx Express"
        else "Authorize priority loading bay release within 60 minutes to protect SLA compliance."
    )

    return f"""[AI DIAGNOSTIC REPORT - {shipment_data['shipment_id']}]
1. ROOT CAUSE: High SLA breach risk ({delay_prob:.1%}) driven primarily by {primary_driver} at {shipment_data['origin_hub']}.
2. RECOMMENDED ACTION: {reroute_advice}
3. ESCALATION EMAIL DRAFT:
   Subject: URGENT SLA ALERT: {shipment_data['shipment_id']} ({delay_prob:.1%} Delay Risk)
   Hi {shipment_data['origin_hub']} Operations Team, Shipment {shipment_data['shipment_id']} ({shipment_data['carrier']}) has been flagged by our predictive risk engine due to {primary_driver}. Please prioritize dock loading immediately or confirm lane diversion to avoid a contractual SLA breach."""


def evaluate_shipment(shipment_data: dict, threshold: float = 0.65):
    input_df = pd.DataFrame([shipment_data]).drop(columns=["shipment_id"])
    delay_prob = model.predict_proba(input_df)[0][1]

    print(f"\n==========================================")
    print(f"SHIPMENT AUDIT: {shipment_data['shipment_id']} ({shipment_data['carrier']})")
    print(f"Predicted SLA Delay Risk: {delay_prob:.1%}")
    print(f"==========================================")

    if delay_prob >= threshold:
        print("STATUS: 🚨 HIGH RISK — Triggering AI Escalation Agent...\n")
        print(generate_ai_mitigation_plan(shipment_data, delay_prob))
    else:
        print("STATUS: ✅ LOW RISK — Cleared for standard dispatch.")


# Test with two realistic shipments from your network
if __name__ == "__main__":
    # Case 1: High-risk CN Rail shipment stuck at Toronto Hub
    high_risk_shipment = {
        "shipment_id": "SHP-109482",
        "carrier": "CN Rail",
        "origin_hub": "Toronto_Hub",
        "shipping_mode": "Intermodal Rail",
        "distance_km": 2450,
        "package_weight_kg": 42.5,
        "warehouse_dwell_hrs": 22.5,
        "weather_severity": "Rain/Snow",
        "is_peak_season": 1,
    }

    # Case 2: Smooth FedEx Express shipment
    low_risk_shipment = {
        "shipment_id": "SHP-109483",
        "carrier": "FedEx Express",
        "origin_hub": "Mississauga_Hub",
        "shipping_mode": "Express",
        "distance_km": 620,
        "package_weight_kg": 8.0,
        "warehouse_dwell_hrs": 5.5,
        "weather_severity": "Clear",
        "is_peak_season": 0,
    }

    evaluate_shipment(high_risk_shipment)
    evaluate_shipment(low_risk_shipment)
