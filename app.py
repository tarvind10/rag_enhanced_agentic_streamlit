import streamlit as st
import pandas as pd
import json
from typing import Dict, Any, Optional

# Import your existing detection function
from multi_agent_shell_detector_genAI import detect_shell_entities

st.set_page_config(page_title="Shell Entity Detector", layout="wide")
st.title("🛡️ Shell Entity Detector (AML/KYC)")

# Define example inputs
EXAMPLE_INPUT_1 = """{
  "outgoing": [
    {
      "transaction_id": "OUT-1001",
      "account_id": "ACC-001",
      "customer_name": "Alice Brown",
      "counterparty_name": "Northwind Trading Ltd",
      "amount": 250000,
      "currency": "USD",
      "country": "UAE",
      "purpose": "Import settlement",
      "source_of_funds": "Salary"
    },
    {
      "transaction_id": "OUT-1002",
      "account_id": "ACC-002",
      "customer_name": "Marcus Lee",
      "counterparty_name": "Global Capital Holdings",
      "amount": 980000,
      "currency": "USD",
      "country": "UAE",
      "purpose": "Investment",
      "source_of_funds": "Unknown"
    }
  ],
  "kyc": [
    {
      "customer_id": "CUST-001",
      "customer_name": "Northwind Trading Ltd",
      "country": "UAE",
      "occupation": "Trading company",
      "source_of_funds": "Trade revenue",
      "business_name": "Northwind Trading Ltd",
      "beneficial_owner": "A. Rahman"
    },
    {
      "customer_id": "CUST-002",
      "customer_name": "Global Capital Holdings",
      "country": "UAE",
      "occupation": "Investment company",
      "source_of_funds": "Unknown",
      "business_name": "Global Capital Holdings",
      "beneficial_owner": "Unknown"
    }
  ],
  "cases": [
    {
      "case_id": "CASE-101",
      "entity_name": "Global Capital Holdings",
      "scenario_name": "Nominee capital movement",
      "alert_type": "Shell-pattern entity review",
      "severity": "HIGH",
      "country": "UAE",
      "description": "Unknown funds source and large transfers.",
      "status": "Escalated"
    }
  ]
}"""

EXAMPLE_INPUT_2 = """{
  "outgoing": [
    {
      "transaction_id": "OUT-2001",
      "account_id": "ACC-101",
      "customer_name": "Omar Haddad",
      "counterparty_name": "Tower Legacy Partners",
      "amount": 1200000,
      "currency": "USD",
      "country": "UAE",
      "purpose": "Unclear",
      "source_of_funds": "Unknown"
    }
  ],
  "incoming": [
    {
      "transaction_id": "IN-2001",
      "account_id": "ACC-102",
      "customer_name": "Aisha Khan",
      "counterparty_name": "Tower Legacy Partners",
      "amount": 1600000,
      "currency": "USD",
      "country": "BVI",
      "purpose": "Unknown",
      "source_of_funds": "Unclear"
    }
  ],
  "kyc": [
    {
      "customer_id": "CUST-005",
      "customer_name": "Tower Legacy Partners",
      "country": "BVI",
      "occupation": "Investment vehicle",
      "source_of_funds": "Unclear",
      "business_name": "Tower Legacy Partners",
      "beneficial_owner": "Unknown"
    }
  ],
  "cases": [
    {
      "case_id": "CASE-101",
      "entity_name": "Tower Legacy Partners",
      "scenario_name": "Offshore investment layering",
      "alert_type": "Unusual offshore structure",
      "severity": "HIGH",
      "country": "BVI",
      "description": "High-value cash movement tied to an investment vehicle with unclear beneficial ownership.",
      "status": "Open"
    }
  ]
}"""

def parse_input_text(text: str) -> Dict[str, Optional[pd.DataFrame]]:
    text = text.strip()
    outgoing_df = pd.DataFrame()
    incoming_df = pd.DataFrame()
    kyc_df = pd.DataFrame()
    cases_df = pd.DataFrame()

    if not text:
        return {"outgoing_df": outgoing_df, "incoming_df": incoming_df, "kyc_df": kyc_df, "cases_df": cases_df}

    try:
        data = json.loads(text)
    except Exception:
        return {"outgoing_df": outgoing_df, "incoming_df": incoming_df, "kyc_df": kyc_df, "cases_df": cases_df}

    if isinstance(data, dict):
        outgoing_df = pd.DataFrame(data.get("outgoing", []))
        incoming_df = pd.DataFrame(data.get("incoming", []))
        kyc_df = pd.DataFrame(data.get("kyc", []))
        cases_df = pd.DataFrame(data.get("cases", []))
    else:
        outgoing_df = pd.DataFrame(data)

    return {"outgoing_df": outgoing_df, "incoming_df": incoming_df, "kyc_df": kyc_df, "cases_df": cases_df}

# Initialize session state for text area persistence
if "json_input" not in st.session_state:
    st.session_state["json_input"] = ""

st.subheader("Input Transaction JSON")

# Quick example loader buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Load Example 1"):
        st.session_state["json_input"] = EXAMPLE_INPUT_1
with col2:
    if st.button("Load Example 2"):
        st.session_state["json_input"] = EXAMPLE_INPUT_2

user_input = st.text_area("Paste JSON input data here:", value=st.session_state["json_input"], height=250, key="json_input_area")
st.session_state["json_input"] = user_input

if st.button("Run Detector"):
    with st.spinner("Analyzing entities..."):
        parsed = parse_input_text(st.session_state["json_input"])
        outgoing_df = parsed["outgoing_df"]
        incoming_df = parsed["incoming_df"]
        kyc_df = parsed["kyc_df"]
        cases_df = parsed["cases_df"]

        detection_df = detect_shell_entities(
            outgoing_df=outgoing_df if not outgoing_df.empty else None,
            incoming_df=incoming_df if not incoming_df.empty else None,
            kyc_df=kyc_df if not kyc_df.empty else None,
            cases_df=cases_df if not cases_df.empty else None,
        )

        if detection_df is None or detection_df.empty:
            st.warning("No entities detected.")
        else:
            st.success("Detection Complete!")
            st.dataframe(detection_df)