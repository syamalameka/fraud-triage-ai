"""
Streamlit Investigator Dashboard
Step 5 & 6: Interactive Fraud Triage Dashboard with Closed-Loop Feedback
Run command: streamlit run app_streamlit.py
"""

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3

from src.gmm_clustering import GMMFraudClusterer
from src.rankers import LightGBMClosedLoopRanker
from src.triage_engine import PrioritizationEngine
from src.feedback_loop import FeedbackManager, retrain_model_with_feedback

st.set_page_config(
    page_title="Fraud Investigation Prioritization Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .card-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.08);
        border-left: 6px solid #e74c3c;
        margin-bottom: 20px;
    }
    .badge-high {
        background-color: #e74c3c;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-medium {
        background-color: #f39c12;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: #27ae60;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-exploit {
        background-color: #2980b9;
        color: white;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 0.75rem;
    }
    .badge-explore {
        background-color: #8e44ad;
        color: white;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    data_path = os.path.join(PROJECT_ROOT, "data", "accounts_data.csv")
    df = pd.read_csv(data_path)
    
    # Fit GMM
    gmm = GMMFraudClusterer(n_components=3, random_state=42)
    # Give cold-start 30 verified seeds
    initial_labels = np.zeros(len(df))
    initial_labels[df[df["is_fraud"] == 1].index[:30]] = 1
    gmm.fit(df, initial_fraud_labels=initial_labels)
    df_transformed = gmm.transform(df)

    # Initial Model
    ranker = LightGBMClosedLoopRanker(random_state=42)
    seed_pool = df_transformed.iloc[:100]
    ranker.fit(seed_pool, seed_pool["is_fraud"].values)

    engine = PrioritizationEngine(daily_capacity=40, exploration_ratio=0.20, random_state=42)
    fb_manager = FeedbackManager()
    
    return df_transformed, gmm, ranker, engine, fb_manager

df_transformed, gmm, ranker, engine, fb_manager = load_system()

# Sidebar
st.sidebar.title("🛡️ Triage Controls")
st.sidebar.markdown("**Daily Capacity Budget:** 40 accounts")
st.sidebar.markdown("- 🎯 **32 Exploitation:** Highest risk scores")
st.sidebar.markdown("- 🔭 **8 Exploration:** Novel pattern discovery")

# Fetch already reviewed accounts
reviewed_ids = fb_manager.get_reviewed_account_ids()
st.sidebar.metric("Total Reviews Recorded", len(reviewed_ids))

if st.sidebar.button("🔄 Retrain LightGBM with Feedback"):
    labeled_fb = fb_manager.get_labeled_feedback()
    if len(labeled_fb) > 0:
        retrain_model_with_feedback(ranker, df_transformed, labeled_fb)
        st.sidebar.success(f"Model retrained with {len(labeled_fb)} human labels!")
    else:
        st.sidebar.warning("No new labels found yet to retrain.")

if st.sidebar.button("⚠️ Reset Feedback Database"):
    fb_manager.reset_feedback_db()
    st.sidebar.info("Database reset.")
    st.rerun()

# Main Header
st.title("Fraud Investigation Prioritization")
st.caption("Closed-Loop, Budget-Aware Triage System for Invoicing Platforms")

# Generate Daily Queue
scores = ranker.score(df_transformed)
engine.set_already_inspected(reviewed_ids)
daily_batch = engine.select_daily_batch(df_transformed, scores, exploration_mode="uncertainty")

# KPI Summary
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Today's Queue Size", len(daily_batch))
with col2:
    st.metric("Exploitation Slots", (daily_batch["selection_type"] == "exploitation").sum())
with col3:
    st.metric("Exploration Slots", (daily_batch["selection_type"] == "exploration").sum())
with col4:
    avg_score = daily_batch["risk_score"].mean() if len(daily_batch) > 0 else 0
    st.metric("Avg Queue Risk Score", f"{avg_score:.2f}")

st.markdown("---")

if len(daily_batch) == 0:
    st.success("🎉 All pending accounts have been investigated!")
else:
    # Review Account Carousel/List
    selected_idx = st.selectbox(
        "Select Account from Prioritized Queue:",
        options=range(len(daily_batch)),
        format_func=lambda i: f"#{i+1} | {daily_batch.iloc[i]['business_name']} ({daily_batch.iloc[i]['account_id']}) - Score: {daily_batch.iloc[i]['risk_score']:.2f} [{daily_batch.iloc[i]['selection_type'].upper()}]"
    )

    account = daily_batch.iloc[selected_idx]

    # Visual Card Display
    badge_class = "badge-high" if account["risk_score"] >= 0.70 else ("badge-medium" if account["risk_score"] >= 0.40 else "badge-low")
    type_badge = "badge-exploit" if account["selection_type"] == "exploitation" else "badge-explore"

    st.markdown(f"""
    <div class="card-container">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h2 style="margin: 0;">{account['business_name']}</h2>
            <div>
                <span class="{type_badge}">{account['selection_type'].upper()}</span>
                <span class="{badge_class}">{account['risk_level'].upper()} RISK</span>
            </div>
        </div>
        <p style="color: #7f8c8d; margin-top: 5px;">
            Account ID: <b>{account['account_id']}</b> · Age: <b>Day {int(account['account_age_days'])} of free trial</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("📊 Account Activity Metrics")
        metrics_table = pd.DataFrame({
            "Activity Metric": [
                "Clients imported",
                "Invoices created",
                "Emails sent",
                "Expenses created",
                "Declined payments",
                "Total invoice amount"
            ],
            "Count / Value": [
                f"{int(account['clients_added'])}",
                f"{int(account['invoices_created'])}",
                f"{int(account['emails_sent'])}",
                f"{int(account['expenses_created'])}",
                f"{int(account['declined_payments'])}",
                f"${account['total_invoice_amount']:,.2f}"
            ]
        })
        st.table(metrics_table)

    with c2:
        st.subheader("🤖 AI/ML Model Assessment")
        st.metric(
            label="Model Risk Score",
            value=f"{account['risk_score']:.2f}",
            help="Model output interpreted as risk score; not a confirmed calibrated probability."
        )
        st.info("ℹ️ *Illustrative score, not a confirmed fraud probability.*")

        st.markdown(f"**GMM Cluster Analysis:** Assigned to Cluster `#{int(account['gmm_cluster'])}`")
        if account["is_in_frc"] == 1:
            st.error("⚠️ Member of **Fraud Risk Cluster (FRC)** (High invoice velocity & low expense profile).")
        else:
            st.success("✅ Outside standard Fraud Risk Cluster.")

        st.markdown(f"- Distance to FRC Centroid: `{account['distance_to_frc']:.2f}`")
        st.markdown(f"- Selection Queue Rationale: `{account['selection_type']}`")

    st.markdown("---")
    st.subheader("⚖️ Investigator Decision (Human-in-the-Loop)")

    with st.form(key=f"decision_form_{account['account_id']}"):
        notes = st.text_input("Investigator Verification Notes (optional):", placeholder="e.g. Verified client website, contact details mismatched.")
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        fraud_submit = col_btn1.form_submit_button("🚨 Confirm Fraud", use_container_width=True)
        gen_submit = col_btn2.form_submit_button("✅ Mark Genuine", use_container_width=True)
        uncert_submit = col_btn3.form_submit_button("❓ Flag Uncertain", use_container_width=True)

        if fraud_submit or gen_submit or uncert_submit:
            decision = "Fraud" if fraud_submit else ("Genuine" if gen_submit else "Uncertain")
            dec_df = pd.DataFrame([{
                "account_id": account["account_id"],
                "business_name": account["business_name"],
                "day_number": 1,
                "risk_score": float(account["risk_score"]),
                "selection_type": str(account["selection_type"]),
                "ranking_method": "LightGBM Closed-Loop",
                "investigator_decision": decision,
                "notes": notes,
                "actual_is_fraud": int(account["is_fraud"])
            }])
            fb_manager.save_investigation_decisions(dec_df)
            st.success(f"Recorded decision for **{account['business_name']}**: `{decision}`. Saved to feedback database!")
            st.rerun()

st.markdown("---")
st.caption("Fraud Triage Prioritization System · Research Benchmark Suite")
