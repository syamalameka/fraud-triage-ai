"""
Realistic Synthetic Invoicing Platform Dataset Generator
Simulates 10,000 registered accounts (~9,800 genuine, ~200 fraud)
Features:
  - emails_sent
  - invoices_created
  - clients_added
  - expenses_created
  - declined_payments
  - account_age_days
  - total_invoice_amount
  - fraud_type (genuine, burst_invoice, card_testing, sleeper_burst)
"""

import os
import sqlite3
import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(DATA_DIR, "accounts_data.csv")
DEFAULT_DB = os.path.join(DATA_DIR, "feedback_store.db")

np.random.seed(42)

def generate_accounts_dataset(n_total=10000, n_fraud=200, output_csv=None):
    if output_csv is None:
        output_csv = DEFAULT_CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    n_genuine = n_total - n_fraud

    # Prefixes and suffixes for business names
    prefixes = ["Acme", "Apex", "BluePeak", "Vertex", "Summit", "Horizon", "Nova", "Catalyst", "Zenith", "Prime",
                "Metro", "SilverLine", "Global", "NextGen", "Omni", "Vanguard", "Frontier", "Pinnacle", "Echo", "Swift"]
    industries = ["Print Co", "Logistics", "Consulting", "Design Studio", "Tech Labs", "Creative", "Legal Services",
                  "Retailers", "Marketing", "Solutions", "Media", "Digital", "Trading", "Engineering", "Ventures"]

    def make_names(count):
        p = np.random.choice(prefixes, size=count)
        s = np.random.choice(industries, size=count)
        nums = np.random.randint(10, 999, size=count)
        return [f"{p[i]} {s[i]} #{nums[i]}" for i in range(count)]

    # --- 1. Genuine Accounts (9,800) ---
    # Distribution 1A: Typical Small Businesses (~6,000)
    n_sb = int(n_genuine * 0.60)
    sb_age = np.random.randint(10, 180, size=n_sb)
    sb_invoices = np.clip(np.random.poisson(lam=12, size=n_sb) + (sb_age // 10), 1, 150)
    sb_emails = np.clip(sb_invoices + np.random.randint(-2, 5, size=n_sb), 1, 200)
    sb_clients = np.clip((sb_invoices * 0.4).astype(int) + np.random.randint(1, 5, size=n_sb), 1, 50)
    sb_expenses = np.clip((sb_invoices * 0.35).astype(int) + np.random.randint(0, 4, size=n_sb), 0, 80)
    sb_declines = np.random.binomial(n=sb_invoices, p=0.015)
    sb_amount = sb_invoices * np.random.uniform(150, 800, size=n_sb)
    sb_type = ["genuine_smb"] * n_sb

    # Distribution 1B: Low Activity Freelancers (~2,500)
    n_free = int(n_genuine * 0.25)
    free_age = np.random.randint(5, 120, size=n_free)
    free_invoices = np.random.randint(1, 6, size=n_free)
    free_emails = np.clip(free_invoices + np.random.randint(0, 3, size=n_free), 1, 10)
    free_clients = np.random.randint(1, 3, size=n_free)
    free_expenses = np.random.binomial(n=free_invoices, p=0.2)
    free_declines = np.random.binomial(n=free_invoices, p=0.01)
    free_amount = free_invoices * np.random.uniform(50, 400, size=n_free)
    free_type = ["genuine_freelance"] * n_free

    # Distribution 1C: Established High Volume Businesses (~1,300)
    n_corp = n_genuine - n_sb - n_free
    corp_age = np.random.randint(60, 365, size=n_corp)
    corp_invoices = np.random.randint(40, 300, size=n_corp)
    corp_emails = (corp_invoices * np.random.uniform(0.9, 1.2, size=n_corp)).astype(int)
    corp_clients = (corp_invoices * np.random.uniform(0.3, 0.6, size=n_corp)).astype(int)
    corp_expenses = (corp_invoices * np.random.uniform(0.2, 0.5, size=n_corp)).astype(int)
    corp_declines = np.random.binomial(n=corp_invoices, p=0.02)
    corp_amount = corp_invoices * np.random.uniform(300, 2500, size=n_corp)
    corp_type = ["genuine_enterprise"] * n_corp

    # Combine Genuine
    gen_df = pd.DataFrame({
        "business_name": make_names(n_genuine),
        "account_age_days": np.concatenate([sb_age, free_age, corp_age]),
        "invoices_created": np.concatenate([sb_invoices, free_invoices, corp_invoices]),
        "emails_sent": np.concatenate([sb_emails, free_emails, corp_emails]),
        "clients_added": np.concatenate([sb_clients, free_clients, corp_clients]),
        "expenses_created": np.concatenate([sb_expenses, free_expenses, corp_expenses]),
        "declined_payments": np.concatenate([sb_declines, free_declines, corp_declines]),
        "total_invoice_amount": np.concatenate([sb_amount, free_amount, corp_amount]).round(2),
        "fraud_pattern": np.concatenate([sb_type, free_type, corp_type]),
        "is_fraud": 0
    })

    # --- 2. Fraud Accounts (200) ---
    # Pattern A: Velocity Burst (Example 1 from prompt: Acme Print Co - Day 3, 60 invoices, 58 emails, 3 declines, 0 expenses)
    n_fa = 90
    fa_age = np.random.randint(1, 5, size=n_fa)  # brand new accounts in free trial
    fa_invoices = np.random.randint(45, 95, size=n_fa)
    fa_emails = (fa_invoices * np.random.uniform(0.92, 1.05, size=n_fa)).astype(int)
    fa_clients = np.random.randint(10, 25, size=n_fa)
    fa_expenses = np.zeros(n_fa, dtype=int)  # zero business expenses!
    fa_declines = np.random.randint(2, 8, size=n_fa)
    fa_amount = fa_invoices * np.random.uniform(600, 3000, size=n_fa)
    fa_type = ["burst_invoice_fraud"] * n_fa

    # Pattern B: Card Testing / Merchant Terminal Fraud
    n_fb = 60
    fb_age = np.random.randint(2, 10, size=n_fb)
    fb_invoices = np.random.randint(20, 50, size=n_fb)
    fb_emails = np.random.randint(5, 20, size=n_fb)
    fb_clients = np.random.randint(2, 6, size=n_fb)
    fb_expenses = np.random.randint(0, 2, size=n_fb)
    fb_declines = np.random.randint(8, 25, size=n_fb)  # huge number of declined payments!
    fb_amount = fb_invoices * np.random.uniform(10, 50, size=n_fb)
    fb_type = ["card_testing_fraud"] * n_fb

    # Pattern C: Novel / Evolving Sleeper Fraud (low early activity, sudden divergence; hard for static models)
    n_fc = n_fraud - n_fa - n_fb  # 50 accounts
    fc_age = np.random.randint(15, 45, size=n_fc)
    fc_invoices = np.random.randint(15, 35, size=n_fc)
    fc_emails = np.random.randint(30, 80, size=n_fc)  # spam emails
    fc_clients = np.random.randint(20, 60, size=n_fc)  # harvested client lists
    fc_expenses = np.random.binomial(n=fc_invoices, p=0.03)
    fc_declines = np.random.randint(1, 4, size=n_fc)
    fc_amount = fc_invoices * np.random.uniform(800, 5000, size=n_fc)
    fc_type = ["sleeper_harvesting_fraud"] * n_fc

    fraud_df = pd.DataFrame({
        "business_name": make_names(n_fraud),
        "account_age_days": np.concatenate([fa_age, fb_age, fc_age]),
        "invoices_created": np.concatenate([fa_invoices, fb_invoices, fc_invoices]),
        "emails_sent": np.concatenate([fa_emails, fb_emails, fc_emails]),
        "clients_added": np.concatenate([fa_clients, fb_clients, fc_clients]),
        "expenses_created": np.concatenate([fa_expenses, fb_expenses, fc_expenses]),
        "declined_payments": np.concatenate([fa_declines, fb_declines, fc_declines]),
        "total_invoice_amount": np.concatenate([fa_amount, fb_amount, fc_amount]).round(2),
        "fraud_pattern": np.concatenate([fa_type, fb_type, fc_type]),
        "is_fraud": 1
    })

    # Combine and shuffle
    df = pd.concat([gen_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.insert(0, "account_id", [f"ACC_{i+1:05d}" for i in range(len(df))])

    # Insert Acme Print Co specifically as ACC_00001 for illustrative alignment with prompt
    acme_idx = df[df["fraud_pattern"] == "burst_invoice_fraud"].index[0]
    df.loc[acme_idx, "business_name"] = "Acme Print Co"

    df.to_csv(output_csv, index=False)
    print(f"Generated {len(df)} accounts ({df['is_fraud'].sum()} fraud) -> {output_csv}")
    return df

def initialize_feedback_database(db_path=None):
    if db_path is None:
        db_path = DEFAULT_DB
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS investigations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        account_id TEXT NOT NULL,
        business_name TEXT,
        day_number INTEGER,
        risk_score REAL,
        selection_type TEXT, -- 'exploitation' or 'exploration'
        ranking_method TEXT,
        investigator_decision TEXT, -- 'Fraud', 'Genuine', 'Uncertain'
        notes TEXT,
        actual_is_fraud INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS daily_audit_summary (
        day_number INTEGER PRIMARY KEY,
        method_name TEXT,
        budget_reviewed INTEGER,
        frauds_found INTEGER,
        precision_at_k REAL,
        cumulative_frauds INTEGER
    )
    """)

    conn.commit()
    conn.close()
    print(f"Initialized SQLite feedback database -> {db_path}")

if __name__ == "__main__":
    generate_accounts_dataset()
    initialize_feedback_database()
