"""
Standalone Interactive Web Dashboard Server (Zero Dependencies)
Step 5 & 6: Prioritization Dashboard with Real-time Decision Logging & Retraining
Runs out-of-the-box using Python standard library (http.server)
"""

import os
import sys
import json
import sqlite3
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

from src.gmm_clustering import GMMFraudClusterer
from src.rankers import LightGBMClosedLoopRanker
from src.triage_engine import PrioritizationEngine
from src.feedback_loop import FeedbackManager, retrain_model_with_feedback

# Global System State
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "accounts_data.csv")
df_all = pd.read_csv(DATA_PATH)

# Initialize GMM
gmm = GMMFraudClusterer(n_components=3, random_state=42)
initial_labels = np.zeros(len(df_all))
initial_labels[df_all[df_all["is_fraud"] == 1].index[:30]] = 1
gmm.fit(df_all, initial_fraud_labels=initial_labels)
df_transformed = gmm.transform(df_all)

# Initialize Model
ranker = LightGBMClosedLoopRanker(random_state=42)
seed_pool = df_transformed.iloc[:100]
ranker.fit(seed_pool, seed_pool["is_fraud"].values)

engine = PrioritizationEngine(daily_capacity=40, exploration_ratio=0.20, random_state=42)
fb_manager = FeedbackManager()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fraud Investigation Prioritization Dashboard</title>
    <style>
        :root {
            --primary: #2563eb;
            --danger: #dc2626;
            --warning: #d97706;
            --success: #16a34a;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
        }
        * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { margin: 0; background: var(--bg); color: var(--text); padding: 24px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; border-bottom: 1px solid #e2e8f0; padding-bottom: 16px; }
        .header h1 { margin: 0; font-size: 24px; color: #0f172a; }
        .header p { margin: 4px 0 0; color: #64748b; font-size: 14px; }
        .grid-kpi { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
        .kpi-card { background: var(--card-bg); padding: 18px; border-radius: 10px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        .kpi-title { font-size: 13px; color: #64748b; text-transform: uppercase; font-weight: 600; }
        .kpi-val { font-size: 26px; font-weight: 700; margin-top: 6px; }
        .main-layout { display: grid; grid-template-columns: 360px 1fr; gap: 24px; }
        .sidebar-list { background: var(--card-bg); border-radius: 10px; border: 1px solid #e2e8f0; padding: 16px; max-height: 720px; overflow-y: auto; }
        .queue-item { padding: 12px; border-radius: 8px; margin-bottom: 8px; cursor: pointer; border: 1px solid #e2e8f0; transition: all 0.2s; }
        .queue-item:hover, .queue-item.active { background: #eff6ff; border-color: var(--primary); }
        .queue-item .title { font-weight: 600; font-size: 14px; }
        .queue-item .meta { font-size: 12px; color: #64748b; display: flex; justify-content: space-between; margin-top: 4px; }
        .badge { padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; color: white; display: inline-block; }
        .badge-exploit { background: #2563eb; }
        .badge-explore { background: #9333ea; }
        .badge-high { background: #dc2626; }
        .badge-med { background: #d97706; }
        .badge-low { background: #16a34a; }
        .account-detail { background: var(--card-bg); border-radius: 10px; border: 1px solid #e2e8f0; padding: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.04); }
        .card-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e2e8f0; padding-bottom: 16px; margin-bottom: 20px; }
        .card-header h2 { margin: 0; font-size: 22px; }
        .metrics-table { width: 100%; border-collapse: collapse; margin: 16px 0; }
        .metrics-table th, .metrics-table td { padding: 12px 14px; text-align: left; border-bottom: 1px solid #f1f5f9; }
        .metrics-table th { background: #f8fafc; font-size: 13px; color: #475569; }
        .metrics-table td { font-size: 14px; font-weight: 500; }
        .risk-callout { background: #fef2f2; border-left: 4px solid var(--danger); padding: 16px; border-radius: 6px; margin-bottom: 20px; }
        .decision-actions { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 20px; }
        .btn { padding: 12px; border-radius: 6px; font-weight: 600; cursor: pointer; border: none; font-size: 14px; transition: opacity 0.2s; }
        .btn:hover { opacity: 0.9; }
        .btn-fraud { background: #dc2626; color: white; }
        .btn-gen { background: #16a34a; color: white; }
        .btn-uncert { background: #475569; color: white; }
        .btn-retrain { background: #0284c7; color: white; padding: 8px 16px; border-radius: 6px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>🛡️ Fraud Investigation Prioritization</h1>
                <p>Closed-Loop, Budget-Aware Triage System (40 Reviews/Day Capacity)</p>
            </div>
            <div>
                <button class="btn btn-retrain" onclick="retrainModel()">🔄 Retrain LightGBM Model</button>
            </div>
        </div>

        <div class="grid-kpi">
            <div class="kpi-card">
                <div class="kpi-title">Daily Review Capacity</div>
                <div class="kpi-val" id="kpiCapacity">40</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Exploitation Slots (80%)</div>
                <div class="kpi-val" id="kpiExploit" style="color: #2563eb;">32</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Exploration Slots (20%)</div>
                <div class="kpi-val" id="kpiExplore" style="color: #9333ea;">8</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Decisions Logged</div>
                <div class="kpi-val" id="kpiDecisions" style="color: #16a34a;">0</div>
            </div>
        </div>

        <div class="main-layout">
            <div class="sidebar-list">
                <h3 style="margin-top: 0; font-size: 16px; color: #334155;">Daily Queue (40 Accounts)</h3>
                <div id="queueList">Loading triage queue...</div>
            </div>

            <div class="account-detail" id="accountDetail">
                <p>Select an account from the queue on the left to start investigation.</p>
            </div>
        </div>
    </div>

    <script>
        let currentQueue = [];
        let currentIndex = 0;

        async function fetchQueue() {
            const res = await fetch('/api/queue');
            const data = await res.json();
            currentQueue = data.queue;
            document.getElementById('kpiDecisions').innerText = data.reviewed_count;
            renderQueueList();
            if (currentQueue.length > 0) {
                renderAccount(0);
            }
        }

        function renderQueueList() {
            const listEl = document.getElementById('queueList');
            if (currentQueue.length === 0) {
                listEl.innerHTML = '<p style="color: #64748b;">No accounts pending review today.</p>';
                return;
            }
            listEl.innerHTML = currentQueue.map((acc, idx) => `
                <div class="queue-item ${idx === currentIndex ? 'active' : ''}" onclick="renderAccount(${idx})">
                    <div style="display: flex; justify-content: space-between;">
                        <span class="title">#${idx+1} ${acc.business_name}</span>
                        <span class="badge ${acc.selection_type === 'exploitation' ? 'badge-exploit' : 'badge-explore'}">
                            ${acc.selection_type === 'exploitation' ? 'EXPLOIT' : 'EXPLORE'}
                        </span>
                    </div>
                    <div class="meta">
                        <span>Risk Score: <b>${acc.risk_score.toFixed(2)}</b></span>
                        <span class="badge ${acc.risk_score >= 0.7 ? 'badge-high' : (acc.risk_score >= 0.4 ? 'badge-med' : 'badge-low')}">
                            ${acc.risk_level.toUpperCase()}
                        </span>
                    </div>
                </div>
            `).join('');
        }

        function renderAccount(idx) {
            currentIndex = idx;
            renderQueueList();
            const acc = currentQueue[idx];
            if (!acc) return;

            const detailEl = document.getElementById('accountDetail');
            detailEl.innerHTML = `
                <div class="card-header">
                    <div>
                        <h2>${acc.business_name}</h2>
                        <span style="color: #64748b; font-size: 13px;">
                            ${acc.account_id} · Day ${acc.account_age_days} of free trial
                        </span>
                    </div>
                    <div>
                        <span class="badge ${acc.selection_type === 'exploitation' ? 'badge-exploit' : 'badge-explore'}" style="font-size: 13px; padding: 4px 10px;">
                            ${acc.selection_type.toUpperCase()}
                        </span>
                        <span class="badge ${acc.risk_score >= 0.7 ? 'badge-high' : (acc.risk_score >= 0.4 ? 'badge-med' : 'badge-low')}" style="font-size: 13px; padding: 4px 10px;">
                            ${acc.risk_level.toUpperCase()} RISK
                        </span>
                    </div>
                </div>

                <div class="risk-callout">
                    <div style="font-weight: 700; font-size: 16px; color: #991b1b;">
                        Model Risk Score: ${acc.risk_score.toFixed(2)}
                    </div>
                    <div style="font-size: 13px; color: #7f1d1d; margin-top: 4px;">
                        <i>Illustrative score, not a confirmed fraud probability. Model ranking output based on GMM Cluster #${acc.gmm_cluster} features.</i>
                    </div>
                </div>

                <h3 style="font-size: 16px; margin: 16px 0 8px;">Activity Summary</h3>
                <table class="metrics-table">
                    <thead>
                        <tr><th>Activity Feature</th><th>Count / Recorded Value</th></tr>
                    </thead>
                    <tbody>
                        <tr><td>Clients imported</td><td><b>${acc.clients_added}</b></td></tr>
                        <tr><td>Invoices created</td><td><b>${acc.invoices_created}</b></td></tr>
                        <tr><td>Emails sent</td><td><b>${acc.emails_sent}</b></td></tr>
                        <tr><td>Expenses created</td><td><b>${acc.expenses_created}</b></td></tr>
                        <tr><td>Declined payments</td><td><b style="color: ${acc.declined_payments > 0 ? '#dc2626' : 'inherit'};">${acc.declined_payments}</b></td></tr>
                        <tr><td>Total invoice amount</td><td><b>$${Number(acc.total_invoice_amount).toLocaleString()}</b></td></tr>
                    </tbody>
                </table>

                <div style="margin-top: 16px;">
                    <label style="font-size: 13px; font-weight: 600;">Investigator Verification Notes:</label>
                    <input type="text" id="investigatorNotes" placeholder="e.g., Checked registered domain, contact details verified" 
                           style="width: 100%; padding: 10px; border-radius: 6px; border: 1px solid #cbd5e1; margin-top: 6px;">
                </div>

                <div class="decision-actions">
                    <button class="btn btn-fraud" onclick="submitDecision('${acc.account_id}', 'Fraud')">🚨 Fraud</button>
                    <button class="btn btn-gen" onclick="submitDecision('${acc.account_id}', 'Genuine')">✅ Genuine</button>
                    <button class="btn btn-uncert" onclick="submitDecision('${acc.account_id}', 'Uncertain')">❓ Uncertain</button>
                </div>
            `;
        }

        async function submitDecision(accId, decision) {
            const notes = document.getElementById('investigatorNotes') ? document.getElementById('investigatorNotes').value : '';
            const acc = currentQueue.find(a => a.account_id === accId);

            const payload = {
                account_id: accId,
                business_name: acc.business_name,
                day_number: 1,
                risk_score: acc.risk_score,
                selection_type: acc.selection_type,
                ranking_method: 'LightGBM Closed-Loop',
                investigator_decision: decision,
                notes: notes,
                actual_is_fraud: acc.is_fraud
            };

            await fetch('/api/decision', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });

            await fetchQueue();
        }

        async function retrainModel() {
            const res = await fetch('/api/retrain', { method: 'POST' });
            const data = await res.json();
            alert(`Model Retrained! Status: ${data.message}`);
            await fetchQueue();
        }

        fetchQueue();
    </script>
</body>
</html>
"""

class TriageRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif parsed.path == "/api/queue":
            reviewed_ids = fb_manager.get_reviewed_account_ids()
            scores = ranker.score(df_transformed)
            engine.set_already_inspected(reviewed_ids)
            daily_batch = engine.select_daily_batch(df_transformed, scores, exploration_mode="uncertainty")
            
            queue_records = daily_batch.to_dict(orient="records")
            response_data = {
                "queue": queue_records,
                "reviewed_count": len(reviewed_ids),
                "total_accounts": len(df_transformed)
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        data = json.loads(body)

        if parsed.path == "/api/decision":
            dec_df = pd.DataFrame([data])
            fb_manager.save_investigation_decisions(dec_df)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success"}).encode("utf-8"))

        elif parsed.path == "/api/retrain":
            global ranker
            labeled_fb = fb_manager.get_labeled_feedback()
            if len(labeled_fb) > 0:
                ranker = retrain_model_with_feedback(ranker, df_transformed, labeled_fb)
                msg = f"Retrained on {len(labeled_fb)} human-in-the-loop labels."
            else:
                msg = "No labels recorded yet to retrain."

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "message": msg}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=8501):
    server = HTTPServer(("127.0.0.1", port), TriageRequestHandler)
    print(f"Fraud Triage Dashboard running at http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    run_server(port)
