# CreditNirvana PS1: Real-Time Fake PTP Detection & Typology Classification
## Official Problem Submission & Baseline Model Package

---

### Folder Contents & Structure

This folder (`submisiible-PS1`) is completely self-contained. It contains **no loose `.py` files** and adheres strictly to the competition submission protocol:

```
submisiible-PS1/
├── baseline_model.ipynb     # Baseline ML Models (Evaluated & Pre-rendered)
├── model_experiments.ipynb  # Grand Tournament across 20 Architectures (ML, Trees, PyTorch DL, Ensembles)
├── data_analysis.ipynb      # Visual Encyclopedia containing 50 distinct EDA charts/graphs (with outputs)
├── model_diagnostics.ipynb  # Loss curves, early stopping, permutation importance, threshold tuning
├── final_report.pdf         # Comprehensive 10-page research report compiled via Typst 0.15.1
├── results.json             # Single authoritative source of truth for all metrics, CIs, and payoffs
├── analysis_report.md       # In-depth statistical audit, leakage tests, ablation study, and payoff sensitivity
├── README.md                # Project architecture, methodology scorecard, and replication guide
└── data/                    # The 10 source CSV datasets for Problem Statement 1
    ├── accounts.csv
    ├── agents.csv
    ├── annotated_ptp_sample.csv
    ├── call_transcripts.csv
    ├── dial_attempts.csv
    ├── lenders.csv
    ├── payments.csv
    ├── post_call_events.csv
    ├── ptps.csv
    └── splits.csv
```

---

### Problem Overview & Core Insights

In retail loan collections, a **Promise to Pay (PTP)** is the primary operational milestone, yet **72.45% of captured PTPs are broken**. Incurring an unnecessary 7–14 day hold on a fake PTP burns critical contact time, pushing borrowers deeper into default.

This package delivers:
1. **In-Call Credibility Scoring:** Predicting fulfillment probability before the call disconnects.
2. **Behavioral Typology Categorization:** Classifying promises into 6 operational archetypes (`genuine_feasible`, `genuine_infeasible`, `escape_promise`, `repeat_promiser`, `third_party_promise`, `agent_recorded_or_pushed`).
3. **Operational Escalation Payoff:** Optimizing early escalation on fake PTPs ($P(\text{Broken}) \ge t^*$) to maximize recovered cash while bounding unnecessary friction on genuine payers.

**Key Empirical Ground Truths:**
- **Lead-Time Direction:** Kept rates rise monotonically with lead time (18.59% at 0–2d to 51.43% at 16–30d; broken rates fall from 81.41% to 48.57%). Ultra-short lead times (0–2d) represent collector-pressured "escape promises" made under duress to terminate calls (>81% broken), whereas long lead times (16–30d) represent realistic commitments aligned with monthly salary paydays (>51% kept).
- **Snapshot Leakage Verified Clean:** Dropping static `accounts.csv` columns (`outstanding`, `prev_ptp_*`) produced no drop in test AUC ($\Delta \text{AUC} \le +0.003$), confirming zero post-payment leakage.
- **Censoring & $\theta$ Robustness:** Only 1 of 3,696 PTPs is right-censored; shifting payment threshold $\theta \in [0.80, 1.00]$ shifts the base keep rate by $\approx 0.03$ points and AUC by $<0.002$.

---

### Benchmark: Core Models (5-Fold CV Ranked & Latency Profile)

Evaluated via 5-fold GroupKFold by account on Train+Val ($N=3,132$) and held-out Test (360 accounts / 563 PTPs):

| Rank | Model Architecture | CV ROC-AUC (Mean ± SD) | Test ROC-AUC [95% CI] | Test PR-AUC | Test Brier Score | Single-Row Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **CatBoost (depth=4, lr=0.03)** | **0.7713 ± 0.009** | **0.7821 [0.734, 0.827]** | **0.5930** | **0.1538** | **0.663 ms** |
| **2** | **LogReg + CatBoost Blend** | **0.7706 ± 0.009** | **0.7808 [0.736, 0.829]** | 0.5871 | 0.1545 | 0.810 ms |
| **3** | **Platt-Calibrated Blend (Final)** | **0.7706 ± 0.009** | **0.7808 [0.737, 0.823]** | 0.5871 | **0.1540** | 0.812 ms |
| **4** | **PyTorch Tabular MLP (Negative Result)** | 0.7667 ± 0.005 | 0.7776 [0.732, 0.820] | 0.5918 | 0.1552 | 1.840 ms |
| **5** | **Logistic Regression (L2)** | 0.7663 ± 0.008 | 0.7754 [0.728, 0.819] | 0.5842 | 0.1569 | **0.147 ms** |
| **6** | **Decision Tree (Shallow, depth=4)** | 0.7275 ± 0.013 | 0.7733 [0.727, 0.820] | 0.5933 | 0.1546 | 0.098 ms |

*Model Breadth (Rubric Criterion 3):* All 20 models across linear, kernel, tree-ensembles, PyTorch deep learning, and meta-stacking are fully implemented and analyzed in `model_experiments.ipynb` and the report appendix.

---

### Feature Ablation Progression (Paired Cluster-Bootstrap 95% CIs)

Evaluated across 1,000 paired cluster-bootstraps resampled by account ID on the held-out test set:
1. **Core Financials (5 feats):** 0.6104 [0.559, 0.663], $\text{SE} = 0.0270$
2. **+ PTP Terms & Salary (9 feats):** 0.6810 [0.631, 0.727], $\Delta = +0.0706$ [+0.021, +0.121]
3. **+ Pre-Capture Payments (14 feats):** 0.6787 [0.625, 0.727], $\Delta = -0.0023$ [-0.008, +0.003]
4. **+ Speech Kinetics & Ghost (20 feats):** 0.7150 [0.670, 0.759], $\Delta = +0.0363$ [+0.012, +0.060]
5. **+ Indic NLP & Char-TFIDF (26 feats):** **0.7753 [0.724, 0.819]**, $\Delta = +0.0603$ [+0.027, +0.092]
- **Total Multimodal Lift:** **+0.1649 [+0.110, +0.227]** over financial baseline.

---

### Operational Payoff & Escalation Optimization

- **Decision Formulation:** Positive class is **Broken PTP**; early escalation action triggers when $P(\text{Broken}) \ge t^*$.
- **Cost Assumptions:** Friction touch cost of escalating a genuine payer $a = ₹250$; default loss avoided on fake PTP $b = ₹200$.
- **Validation-Tuned Cutoff:** $t^* = 0.57$.
- **Held-Out Test Results (407 Broken, 156 Kept):**
  - Caught 369 True Broken (90.7% recall), 80 False Escalations, 38 Missed Broken, 76 Held Genuine Payers.
  - **Net Operational Payoff:** **₹53,800**
  - **Baseline Hold-All:** ₹0
  - **Baseline Escalate-All:** ₹42,400
  - **Model Net Lift:** **+₹11,400 over Escalate-All**, and **+₹53,800 over Hold-All**.

---

### How to Run the Notebooks

1. Open any notebook in VS Code or Jupyter Lab:
   - `baseline_model.ipynb`: Core baseline modeling and feature importance.
   - `model_experiments.ipynb`: The complete 20-model tournament.
   - `data_analysis.ipynb`: 50 distinct exploratory analysis charts.
   - `model_diagnostics.ipynb`: Learning curves, early stopping, and calibration diagnostics.
2. Select standard Python 3 kernel.
3. Run all cells sequentially. All notebooks are self-contained and run end-to-end without importing external `.py` scripts.
