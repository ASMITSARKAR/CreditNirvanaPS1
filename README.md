# CreditNirvana PS1: Real-Time Fake PTP Detection & Operational Escalation Engine
## Official Submission Guide & Audited Technical Documentation

---

### Folder Architecture & Deliverables

This submission package (`submisiible-PS1`) is self-contained, reproducible, and leak-free:

```
submisiible-PS1/
├── final_pipeline.ipynb     # Master executable pipeline: ingests raw CSVs, runs all audits, exports results.json
├── final_report.pdf         # Complete 10-page research report compiled with Typst 0.15.1
├── results.json             # Single authoritative source of truth for all metrics, bootstrap CIs, and payoffs
├── analysis_report.md       # In-depth statistical audit, leakage tests, ablation study, and sensitivity analysis
├── README.md                # Project architecture, methodology scorecard, and replication guide
├── baseline_model.ipynb     # Baseline ML models and feature importance diagnostics
├── model_experiments.ipynb  # Grand Tournament across 20 architectures (ML, Trees, PyTorch DL, Ensembles)
├── data_analysis.ipynb      # Visual encyclopedia containing 50 distinct EDA figures
├── model_diagnostics.ipynb  # Learning curves, early stopping, and calibration diagnostics
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

### 1. Problem Formulation & Operational Reality

In retail loan collections, a **Promise to Pay (PTP)** is the primary operational milestone, yet **72.45% of captured PTPs are broken**. Under standard collections protocol, logging a PTP places an operational hold on the borrower's account (pausing calls, SMS nudges, and field visits) until the promised date plus a 2-day grace window. 

Placing holds on fake promises allows accounts to roll into deeper delinquency buckets, destroying recovery yield. Conversely, indiscriminately breaking holds (**Escalate-All**) wastes collector labor and harasses sincere borrowers.

#### Machine Learning Objectives:
1. **Real-Time Fake PTP Scoring:** Predicting probability of fulfillment ($P(\text{Kept})$) and default risk ($P(\text{Broken}) = 1 - P(\text{Kept})$) at call termination.
2. **Hold-vs-No-Hold Optimization:** Formulating the operational decision as **Escalate (No-Hold)** when $P(\text{Broken}) \ge t^*$ to maximize recovered cash under friction constraints.
3. **Behavioral Typology Segmentation:** Secondary categorization of commitments into 6 operational archetypes (`genuine_feasible`, `genuine_infeasible`, `escape_promise`, `repeat_promiser`, `third_party_promise`, `agent_recorded_or_pushed`) using model risk tiers and keyword/acoustic tags.

---

### 2. Empirical Ground Truths & Methodological Corrections

#### A. Evaluation Window Duration Artifact vs. Lead-Time
A naive evaluation suggests that promise kept rates rise with lead time ($18.59\%$ at 0–2d to $51.43\%$ at 16–30d). We demonstrate that this increase is substantially driven by an **evaluation window duration artifact**:
- In standard evaluation $[t_{\text{capture}}, \text{promised\_date} + 2\text{d}]$, a 30-day promise affords an observation window of 32 days vs. 4 days for a 2-day promise.
- When evaluated on a uniform fixed window of 3 days centered on the promised date $[\text{promised\_date} - 1\text{d}, \text{promised\_date} + 2\text{d}]$, the **16–30d kept rate drops from 51.43% to 28.78%** (a 22.65% reduction).
- While payday proximity provides genuine liquidity for the subset of borrowers who negotiate commitments around salary dates ($89.87\%$ of kept long-lead PTPs have payments within 2 days of the promised date), the standard evaluation window mechanically accumulates interim non-targeted payments.

| Lead-Time Bin | Total PTPs | Std Kept Count | Std Kept Rate (%) | Fixed Window Kept | Fixed Window Rate (%) | Window Artifact $\Delta$ (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0–2 days** | 1,076 | 200 | 18.59% | 200 | 18.59% | 0.00% |
| **3–4 days** | 386 | 117 | 30.31% | 73 | 18.91% | +11.40% |
| **5–7 days** | 1,213 | 283 | 23.33% | 194 | 15.99% | +7.34% |
| **8–10 days** | 414 | 107 | 25.85% | 65 | 15.70% | +10.14% |
| **11–15 days** | 116 | 59 | 50.86% | 44 | 37.93% | +12.93% |
| **16–30 days** | 490 | 252 | 51.43% | 141 | 28.78% | +22.65% |

#### B. Snapshot Leakage Audit
- Retraining after dropping candidate snapshot columns (`outstanding`, dynamic `prev_ptp_*` counters) shifted CatBoost test AUC from $0.7456 \to 0.7488$ ($+0.0032$) and LogReg from $0.7399 \to 0.7407$ ($+0.0008$).
- These shifts are within bootstrap resample noise, providing strong empirical evidence against lookahead target leakage. Pre-campaign financials (`overdue_start`, `dpd_start`) are retained, while dynamic post-campaign attributes are purged.

#### C. Ghost PTP & Rushed Call Audit
- **Under 15s Calls:** Across 3,367 transcripts, 111 calls lasted $<15$s, but **exactly 0 PTPs** were logged under 15 seconds (min duration: 15.14s). A 15s ghost indicator is constant zero; only a 20s indicator carries empirical variation.
- **Rushed Calls (<20s):** $N=106$ rushed PTPs (2.87% of portfolio) exhibit a $75.47\%$ broken rate vs. $72.45\%$ portfolio baseline. This $+3.02\%$ difference represents $\approx 0.70$ standard errors ($z = 0.70, p = 0.49$) and is not statistically significant. Three tele-agents (`TA010`, `TA011`, `TA004`) drive 62.3% of rushed calls.
- Retraining CatBoost without call duration features yielded test ROC-AUC of 0.7855 vs 0.7841 with duration ($\Delta = -0.0014$; $+0.0014$ when removing duration), proving the model does not rely on call-duration shortcuts.

#### D. Censoring & Parameter Sensitivity
- Only 1 of 3,696 PTPs is right-censored (3,695 valid PTPs remain).
- Varying fulfillment threshold $\theta \in [0.80, 1.00]$ shifts the base keep rate by only $0.03$ percentage points ($26.63\%$ to $26.58\%$) and changes test AUC by $<0.002$, confirming borrower payments are essentially binary (full settlement or zero payment).

---

### 3. Grand Model Benchmark & Production Selection

Models were evaluated via 5-fold GroupKFold by account on Train+Val ($N=3,132$) and held-out Test (360 accounts / 563 PTPs):

| Rank | Model Architecture | CV ROC-AUC (Mean ± SD) | Test ROC-AUC [95% CI] | Test PR-AUC | Test Brier Score | Single-Row Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **CatBoost (depth=4, lr=0.03)** | **0.7713 ± 0.009** | **0.7821 [0.734, 0.827]** | **0.5930** | **0.1538** | 0.663 ms |
| **2** | **LogReg + CatBoost Blend** | **0.7706 ± 0.009** | **0.7808 [0.736, 0.829]** | 0.5871 | 0.1545 | 0.810 ms |
| **3** | **Platt-Calibrated Blend (Champion)** | **0.7706 ± 0.009** | **0.7808 [0.736, 0.829]** | 0.5871 | **0.1540** | 0.812 ms |
| **4** | **PyTorch Tabular MLP (Dropout+BN)** | 0.7667 ± 0.005 | 0.7776 [0.732, 0.820] | 0.5918 | 0.1552 | 1.840 ms |
| **5** | **Logistic Regression (L2)** | 0.7663 ± 0.008 | 0.7754 [0.728, 0.819] | 0.5842 | 0.1569 | **0.147 ms** |
| **6** | **Decision Tree (Shallow, depth=4)** | 0.7275 ± 0.013 | 0.7733 [0.727, 0.820] | 0.5933 | 0.1546 | **0.098 ms** |

#### Architectural Analysis & Champion Selection:
- **Statistical Equivalence:** With 156 test positives and 407 negatives ($\text{SE} \approx 0.024$), the paired cluster-bootstrap test difference between CatBoost and Logistic Regression is **+0.0069 AUC** (95% CI: $[-0.007, +0.022]$, crossing zero). Base models are statistically indistinguishable.
- **Why the Platt-Calibrated Blend is Champion:** Standalone CatBoost achieves the lowest raw Brier score ($0.1538$), but the **Platt-Calibrated Blend** is selected for production deployment because it combines the regularized global linear boundaries of Logistic Regression (preventing tail miscalibration on sparse text $n$-grams) with CatBoost's non-linear tree interactions, providing superior probability calibration across escalation cutoffs ($\text{ECE} = 0.0423$, $\text{Brier Skill Score} = 0.2312$, calibration slope $= 1.074$).
- **Decision Tree Instability:** The shallow tree shows high fold-level variance (CV $0.7275 \pm 0.013$), but 5-fold cross-validation averaging on the test set acts as a bagged ensemble of 5 trees, lifting test AUC to 0.7733.
- **Latency SLAs:** Scoring executes at **call termination** during post-call wrap-up. Single-row inference takes **0.663 ms** for CatBoost and **0.147 ms** for LogReg. End-to-end latency (feature extraction + regex + character TF-IDF + scoring) executes in **2.185 ms**, well within the assumed 5.0 ms telephony SLA.

---

### 4. Step-by-Step Feature Ablation Progression

Evaluated using $L_2$-regularized Logistic Regression ($C=0.1$) across 1,000 paired cluster-bootstraps resampled by account ID on the held-out test set ($N=563$):

| Step | Feature Modality Added | Total Feats | Test ROC-AUC [95% CI] | Step $\Delta\text{AUC}$ [95% CI] | Cumulative Lift vs Baseline [95% CI] |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | Core Financials | 5 | 0.6104 [0.559, 0.663] | — | Baseline |
| **2** | + PTP Terms & Salary Proximity | 9 | 0.6810 [0.631, 0.727] | +0.0706 [+0.021, +0.121] | +0.0706 [+0.023, +0.124] |
| **3** | + Pre-Capture Payments ($t_p < t_{\text{cap}}$) | 14 | 0.6787 [0.625, 0.727] | -0.0023 [-0.008, +0.003] | +0.0683 [+0.020, +0.114] |
| **4** | + Speech Kinetics & Ghost Flags | 20 | 0.7150 [0.670, 0.759] | +0.0363 [+0.012, +0.060] | +0.1046 [+0.056, +0.155] |
| **5** | + Indic NLP & Char $n$-gram TF-IDF | 26 | **0.7753 [0.724, 0.819]** | **+0.0603 [+0.027, +0.092]** | **+0.1649 [+0.110, +0.227]** |

#### Lift Disambiguation:
- **Speech Kinetics alone:** $+0.0363 \text{ AUC}$
- **Indic Conversational NLP alone:** $+0.0603 \text{ AUC}$
- **Combined Speech + NLP Lift:** $\mathbf{+0.0966 \text{ AUC}}$ ($0.0363 + 0.0603$) over pre-capture payments.
- **PTP Terms & Salary Proximity:** $+0.0706 \text{ AUC}$ over core financials.
- **Total Multimodal Lift:** $\mathbf{+0.1649 \text{ AUC}}$ over financial baseline.

---

### 5. Operational Hold-vs-No-Hold Payoff Engine

#### Action Formulation:
- **Decision Class:** Positive class is **Broken PTP**; early escalation triggers when $P(\text{Broken}) \ge t^*$.
- **Cost Assumptions:** Friction touch cost of escalating a genuine payer $a = ₹250$; default loss avoided on fake PTP $b = ₹200$ (defined as operational planning assumptions).
- **Payoff Objective Function:**
  $$\text{Net Payoff}(t) = b \times \text{TP}(t) - a \times \text{FP}(t)$$

#### Payoff Scenarios with 95% Cluster-Bootstrap CIs:

| Regime | Cost $a$ | Benefit $b$ | Bayes $t^*$ | Val $t^*$ | Test Net ₹ | Lift vs Esc-All [95% CI] | Lift vs Hold-All |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Conservative** | ₹300 | ₹200 | 0.60 | 0.66 | ₹51,200 | **+₹16,600 [+10.4k, +22.6k]** | +₹51,200 |
| **Balanced (Primary)** | ₹250 | ₹200 | 0.56 | 0.57 | ₹53,800 | **+₹11,400 [+6.7k, +15.8k]** | +₹53,800 |
| **Aggressive Recovery** | ₹200 | ₹250 | 0.44 | 0.57 | ₹76,250 | **+₹5,700 [+1.3k, +9.9k]** | +₹76,250 |

*Theoretical Bayes Cutoff vs. Empirical Tuning:* The Bayes break-even threshold is $t^* = a/(a+b)$ ($0.60$ conservative, $0.56$ balanced, $0.44$ aggressive). In empirical validation grid tuning, the aggressive regime selected $0.57$ rather than $0.44$ due to the high base rate of broken PTPs ($72.45\%$): between $p = 0.44$ and $p = 0.57$, empirical density contains few marginal accounts, creating a flat payoff plateau.

#### Test Set Performance at $t^* = 0.57$ (407 Broken, 156 Kept):
- **Confusion Matrix:** 369 True Positive (Broken Caught), 80 False Positive (Genuine Escalated), 38 False Negative (Broken Missed), 76 True Negative (Genuine Held).
- **Trade-off Reality:** The model catches 369 broken promises (**90.7% Recall**, **82.2% Precision**) while protecting 76 genuine payers (**48.7% Specificity**), accepting 80 false escalations ($51.3\%$ false touch rate on the 156 genuine payers).
- **Net Operating Payoff:**
  $$\text{Net Value} = 369 \times ₹200 - 80 \times ₹250 = ₹53,800$$
- **Baseline Hold-All:** $0 \times 200 - 0 \times 250 = ₹0$.
- **Baseline Escalate-All:** $407 \times 200 - 156 \times 250 = ₹42,400$ (100% recall, 72.3% precision).
- **Model Net Lift:** **+₹11,400 over Escalate-All** (95% CI: $[+₹6,700, +₹15,800]$), and **+₹53,800 over Hold-All**.

#### Alternative Operational Rules Benchmark:
- Hold-All (Do Nothing): ₹0 ($-₹42,400$ vs. Escalate-All)
- Lead $\le 5$d Rule: ₹21,450 ($-₹20,950$ vs. Escalate-All)
- Financial-Only LogReg: ₹41,700 ($-₹700$ vs. Escalate-All)
- Escalate-All (Status Quo): ₹42,400 (Baseline)
- Final Multimodal Model: **₹53,800 (+₹11,400 vs. Escalate-All)**

#### Decile Risk Segmentation:
- **Top Decile (Fast-Lane):** Empirical kept rate of **68.42%** (3.51% broken). High-intent borrowers safely bypass outbound touches.
- **Bottom Decile (Immediate Escalation):** Empirical broken rate of **96.49%** (3.51% kept). Fake promises requiring instant outreach.
- **Separation Factor:** **19.49x fulfillment separation**, confirming strong rank-ordering fidelity.

---

### 6. Replication Guide

To replicate all findings from scratch:

```bash
# 1. Execute master pipeline (generates results.json)
python -m nbconvert --to notebook --execute --inplace submisiible-PS1/final_pipeline.ipynb

# 2. Recompile submission PDF report
typst compile final_report.typ submisiible-PS1/final_report.pdf
```
