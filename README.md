# CACEIS — Human Capital Risk Intelligence Cockpit

Final deliverable for the Albert School × CACEIS Human Capital Valuation challenge.

A decision-support system that quantifies, predicts, and governs human capital risk at CACEIS with the same rigour applied to financial risk.

---

## What this delivers

- **Data pipeline** integrating 11 source files, 275,609 records across 18 countries (2023–2025), with documented identifier reconciliation
- **5 KPIs** computed from real CACEIS data — productivity, attrition, training, performance, absenteeism
- **Functional attrition risk model** — XGBoost classifier, AUC test = 0.836, output at cohort level only
- **Knowledge Concentration Index** — surfaces succession blind spots where high attrition risk meets accumulated tenure
- **Financial simulator** — interactive what-if scenarios (attrition target, replacement cost, training investment) with live € impact
- **Governance framework** — GDPR Art. 22, AI Act high-risk obligations, design choices documented
- **Recommendation engine** — actionable rules per critical cohort

## Quick start

```bash
git clone <repo-url>
cd caceis_repo
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run src/cockpit.py
```

The cockpit will open at http://localhost:8501.

## Repository structure

```
caceis_repo/
├── README.md
├── requirements.txt
├── src/
│   ├── cockpit.py              # Streamlit app — main entry point
│   ├── train_model.py          # Reproduces the XGBoost attrition model
│   └── due_diligence.py        # Verifies every figure cited in the deck
├── notebooks/
│   └── CACEIS_HCV_Deliverable.ipynb   # Full analytical workflow
├── models/
│   └── attrition_model.pkl     # Trained XGBoost model + encoders + metrics
├── data/
│   ├── cohort_risk.pkl         # Pre-computed cohort risk scores
│   ├── country_risk.pkl
│   └── concentration.pkl       # Knowledge concentration index
└── docs/
    ├── deck_final.pdf          # Final 12-slide presentation
    └── speaking_notes.pdf      # Oral presentation script
```

## Reproducing the results

The full pipeline is reproducible from raw data:

```bash
# 1. Place CACEIS source data in data/raw/ (or set CACEIS_DATA_DIR)
# 2. Train the model (≈ 2 min on a laptop)
python src/train_model.py
# 3. Verify every figure cited in the deck
python src/due_diligence.py
# 4. Launch the cockpit
streamlit run src/cockpit.py
```

## Model specifications

| Item | Value |
|---|---|
| Algorithm | XGBoost classifier |
| Target | Exit within 3 months (panel-derived from monthly snapshots) |
| Training set | 201,199 rows (Jan 2023 – Mar 2025) |
| Test set | 49,758 rows (Apr 2025 – Sep 2025) — temporal split |
| Class balance | scale_pos_weight = 56.2 (1.8% exit rate) |
| AUC test | 0.836 |
| Avg precision | 0.297 |
| Top features | Contract type (23%), tenure group (15%), age range (11%) |

## Governance constraints

| Concern | Constraint applied |
|---|---|
| GDPR Art. 22 | No automated decision at individual level — output is cohort-level only |
| GDPR Art. 6 | Legal basis = legitimate interest (Art. 6(1)(f)) with balancing test |
| AI Act Annex III §4 | Workforce decisions classified high-risk — Art. 14 oversight applied |
| French Labor Code | CSE consultation required before deployment (L.2312-38) |
| Purpose limitation | Retention model cannot be repurposed for termination decisions |

## Data sources

| File | Rows | Use |
|---|---|---|
| Data.xlsx (HR Master) | 275,609 | Workforce panel, attrition target |
| AlbertSchool_CACEIS_PL-FTE_22-25_Sent.xlsx | — | NBI, FTE, GOI, training cost |
| Training_Records_Unnamed.xlsx | 14,943 | Training participation |
| Quick_Review_Unnamed.xlsx | — | Post-training satisfaction |
| 20250218 Stats CACEIS EAE EP | 5,766 | Performance evaluations |
| Absentéisme files (2024 + 2025) | 261,702 | Absence by cause |

## Identifier reconciliation

Four anonymisation systems prevent direct individual joins across all sources:

| Key | Sources | Cross-system join |
|---|---|---|
| ID Employee | HR Master | Isolated |
| Employee Code | Training, Absenteeism | = Matricule (1,906 confirmed matches) |
| Matricule | Quick / Cold Review | = Employee Code (used for individual training analysis) |
| IUG | EAE, Notes eval | Self-consistent year-on-year (2,347 matches) |

Cross-source analysis at HR Master level remains aggregate (entity, country, period). Individual-level analysis is limited to the 1,906 confirmed bridges.

## Key findings

- **22.9%** annual exit rate (panel-derived, rolling 12M, 2025)
- **35–50 M€/year** estimated unbooked replacement cost — absent from every P&L
- **~300 M€** estimated unrealised NBI in 2023 alone (integration year, vs 2022 productivity baseline)
- **94%** of EAE-evaluated employees rated 3 or 4 — distribution too compressed to support succession decisions
- **0.65%** of HR cost spent on training in 2025 (down 37% from 2023)

## License

This work was produced as part of the Albert School curriculum.
Data is proprietary to CACEIS — not redistributed in this repository.

## Authors

ScoreAI team — Albert School Bachelor in Business & Data, 2026.
