"""
CACEIS — Human Capital Risk Intelligence
A decision-support system for workforce risk.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import joblib

st.set_page_config(
    page_title="CACEIS HCRI",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={'Get Help': None, 'Report a bug': None, 'About': None},
)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

# ── Design tokens ─────────────────────────────────────────────────────────────
INK     = "#0F1F16"
INK2    = "#3A4A42"
MUTED   = "#5F6F67"
BORDER  = "#E6EBE8"
SURFACE = "#FFFFFF"
BG      = "#F7F9F8"
PRIMARY = "#1E6B42"     # forest green - product palette
ACCENT  = "#0E5C38"
ALERT   = "#A03030"
WARN    = "#A16207"
SUCCESS = "#157347"
CACEIS_BLUE = "#1A3D7C"  # CACEIS brand reference - used only in header mark

# ── Polished CSS ──────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    .stApp {{ background: {BG}; }}
    .block-container {{
        padding-top: 1.4rem; padding-bottom: 4rem;
        max-width: 1240px; padding-left: 2.5rem; padding-right: 2.5rem;
    }}
    #MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}

    .stApp, .stApp p, .stApp div, .stApp span, .stApp li {{
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
        color: {INK};
    }}
    h1, h2, h3, h4, h5 {{
        font-family: 'Georgia', 'Times New Roman', serif;
        color: {INK}; font-weight: 400; letter-spacing: -0.01em;
    }}

    /* ── HEADER with CACEIS wordmark ─────────────────────────────────────── */
    .header-strip {{
        display: flex; justify-content: space-between; align-items: center;
        padding: 14px 0 18px; border-bottom: 1px solid {BORDER};
        margin-bottom: 32px;
    }}
    .header-left {{
        display: flex; align-items: center; gap: 18px;
    }}
    .caceis-mark {{
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 20px; font-weight: 700; letter-spacing: 2px;
        color: {CACEIS_BLUE}; line-height: 1;
        padding-right: 18px; border-right: 1px solid {BORDER};
    }}
    .product-name {{
        font-family: 'Georgia', serif; font-size: 16px;
        color: {INK}; line-height: 1.2;
    }}
    .product-sub {{
        display: block; font-family: 'Inter', sans-serif;
        font-size: 10px; color: {MUTED};
        letter-spacing: 1.6px; text-transform: uppercase;
        margin-top: 3px;
    }}
    .header-meta {{
        font-size: 10.5px; color: {MUTED}; letter-spacing: 1.5px;
        text-transform: uppercase;
    }}

    /* ── FOCAL STATEMENT ─────────────────────────────────────────────────── */
    .focal-eyebrow {{
        font-size: 10.5px; color: {MUTED}; letter-spacing: 2px;
        text-transform: uppercase; font-family: 'Inter', sans-serif;
        margin-bottom: 14px;
    }}
    .focal {{
        font-family: 'Georgia', serif; font-size: 30px; line-height: 1.32;
        color: {INK}; margin: 0 0 18px 0; max-width: 920px;
        font-weight: 400;
    }}
    .focal em {{ font-style: normal; color: {ALERT}; }}
    .focal strong {{
        font-weight: 400; color: {INK};
        border-bottom: 2px solid {PRIMARY}; padding-bottom: 1px;
    }}
    .focal-directive {{
        font-size: 12.5px; color: {MUTED}; font-style: italic;
        margin: 4px 0 0; max-width: 920px;
    }}

    /* ── CARDS ───────────────────────────────────────────────────────────── */
    .quiet-card {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px;
        padding: 18px 22px; height: 100%;
    }}
    .qc-label {{
        font-size: 10px; color: {MUTED}; letter-spacing: 1.6px;
        text-transform: uppercase; margin-bottom: 6px;
    }}
    .qc-value {{
        font-family: 'Georgia', serif; font-size: 22px;
        color: {INK}; line-height: 1.2;
    }}
    .qc-context {{ font-size: 12px; color: {INK2}; margin-top: 6px; }}

    /* ── TABS ────────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0; background: transparent; border-bottom: 1px solid {BORDER};
        padding-bottom: 0;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent; border: none; color: {MUTED};
        font-size: 13px; font-weight: 500; padding: 10px 18px;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: {INK}; border-bottom: 2px solid {PRIMARY};
    }}
    .stTabs [data-baseweb="tab-panel"] {{ padding-top: 28px; }}

    /* ── BUTTONS ─────────────────────────────────────────────────────────── */
    .stButton > button {{
        background: {PRIMARY}; color: white; border: none; border-radius: 4px;
        padding: 10px 20px; font-size: 13px; font-weight: 500;
        text-align: left; height: auto; line-height: 1.4;
    }}
    .stButton > button:hover {{ background: {ACCENT}; color: white; }}
    .stButton > button:focus {{ background: {ACCENT}; color: white; box-shadow: none; }}

    /* Cohort row buttons - distinct from primary buttons */
    .cohort-button .stButton > button {{
        background: white !important; color: {INK} !important;
        border: 1px solid {BORDER} !important;
        text-align: left; padding: 14px 18px;
        font-size: 13px; font-weight: 500; line-height: 1.5;
    }}
    .cohort-button .stButton > button:hover {{
        border-color: {PRIMARY} !important;
        background: #FAFCFB !important;
    }}

    /* ── SECTIONS ────────────────────────────────────────────────────────── */
    .eyebrow {{
        font-size: 10.5px; color: {MUTED}; letter-spacing: 2px;
        text-transform: uppercase; margin-bottom: 4px;
        font-family: 'Inter', sans-serif;
    }}
    .section-title {{
        font-family: 'Georgia', serif; font-size: 22px; color: {INK};
        margin-bottom: 6px; line-height: 1.3;
    }}
    .section-desc {{
        font-size: 13px; color: {INK2}; margin-bottom: 22px;
        max-width: 720px; line-height: 1.6;
    }}
    .divider {{ border-top: 1px solid {BORDER}; margin: 36px 0; }}

    .insight {{
        background: {SURFACE}; border-left: 3px solid {PRIMARY};
        padding: 14px 18px; margin: 20px 0; font-size: 13.5px;
        line-height: 1.6; color: {INK};
    }}
    .insight-label {{
        font-size: 10px; letter-spacing: 1.5px; color: {PRIMARY};
        text-transform: uppercase; font-weight: 600; margin-bottom: 4px;
    }}

    /* Help hint above treemap */
    .nav-hint {{
        background: #F0F7F3; border: 1px solid #D5E8DD;
        padding: 10px 16px; border-radius: 4px;
        font-size: 12px; color: {INK2}; margin-bottom: 18px;
        display: flex; align-items: center; gap: 12px;
    }}
    .nav-hint-icon {{
        flex-shrink: 0; color: {PRIMARY}; font-weight: 600;
        font-size: 13px;
    }}

    [data-testid="stSelectbox"] label,
    [data-testid="stNumberInput"] label,
    [data-testid="stSlider"] label {{
        font-size: 11px !important; color: {MUTED} !important;
        text-transform: uppercase; letter-spacing: 1px;
    }}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if 'selected_cohort_idx' not in st.session_state:
    st.session_state.selected_cohort_idx = None
if 'scenario' not in st.session_state:
    st.session_state.scenario = 'targeted'

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    cohorts = pd.read_pickle(DATA_DIR / 'cohort_risk.pkl')
    country = pd.read_pickle(DATA_DIR / 'country_risk.pkl')
    concentration = pd.read_pickle(DATA_DIR / 'concentration.pkl')
    return cohorts, country, concentration

@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_DIR / 'attrition_model.pkl')

cohorts, country, concentration = load_data()
bundle = load_model()
fi = pd.DataFrame(bundle['feature_importance'])

n_critical = (cohorts['risk_band'] == 'Critical').sum()
n_high = (cohorts['risk_band'] == 'High').sum()
n_at_risk_cohorts = n_critical + n_high
employees_at_risk = int(cohorts[cohorts['risk_band'].isin(['Critical','High'])]['n'].sum())

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="header-strip">
  <div class="header-left">
    <div class="caceis-mark">CACEIS</div>
    <div>
      <div class="product-name">Human Capital Risk Intelligence</div>
      <div class="product-sub">Decision-support system</div>
    </div>
  </div>
  <div class="header-meta">May 2026</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Today's brief", "Cohort intelligence", "Scenario lab",
    "Knowledge map", "Methodology"
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — TODAY'S BRIEF
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f"""
    <div class="focal-eyebrow">Status — May 2026</div>
    <div class="focal">
      Today, <em>{employees_at_risk:,} employees</em> are concentrated in
      <strong>{n_at_risk_cohorts} cohorts</strong> where the model predicts
      elevated departure risk over the next quarter. Annual unbooked liability:
      <em>35–50 M€</em>.
    </div>
    <div class="focal-directive">
      → To explore these cohorts, open the <em style="font-style: normal; color: {INK}; font-weight: 500;">Cohort intelligence</em> tab above.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height: 36px;"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(f"""
        <div class="quiet-card">
          <div class="qc-label">Productivity</div>
          <div class="qc-value">325 k€</div>
          <div class="qc-context">NBI per FTE in 2025 · +4% vs 2022</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="quiet-card">
          <div class="qc-label">Workforce flow</div>
          <div class="qc-value">22.9%</div>
          <div class="qc-context">Annual exit rate · ~1,500 departures / year</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="quiet-card">
          <div class="qc-label">Investment in capital</div>
          <div class="qc-value">0.65%</div>
          <div class="qc-context">Training / HR cost · –37% from 2023</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height: 48px;"></div>', unsafe_allow_html=True)

    cc1, cc2 = st.columns([3, 2], gap="large")
    with cc1:
        st.markdown(f'<div class="eyebrow">NBI per FTE — 2022 to 2025</div>',
                    unsafe_allow_html=True)
        years = [2022, 2023, 2024, 2025]
        values = [313, 262, 315, 325]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years, y=values, mode='lines+markers',
            line=dict(color=PRIMARY, width=2.5, shape='spline'),
            marker=dict(size=8, color=PRIMARY, line=dict(width=2, color='white')),
            hovertemplate='<b>%{x}</b><br>%{y} k€<extra></extra>',
        ))
        fig.add_annotation(x=2023, y=262, text="Integration year",
                          showarrow=True, arrowhead=0, arrowcolor=MUTED,
                          ax=0, ay=42, font=dict(size=10, color=MUTED))
        fig.update_layout(
            height=240, margin=dict(l=0, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, tickfont=dict(color=MUTED, size=11),
                       linecolor=BORDER, dtick=1),
            yaxis=dict(gridcolor=BORDER, gridwidth=0.5,
                       tickfont=dict(color=MUTED, size=11),
                       ticksuffix=" k€", title=None),
            font=dict(family='Inter, sans-serif'),
            showlegend=False, hovermode='x unified',
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with cc2:
        st.markdown(f"""
        <div style="padding-top: 4px;">
          <div class="eyebrow">A narrow recovery story</div>
          <div style="font-size: 13px; color: {INK2}; line-height: 1.7; margin-top: 10px;">
            The 2023 dip cost ~326 M€ in unrealised NBI at constant productivity.
            The recovery is real, but the workforce cost ratio (37% of NBI) has not
            moved across four years. Growth is carried by revenue expansion,
            not workforce efficiency.
          </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — COHORT INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="eyebrow">Workforce segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Where attrition risk is concentrated</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="section-desc">
      Each row is a workforce cohort. Risk is the model's predicted departure probability
      over the next 3 months, averaged within the cohort. Click any cohort to see the
      diagnostic and recommended action.
    </div>
    """, unsafe_allow_html=True)

    fcol1, fcol2, fcol3, fcol4 = st.columns([1.2, 1.5, 1, 0.8])
    with fcol1:
        countries_list = ['All countries'] + sorted(cohorts['COUNTRY_GROUP_LABEL_EN'].unique())
        sel_country = st.selectbox("Country", countries_list, label_visibility="collapsed")
    with fcol2:
        contracts_list = ['All contract types'] + sorted(cohorts['CONTRACT_GROUP_LABEL_EN'].unique())
        sel_contract = st.selectbox("Contract", contracts_list, label_visibility="collapsed")
    with fcol3:
        sel_band = st.selectbox("Risk", ['All risk levels','Critical','High','Medium','Low'],
                                label_visibility="collapsed")
    with fcol4:
        min_size = st.number_input("Min size", 5, 200, 10, 5, label_visibility="collapsed")

    filtered = cohorts.copy()
    if sel_country != 'All countries':
        filtered = filtered[filtered['COUNTRY_GROUP_LABEL_EN'] == sel_country]
    if sel_contract != 'All contract types':
        filtered = filtered[filtered['CONTRACT_GROUP_LABEL_EN'] == sel_contract]
    if sel_band != 'All risk levels':
        filtered = filtered[filtered['risk_band'] == sel_band]
    filtered = filtered[filtered['n'] >= min_size].sort_values('risk_mean', ascending=False).reset_index(drop=True)

    st.markdown(f"<div style='font-size: 12px; color: {MUTED}; margin: 18px 0 12px;'>"
                f"{len(filtered)} cohorts · {int(filtered['n'].sum()):,} employees</div>",
                unsafe_allow_html=True)

    if len(filtered) == 0:
        st.info("No cohorts match these filters.")
    else:
        # Cohort cards — white background, hover green border
        st.markdown('<div class="cohort-button">', unsafe_allow_html=True)
        col_left, col_right = st.columns(2, gap="medium")
        for i in range(min(12, len(filtered))):
            row = filtered.iloc[i]
            target = col_left if i % 2 == 0 else col_right
            risk_pct = row['risk_mean'] * 100
            band = str(row['risk_band'])
            band_color = {'Critical': ALERT, 'High': WARN, 'Medium': '#3730A3', 'Low': SUCCESS}.get(band, MUTED)
            label = (f"{row['COUNTRY_GROUP_LABEL_EN']} · {row['CONTRACT_GROUP_LABEL_EN']} · {row['Age range']}\n"
                     f"{int(row['n'])} people · {risk_pct:.0f}% risk · {band}")
            with target:
                if st.button(label, key=f"coh_{i}_{row['COUNTRY_GROUP_LABEL_EN']}_{row['Age range']}",
                            use_container_width=True):
                    st.session_state.selected_cohort_idx = i
                    st.session_state.selected_cohort_data = row.to_dict()
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.selected_cohort_idx is not None and 'selected_cohort_data' in st.session_state:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            row = st.session_state.selected_cohort_data
            risk_pct = row['risk_mean'] * 100

            st.markdown(f"""
            <div class="eyebrow">Diagnostic</div>
            <div class="section-title">
              {row['COUNTRY_GROUP_LABEL_EN']} · {row['CONTRACT_GROUP_LABEL_EN']} · age {row['Age range']}
            </div>
            """, unsafe_allow_html=True)

            d1, d2, d3 = st.columns(3, gap="medium")
            with d1:
                st.markdown(f"""
                <div class="quiet-card">
                  <div class="qc-label">Predicted attrition</div>
                  <div class="qc-value" style="color: {ALERT};">{risk_pct:.0f}%</div>
                  <div class="qc-context">over next 3 months</div>
                </div>
                """, unsafe_allow_html=True)
            with d2:
                st.markdown(f"""
                <div class="quiet-card">
                  <div class="qc-label">Cohort size</div>
                  <div class="qc-value">{int(row['n'])}</div>
                  <div class="qc-context">employees in this segment</div>
                </div>
                """, unsafe_allow_html=True)
            with d3:
                tenure_y = row['tenure_avg'] / 12
                st.markdown(f"""
                <div class="quiet-card">
                  <div class="qc-label">Average tenure</div>
                  <div class="qc-value">{tenure_y:.1f} years</div>
                  <div class="qc-context">at CACEIS</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="eyebrow">Why the model flags this cohort</div>',
                        unsafe_allow_html=True)

            drivers = []
            contract = str(row['CONTRACT_GROUP_LABEL_EN'])
            if 'Internship' in contract:
                drivers.append(("Contract type", "Internships are time-bound. High exit at end-of-term is structural — but it compounds when conversion to permanent contract is low."))
            elif 'Temporary' in contract:
                drivers.append(("Contract type", "Temporary contracts carry structural attrition. The model does not separate end-of-contract from preventable losses — that distinction requires data we don't have."))
            elif 'Apprentice' in contract:
                drivers.append(("Contract type", "Apprenticeships end at fixed terms. Exit at end-of-program is structural."))
            else:
                drivers.append(("Contract type", "Permanent contract attrition is voluntary by definition — typically signals retention issue, market pull, or career-stage decision."))

            if row['tenure_avg'] < 12:
                drivers.append(("Tenure", "Short tenure is one of the strongest predictors. The first 12 months carry the highest departure risk in the data."))
            elif row['tenure_avg'] < 36:
                drivers.append(("Tenure", "Medium tenure (1–3 years) signals career-stage decisions: promotion expectations, market opportunities, internal mobility."))
            else:
                drivers.append(("Tenure", "High tenure means accumulated knowledge. When these employees leave, the cost of replacement is disproportionate."))

            ar = str(row['Age range'])
            if '20-29' in ar:
                drivers.append(("Age range", "Early-career employees have higher market mobility and are most exposed to external offers."))
            elif '50-59' in ar or '60' in ar:
                drivers.append(("Age range", "Late-career signals — potential pre-retirement decisions or planned exits."))

            for label, text in drivers:
                st.markdown(f"""
                <div style="background: white; border-left: 3px solid {PRIMARY};
                           padding: 12px 16px; margin-bottom: 8px; border-radius: 0 4px 4px 0;">
                  <div style="font-size: 10.5px; letter-spacing: 1.5px; color: {PRIMARY};
                              text-transform: uppercase; font-weight: 600; margin-bottom: 4px;">{label}</div>
                  <div style="font-size: 13px; color: {INK2}; line-height: 1.55;">{text}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="eyebrow">Recommended action</div>', unsafe_allow_html=True)

            if 'Internship' in contract:
                action = "Convert top 30% of interns to permanent contracts before end of program. Run structured exit interviews on the rest to learn what would have made conversion attractive."
                horizon = "30 days"
            elif 'Temporary' in contract:
                action = "Review temporary-to-permanent conversion policy at the entity level. The model cannot distinguish structural end-of-contract from preventable losses without contract-end-reason data."
                horizon = "60 days"
            elif row['tenure_avg'] < 12:
                action = "Reinforced onboarding programme: assigned mentor, structured 30/60/90 check-ins, clear 12-month career path conversation."
                horizon = "30 days"
            else:
                action = "Manager-level retention review: career path clarification, market salary benchmark, growth opportunity discussion. Targeted at this cohort's specific signals."
                horizon = "60 days"

            st.markdown(f"""
            <div class="insight">
              <div class="insight-label">{horizon} action</div>
              {action}
            </div>
            """, unsafe_allow_html=True)

            replacement_cost_kEUR = 45
            current_loss = int(row['n']) * row['risk_mean'] * replacement_cost_kEUR
            target_risk = row['risk_mean'] * 0.7
            target_loss = int(row['n']) * target_risk * replacement_cost_kEUR
            saved = current_loss - target_loss

            st.markdown(f"""
            <div style="display: flex; gap: 32px; margin-top: 16px; padding: 16px 0;
                        border-top: 1px solid {BORDER};">
              <div>
                <div style="font-size: 10px; letter-spacing: 1.5px; color: {MUTED};
                            text-transform: uppercase;">Current annual exposure</div>
                <div style="font-family: Georgia, serif; font-size: 22px; color: {INK};">
                  {current_loss:.0f} k€
                </div>
              </div>
              <div style="border-left: 1px solid {BORDER}; padding-left: 32px;">
                <div style="font-size: 10px; letter-spacing: 1.5px; color: {MUTED};
                            text-transform: uppercase;">Estimated saving (30% risk reduction)</div>
                <div style="font-family: Georgia, serif; font-size: 22px; color: {SUCCESS};">
                  {saved:.0f} k€/year
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — SCENARIO LAB
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="eyebrow">Decision support</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Scenario lab — three calibrated paths</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="section-desc">
      Three calibrated scenarios for the next three years. Each models attrition trajectory,
      training investment, and replacement cost. Switch between them or adjust assumptions
      below to compare outcomes.
    </div>
    """, unsafe_allow_html=True)

    scenarios_def = {
        'status_quo': {
            'title': 'Status quo',
            'desc': 'No intervention. Attrition stays at 22.9%, training stays at 0.65% of HR cost.',
            'attrition_path': [22.9, 23.1, 23.3],
            'training_pct': 0.65,
            'replacement_cost': 45,
        },
        'targeted': {
            'title': 'Targeted action',
            'desc': 'Retention focus on top 10 critical cohorts. Attrition reduced to 19% in 3 years. Training back to 1.0% of HR cost.',
            'attrition_path': [22.9, 20.5, 19.0],
            'training_pct': 1.0,
            'replacement_cost': 45,
        },
        'aggressive': {
            'title': 'Aggressive investment',
            'desc': 'Group-wide retention + EAE recalibration + market salary alignment. Attrition to 16% in 3 years.',
            'attrition_path': [22.9, 19.5, 16.0],
            'training_pct': 1.4,
            'replacement_cost': 45,
        },
    }

    sc1, sc2, sc3 = st.columns(3, gap="medium")
    for col, (key, scen) in zip([sc1, sc2, sc3], scenarios_def.items()):
        with col:
            is_active = st.session_state.scenario == key
            if st.button(scen['title'], key=f"sc_{key}", use_container_width=True):
                st.session_state.scenario = key
                st.rerun()
            border = PRIMARY if is_active else BORDER
            bg_card = '#F0F7F3' if is_active else 'white'
            st.markdown(f"""
            <div style="background: {bg_card}; border: 1px solid {border}; border-radius: 4px;
                        padding: 12px 14px; font-size: 12px; color: {INK2}; line-height: 1.55;
                        margin-top: 8px; min-height: 90px;">
              {scen['desc']}
            </div>
            """, unsafe_allow_html=True)

    scen = scenarios_def[st.session_state.scenario]

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Adjust the assumptions</div>', unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3, gap="medium")
    with a1:
        end_attrition = st.slider("Year 3 attrition target (%)", 12.0, 25.0,
                                  float(scen['attrition_path'][-1]), 0.5)
    with a2:
        training_pct = st.slider("Training spend (% of HR cost)", 0.5, 2.0,
                                 float(scen['training_pct']), 0.05)
    with a3:
        replacement_cost = st.slider("Replacement cost (k€/person)", 30, 80,
                                     int(scen['replacement_cost']), 5)

    fte = 6453
    hr_cost_m = 770
    years_proj = [2025, 2026, 2027, 2028]
    attr_path = [22.9,
                 22.9 + (end_attrition - 22.9) * 0.4,
                 22.9 + (end_attrition - 22.9) * 0.75,
                 end_attrition]
    attr_cost = [fte * (a/100) * replacement_cost / 1000 for a in attr_path]
    training_cost_m = hr_cost_m * training_pct / 100
    current_training_m = hr_cost_m * 0.65 / 100
    training_delta = training_cost_m - current_training_m

    cumulative = []
    cum = 0
    baseline_cost = fte * 0.229 * replacement_cost / 1000
    for ac in attr_cost:
        net = (baseline_cost - ac) - training_delta
        cum += net
        cumulative.append(cum)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    p1, p2 = st.columns([3, 2], gap="large")
    with p1:
        st.markdown('<div class="eyebrow">Net cumulative impact — 3 years</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">{cumulative[-1]:+.0f} M€ over 3 years</div>',
                    unsafe_allow_html=True)

        line_color = PRIMARY if cumulative[-1] > 0 else ALERT
        fill_color = 'rgba(30,107,66,0.10)' if cumulative[-1] > 0 else 'rgba(160,48,48,0.10)'

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years_proj, y=cumulative, mode='lines+markers',
            line=dict(color=line_color, width=3, shape='spline'),
            marker=dict(size=10, line=dict(width=2, color='white')),
            fill='tozeroy', fillcolor=fill_color,
            hovertemplate='<b>%{x}</b><br>Cumulative: %{y:+.1f} M€<extra></extra>',
        ))
        fig.add_hline(y=0, line_color=BORDER, line_width=1)
        fig.update_layout(
            height=320, margin=dict(l=0, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, tickfont=dict(color=MUTED, size=11),
                       linecolor=BORDER, dtick=1),
            yaxis=dict(gridcolor=BORDER, gridwidth=0.5,
                       tickfont=dict(color=MUTED, size=11),
                       ticksuffix=" M€", title=None, zeroline=False),
            font=dict(family='Inter, sans-serif'),
            showlegend=False, hovermode='x unified',
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with p2:
        st.markdown('<div class="eyebrow">Attrition trajectory</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=years_proj, y=attr_path, mode='lines+markers',
            line=dict(color=ALERT, width=2),
            marker=dict(size=7, line=dict(width=1.5, color='white')),
            hovertemplate='<b>%{x}</b><br>%{y:.1f}%<extra></extra>',
        ))
        fig2.update_layout(
            height=320, margin=dict(l=0, r=20, t=44, b=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, tickfont=dict(color=MUTED, size=11),
                       linecolor=BORDER, dtick=1),
            yaxis=dict(gridcolor=BORDER, gridwidth=0.5,
                       tickfont=dict(color=MUTED, size=11),
                       ticksuffix="%", title=None, range=[10, 26]),
            font=dict(family='Inter, sans-serif'),
            showlegend=False, hovermode='x unified',
        )
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Sensitivity to assumptions</div>', unsafe_allow_html=True)
    st.markdown(f"<div style='font-size: 13px; color: {INK2}; margin-bottom: 18px;'>"
                "How much does the 3-year outcome change if a key assumption is wrong by 20%?</div>",
                unsafe_allow_html=True)

    attr_diff = (0.229 - end_attrition/100)
    sens_rc = abs(fte * attr_diff * 3 * replacement_cost * 0.2 / 1000)
    sens_training = abs(training_delta) * 0.2 * 3
    sens_attr = fte * 0.02 * replacement_cost / 1000 * 3
    sens_data = [
        ('Replacement cost ±20%', sens_rc),
        ('Training investment ±20%', sens_training),
        ('Year 3 target ±2pp', sens_attr),
    ]

    fig3 = go.Figure(go.Bar(
        y=[s[0] for s in sens_data],
        x=[s[1] for s in sens_data],
        orientation='h',
        marker_color=MUTED,
        text=[f"±{s[1]:.1f} M€" for s in sens_data],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Impact: ±%{x:.1f} M€<extra></extra>',
    ))
    fig3.update_layout(
        height=180, margin=dict(l=0, r=80, t=10, b=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, autorange='reversed',
                   tickfont=dict(color=INK, size=12)),
        font=dict(family='Inter, sans-serif'),
        showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — KNOWLEDGE MAP
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="eyebrow">Succession risk</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Where knowledge is concentrated and at risk</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="section-desc">
      Each rectangle is a workforce cohort. Size shows headcount, colour shows concentration risk —
      a function of attrition risk and accumulated tenure. Larger and darker means more knowledge
      is exposed if those people leave.
    </div>
    """, unsafe_allow_html=True)

    # Navigation hint — fixes the "took 5 minutes to figure out how to go back"
    st.markdown(f"""
    <div class="nav-hint">
      <span class="nav-hint-icon">↳</span>
      <span><strong>Click any zone</strong> to zoom in.
      <strong>Click the breadcrumb path</strong> at the top of the chart (CACEIS → country → contract)
      to navigate back up.</span>
    </div>
    """, unsafe_allow_html=True)

    treemap_data = cohorts.copy()
    treemap_data = treemap_data[treemap_data['n'] >= 10]
    treemap_data['concentration_score'] = (
        treemap_data['risk_mean'] * (treemap_data['tenure_avg'] / 12).clip(upper=8)
    )

    fig = px.treemap(
        treemap_data,
        path=[px.Constant("All workforce"), 'COUNTRY_GROUP_LABEL_EN',
              'CONTRACT_GROUP_LABEL_EN', 'Age range'],
        values='n',
        color='concentration_score',
        color_continuous_scale=[[0, '#DCFCE7'], [0.3, '#FEF3C7'],
                                 [0.6, '#FED7AA'], [1, '#7F1D1D']],
        custom_data=['risk_mean', 'tenure_avg', 'concentration_score'],
    )
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Headcount: %{value}<br>Risk: %{customdata[0]:.0%}<br>Tenure: %{customdata[1]:.1f} months<br>Concentration: %{customdata[2]:.2f}<extra></extra>',
        marker=dict(line=dict(color='white', width=2)),
        textfont=dict(family='Inter, sans-serif', size=11),
        root_color="white",
    )
    fig.update_layout(
        height=540, margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(title="Concentration", thickness=10, len=0.7,
                                tickfont=dict(size=10, color=MUTED)),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif'),
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown(f"""
    <div class="insight">
      <div class="insight-label">Read this</div>
      The largest darker zones are succession blind spots — cohorts where many people hold
      meaningful tenure AND show elevated departure risk. These are the priorities for
      knowledge documentation, cross-training, and structured succession planning.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="eyebrow">For technical review</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How this was built</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="section-desc">
      Data, model, governance, and reproducibility. Source code is in the Git repository.
    </div>
    """, unsafe_allow_html=True)

    m1, m2 = st.columns(2, gap="large")
    with m1:
        st.markdown(f"""
        <div class="qc-label">DATA FOUNDATION</div>
        <div style="font-size: 13px; color: {INK2}; line-height: 1.7; margin: 8px 0 24px;">
          11 source files · 275,609 HR records · 18 countries · January 2023 – December 2025.
          Four distinct anonymisation systems prevent direct individual joins. Cross-source
          analysis stays at the aggregate level (entity, country, period). The 1,906 confirmed
          Matricule = Employee Code matches enable individual-level training analysis.
        </div>

        <div class="qc-label">MODEL SPECIFICATION</div>
        <div style="font-size: 13px; color: {INK2}; line-height: 1.7; margin: 8px 0 24px;">
          XGBoost classifier · target = exit within 3 months (panel-derived from monthly
          snapshots) · 12 features (contract, tenure, age, country, entity, period).
          Train: 201,199 rows pre-April 2025. Test: 49,758 rows from April 2025 onwards.
          Temporal split — no leakage. Class imbalance handled via scale_pos_weight = 56.2.
        </div>

        <div class="qc-label">PERFORMANCE</div>
        <div style="font-size: 13px; color: {INK2}; line-height: 1.7; margin: 8px 0 24px;">
          AUC test = 0.836 · Average precision = 0.297 (vs 0.018 random baseline) ·
          Recall on minority class = 73.6% at threshold 0.5. Top drivers: contract type
          (23%), tenure (15%), age range (11%).
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="qc-label">GOVERNANCE</div>
        <div style="font-size: 13px; color: {INK2}; line-height: 1.7; margin: 8px 0 24px;">
          GDPR Art. 22 prevents automated individual decisions with significant effect.
          AI Act Annex III §4 classifies workforce decision systems as high-risk —
          our descriptive dashboard does not trigger this, the predictive model does.
          Cohort-level outputs only. Purpose firewall enforced. CSE consultation required
          before any deployment.
        </div>

        <div class="qc-label">DELIBERATELY EXCLUDED</div>
        <div style="font-size: 13px; color: {INK2}; line-height: 1.7; margin: 8px 0 24px;">
          Individual-level prediction. Performance ranking models (the EAE data is too
          compressed). Causal claims about which features drive attrition (the model
          predicts, it does not explain causation). Any output that would require
          identifying named individuals.
        </div>

        <div class="qc-label">REPRODUCIBILITY</div>
        <div style="font-size: 13px; color: {INK2}; line-height: 1.7; margin: 8px 0 24px;">
          <code style="background: white; padding: 2px 6px; border-radius: 3px; font-size: 12px;
                       border: 1px solid {BORDER};">python src/train_model.py</code>
          rebuilds the model from raw data in ≈ 2 minutes.
          <code style="background: white; padding: 2px 6px; border-radius: 3px; font-size: 12px;
                       border: 1px solid {BORDER};">python src/due_diligence.py</code>
          verifies every figure cited.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="qc-label">FEATURE IMPORTANCE</div>', unsafe_allow_html=True)
    st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

    fig_fi = go.Figure(go.Bar(
        y=fi.head(8)['feature'].iloc[::-1],
        x=fi.head(8)['importance'].iloc[::-1] * 100,
        orientation='h',
        marker_color=PRIMARY,
        text=[f"{v*100:.1f}%" for v in fi.head(8)['importance'].iloc[::-1]],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>%{x:.1f}%<extra></extra>',
    ))
    fig_fi.update_layout(
        height=280, margin=dict(l=0, r=80, t=10, b=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, tickfont=dict(color=INK, size=12)),
        font=dict(family='Inter, sans-serif'),
        showlegend=False,
    )
    st.plotly_chart(fig_fi, use_container_width=True, config={'displayModeBar': False})
