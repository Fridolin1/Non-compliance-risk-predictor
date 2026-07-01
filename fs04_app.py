"""
FS-04 Non-Compliance Risk Predictor — Streamlit UI Prototype
================================================================

This is a UI prototype for the FS-04 project. Predictions currently
use a simple RULE-BASED placeholder (see `predict_risk()` below) so the
interface can be demoed and tested before the trained ML model is wired in.

HOW TO CONNECT YOUR REAL MODEL
-------------------------------
1. Save your trained model and scaler (e.g. with joblib):
       joblib.dump(model, "fs04_model.pkl")
       joblib.dump(scaler, "fs04_scaler.pkl")

2. At the top of this file, load them once:
       import joblib
       model = joblib.load("fs04_model.pkl")
       scaler = joblib.load("fs04_scaler.pkl")

3. Replace the body of `predict_risk()` with your real prediction logic
   (a template is included in its docstring).

4. Replace `get_feature_importance()` with your model's actual
   feature_importances_ / coefficients.

Run with:
    streamlit run fs04_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="FS-04 | Non-Compliance Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLING (Navy / Teal theme)
# ============================================================
st.markdown(
    """
    <style>
        .stApp { background-color: #f4f6f9; }

        [data-testid="stSidebar"] { background-color: #146740; }
        [data-testid="stSidebar"] * { color: #FFFFFF !important; }

        h1, h2, h3 { color: #0f1f3d; }

        .section-title {
            color: #0f1f3d;
            border-bottom: 2px solid #0d9488;
            padding-bottom: 0.4rem;
            margin-top: 1.4rem;
        }

        .metric-card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 1.1rem 1.4rem;
            box-shadow: 0 1px 4px rgba(15, 31, 61, 0.08);
            border-left: 5px solid #0d9488;
            margin-bottom: 0.6rem;
        }
        .metric-label {
            font-size: 0.8rem;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            font-size: 1.9rem;
            font-weight: 700;
            color: #0f1f3d;
        }

        .risk-badge {
            padding: 0.3rem 0.9rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .demo-banner {
            background-color: #1e3a5f;
            color: #fcd34d;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.85rem;
            margin-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# REGULATORY THRESHOLDS (Bank of Tanzania benchmarks)
# ============================================================
THRESHOLDS = {
    "car_min": 12.0,        # Minimum Capital Adequacy Ratio (%)
    "npl_max": 5.0,         # Maximum acceptable NPL ratio (%)
    "liquidity_min": 20.0,  # Minimum liquidity ratio (%)
}

RISK_COLORS = {"High": "#dc2626", "Medium": "#d97706", "Low": "#059669"}


# ============================================================
MODELS_DIR=Path(__file__).parent/ 'models'
@st.cache_resource
def load_models():
    dt_model=joblib.load(MODELS_DIR/'Fs04_Dtree_model.pkl')
    feature_order=joblib.load(MODELS_DIR/'Fs04_feature_order.pkl')
    return dt_model,feature_order
dt_model,feature_order=load_models()
# ============================================================
def predict_risk(car, npl, liquidity, late_reports, audit_findings):
    
    row=pd.DataFrame([[car, npl, liquidity, late_reports, audit_findings]],columns=feature_order)
    proba=dt_model.predict_proba(row)[0]

    p_nonCompliant=proba[0]
    if p_nonCompliant>=0.5:
        risk_level='High'
    elif p_nonCompliant>=0.15:
        risk_level='Medium'
    else:
        risk_level='Low'
    confidence=max(proba)
    return risk_level,confidence

def get_feature_importance():
    fi=pd.DataFrame({'Feature':['NPL Ratio','Capital Adequency Ratio','Liquidity Ratio','Audit Findings','Late Reports'],
                     'Importance':dt_model.feature_importances_})
    return fi.sort_values('Importance',ascending=True)


# ============================================================
# HELPERS
# ============================================================
def style_risk(val):
    colors = {
        "High": "background-color: #fee2e2; color: #b91c1c; font-weight: 600;",
        "Medium": "background-color: #fef3c7; color: #b45309; font-weight: 600;",
        "Low": "background-color: #d1fae5; color: #047857; font-weight: 600;",
    }
    return colors.get(val, "")


def style_table(data, risk_col="Risk Level"):
    if risk_col not in data.columns:
        return data.style
    try:
        return data.style.map(style_risk, subset=[risk_col])
    except AttributeError:  # older pandas
        return data.style.applymap(style_risk, subset=[risk_col])


def metric_card(col, label, value, color="#0d9488", value_size="1.9rem"):
    col.markdown(
        f"""
        <div class="metric-card" style="border-left-color:{color};">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="font-size:{value_size}; color:{color if color != '#0d9488' else '#0f1f3d'};">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SAMPLE DATA (for Dashboard / Institution Detail / Reports)
# ============================================================
@st.cache_data
def load_sample_data(n=150):
    rng = np.random.default_rng(42)

    bank_names = ["NMB", "CRDB", "Stanbic", "Equity", "Exim", "DTB", "Akiba", "Azania", "Mwanga", "Amana"]
    sacco_names = ["Tujenge", "Imani", "Wekeza", "Umoja", "Maendeleo", "Upendo", "Jitegemee", "Faraja", "Neema", "Baraka"]
    mfi_names = ["Bayport", "Letshego", "Yetu", "Pesa Fasta", "VisionFund", "BRAC", "FINCA", "Platinum", "Mkombozi", "Watu"]

    types, names = [], []
    for i in range(n):
        t = rng.choice(["Bank", "SACCO", "MFI"], p=[0.25, 0.40, 0.35])
        types.append(t)
        if t == "Bank":
            base = rng.choice(bank_names)
            names.append(f"{base} Bank #{i+1:03d}")
        elif t == "SACCO":
            base = rng.choice(sacco_names)
            names.append(f"{base} SACCO #{i+1:03d}")
        else:
            base = rng.choice(mfi_names)
            names.append(f"{base} MFI #{i+1:03d}")

    car = np.clip(rng.normal(15, 5, n), 5, 32)
    npl = np.clip(rng.normal(11, 7, n), 0.5, 33.5)
    liquidity = np.clip(rng.normal(22, 7, n), 5, 45)
    late_reports = rng.poisson(2, n)
    audit_findings = np.clip(rng.normal(5, 2.6, n), 0, 10)

    risks, confs = [], []
    for c, n_, l, lr, a in zip(car, npl, liquidity, late_reports, audit_findings):
        r, conf = predict_risk(c, n_, l, lr, a)
        risks.append(r)
        confs.append(conf)

    days_ago = rng.integers(0, 30, n)
    last_assessed = [datetime.now() - timedelta(days=int(d)) for d in days_ago]

    df = pd.DataFrame(
        {
            "Institution": names,
            "Type": types,
            "CAR (%)": np.round(car, 2),
            "NPL Ratio (%)": np.round(npl, 2),
            "Liquidity Ratio (%)": np.round(liquidity, 2),
            "Audit Findings": audit_findings,
            "Late Reports": late_reports,
            "Risk Level": risks,
            "Confidence": np.round(confs, 2),
            "Last Assessed": last_assessed,
        }
    )
    return df


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("## 🏦 FS-04")
    st.markdown("**Non-Compliance Risk Predictor**")
    st.caption("Bank of Tanzania | Regulatory Supervision Tool")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔍 Predict", "📁 Batch Upload", "🏢 Institution Detail", "📄 Reports"],
        label_visibility="collapsed",
    )


# ============================================================
# PAGE: DASHBOARD
# ============================================================
def render_dashboard(df):
    st.title("Compliance Risk Dashboard")
    st.caption("Overview of monitored financial institutions")

    total = len(df)
    high = int((df["Risk Level"] == "High").sum())
    medium = int((df["Risk Level"] == "Medium").sum())
    low = int((df["Risk Level"] == "Low").sum())

    cols = st.columns(4)
    metric_card(cols[0], "Total Institutions", total)
    metric_card(cols[1], "High Risk", high, color=RISK_COLORS["High"])
    metric_card(cols[2], "Medium Risk", medium, color=RISK_COLORS["Medium"])
    metric_card(cols[3], "Low Risk", low, color=RISK_COLORS["Low"])

    col1, col2 = st.columns([1, 1.4])

    with col1:
        st.markdown('<h3 class="section-title">Risk Distribution</h3>', unsafe_allow_html=True)
        risk_counts = df["Risk Level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        fig = px.pie(
            names=risk_counts.index,
            values=risk_counts.values,
            color=risk_counts.index,
            color_discrete_map=RISK_COLORS,
            hole=0.55,
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown('<h3 class="section-title">Risk by Institution Type</h3>', unsafe_allow_html=True)
        type_risk = df.groupby(["Type", "Risk Level"]).size().reset_index(name="Count")
        fig2 = px.bar(
            type_risk,
            x="Type",
            y="Count",
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
            barmode="stack",
            category_orders={"Risk Level": ["Low", "Medium", "High"]},
        )
        fig2.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
        st.plotly_chart(fig2, width="stretch")

    st.markdown('<h3 class="section-title">Recent Assessments</h3>', unsafe_allow_html=True)
    recent = df.sort_values("Last Assessed", ascending=False).head(10).copy()
    recent["Last Assessed"] = recent["Last Assessed"].dt.strftime("%Y-%m-%d")
    display_cols = ["Institution", "Type", "CAR (%)", "NPL Ratio (%)", "Liquidity Ratio (%)", "Risk Level", "Last Assessed"]
    st.dataframe(style_table(recent[display_cols]), width="stretch", hide_index=True)


# ============================================================
# PAGE: PREDICT
# ============================================================
def render_predict():
    st.title("Predict Compliance Risk")
    st.caption("Enter an institution's regulatory indicators to assess non-compliance risk")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Institution Name", placeholder="e.g. Amani SACCO")
            inst_type = st.selectbox("Institution Type", ["Bank", "SACCO", "MFI"])
            car = st.number_input("Capital Adequacy Ratio (CAR) %", min_value=0.0, max_value=40.0, value=14.0, step=0.1)
            npl = st.number_input("Non-Performing Loan (NPL) Ratio %", min_value=0.0, max_value=30.0, value=4.0, step=0.1)
        with col2:
            liquidity = st.number_input("Liquidity Ratio %", min_value=0.0, max_value=60.0, value=25.0, step=0.1)
            audit_findings = st.number_input("Audit Findings (count)", min_value=0, max_value=10, value=1, step=1)
            late_reports = st.number_input("Late Report Submissions (count)", min_value=0, max_value=10, value=0, step=1)

        submitted = st.form_submit_button("Predict Risk", width="stretch")

    if not submitted:
        return

    risk, confidence = predict_risk(car, npl, liquidity, audit_findings, late_reports)

    st.markdown("---")
    col1, col2, col3 = st.columns([1.2, 1, 1.6])
    metric_card(col1, "Predicted Risk Level", risk, color=RISK_COLORS[risk])
    metric_card(col2, "Confidence", f"{confidence * 100:.1f}%")
    metric_card(col3, "Institution", f"{name or 'Unnamed'} ({inst_type})", value_size="1.2rem")

    st.markdown('<h3 class="section-title">Indicator Breakdown vs. BoT Thresholds</h3>', unsafe_allow_html=True)

    indicators = [
        ("Capital Adequacy Ratio", car, THRESHOLDS["car_min"], "%", "min"),
        ("NPL Ratio", npl, THRESHOLDS["npl_max"], "%", "max"),
        ("Liquidity Ratio", liquidity, THRESHOLDS["liquidity_min"], "%", "min"),
    ]
    for label, value, threshold, unit, direction in indicators:
        ok = (value >= threshold) if direction == "min" else (value <= threshold)
        status = "✓ Meets requirement" if ok else "⚠ Needs attention"
        color = "#059669" if ok else "#dc2626"
        cmp_symbol = "≥" if direction == "min" else "≤"
        denom = threshold * 1.5 if direction == "min" else threshold * 2
        pct = float(min(value / denom, 1.0)) if denom else 0.0

        st.markdown(
            f"**{label}**: {value:.2f}{unit} &nbsp;|&nbsp; Threshold: {cmp_symbol} {threshold}{unit} "
            f"&nbsp;|&nbsp; <span style='color:{color}; font-weight:600;'>{status}</span>",
            unsafe_allow_html=True,
        )
        st.progress(pct)

    st.markdown('<h3 class="section-title">Feature Importance (Model)</h3>', unsafe_allow_html=True)
    fi = get_feature_importance()
    fig = px.bar(fi, x="Importance", y="Feature", orientation="h", color_discrete_sequence=["#0d9488"])
    fig.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, width="stretch")


# ============================================================
# PAGE: BATCH UPLOAD
# ============================================================
def render_batch_upload():
    st.title("Batch Prediction")
    st.caption("Upload a CSV file with institution data to run predictions in bulk")

    st.info("Expected columns: Institution, Type, CAR, NPL, Liquidity, Audit Findings, Late Reports")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        data = pd.read_csv(uploaded)
        st.markdown('<h3 class="section-title">Preview</h3>', unsafe_allow_html=True)
        st.dataframe(data.head(), width="stretch")

        if st.button("Run Predictions", width="stretch"):
            results, confs = [], []
            for _, row in data.iterrows():
                r, c = predict_risk(
                    row.get("car_pct", 0),
                    row.get("npl_ratio_pct", 0),
                    row.get("liquidity_ratio", 0),
                    row.get("late_reports_count", 0),
                    row.get("audit_findings", 0),
                )
                results.append(r)
                confs.append(round(c, 2))
            data["Risk Level"] = results
            data["Confidence"] = confs

            st.markdown('<h3 class="section-title">Results</h3>', unsafe_allow_html=True)
            st.dataframe(style_table(data), width="stretch", hide_index=True)

            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results (CSV)", csv, "fs04_batch_results.csv", "text/csv", width="stretch"
            )
    else:
        st.markdown("##### Sample template")
        sample = pd.DataFrame(
            {
                "Institution": ["Example SACCO"],
                "Type": ["SACCO"],
                "car_pct": [13.5],
                "npl_ratio_pct": [3.2],
                "liquidity_ratio": [22.0],
                "late_reports_count": [1],
                "audit_findings": [0],
            }
        )
        st.dataframe(sample, width="stretch", hide_index=True)
        csv = sample.to_csv(index=False).encode("utf-8")
        st.download_button("Download Template", csv, "fs04_template.csv", "text/csv")


# ============================================================
# PAGE: INSTITUTION DETAIL
# ============================================================
def render_institution_detail(df):
    st.title("Institution Detail")

    selected = st.selectbox("Select Institution", df["Institution"].tolist())
    row = df[df["Institution"] == selected].iloc[0]

    col1, col2, col3 = st.columns(3)
    metric_card(col1, "Type", row["Type"], value_size="1.4rem")
    metric_card(col2, "Risk Level", row["Risk Level"], color=RISK_COLORS[row["Risk Level"]])
    metric_card(col3, "Confidence", f"{row['Confidence'] * 100:.0f}%")

    st.markdown('<h3 class="section-title">Regulatory Indicators</h3>', unsafe_allow_html=True)

    indicators = [
        ("Capital Adequacy Ratio", row["CAR (%)"], THRESHOLDS["car_min"], "min"),
        ("NPL Ratio", row["NPL Ratio (%)"], THRESHOLDS["npl_max"], "max"),
        ("Liquidity Ratio", row["Liquidity Ratio (%)"], THRESHOLDS["liquidity_min"], "min"),
    ]
    for label, value, threshold, direction in indicators:
        ok = (value >= threshold) if direction == "min" else (value <= threshold)
        status = "✓ Compliant" if ok else "⚠ Flagged"
        color = "#059669" if ok else "#dc2626"
        cmp_symbol = "≥" if direction == "min" else "≤"
        denom = threshold * 1.5 if direction == "min" else threshold * 2
        pct = float(min(value / denom, 1.0)) if denom else 0.0

        st.markdown(
            f"**{label}**: {value:.2f}% &nbsp;|&nbsp; Threshold: {cmp_symbol} {threshold}% "
            f"&nbsp;|&nbsp; <span style='color:{color}; font-weight:600;'>{status}</span>",
            unsafe_allow_html=True,
        )
        st.progress(pct)

    st.markdown('<h3 class="section-title">Assessment History (sample trend)</h3>', unsafe_allow_html=True)
    rng = np.random.default_rng(abs(hash(selected)) % (2**32))
    base_score = {"Low": 0.2, "Medium": 0.5, "High": 0.8}[row["Risk Level"]]
    dates = pd.date_range(end=row["Last Assessed"], periods=6, freq="30D")
    trend = pd.DataFrame(
        {
            "Date": dates,
            "Risk Score": np.clip(rng.normal(base_score, 0.07, 6), 0, 1),
        }
    )
    fig = px.line(trend, x="Date", y="Risk Score", markers=True, color_discrete_sequence=["#0d9488"])
    fig.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10), yaxis_range=[0, 1])
    st.plotly_chart(fig, width="stretch")


# ============================================================
# PAGE: REPORTS
# ============================================================
def render_reports(df):
    st.title("Compliance Reports")
    st.caption("Filter monitored institutions and export results for regulatory review")

    col1, col2 = st.columns(2)
    with col1:
        type_filter = st.multiselect("Institution Type", df["Type"].unique().tolist(), default=df["Type"].unique().tolist())
    with col2:
        risk_filter = st.multiselect("Risk Level", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])

    filtered = df[df["Type"].isin(type_filter) & df["Risk Level"].isin(risk_filter)]
    st.markdown(f"**{len(filtered)} institutions match the selected filters**")

    display_cols = [
        "Institution", "Type", "CAR (%)", "NPL Ratio (%)", "Liquidity Ratio (%)",
        "Audit Findings", "Late Reports", "Risk Level", "Confidence",
    ]
    display = filtered[display_cols]
    st.dataframe(style_table(display), width="stretch", hide_index=True, height=420)

    csv = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Report (CSV)", csv, "fs04_compliance_report.csv", "text/csv", width="stretch"
    )


# ============================================================
# MAIN ROUTING
# ============================================================
df = load_sample_data()

if page == "📊 Dashboard":
    render_dashboard(df)
elif page == "🔍 Predict":
    render_predict()
elif page == "📁 Batch Upload":
    render_batch_upload()
elif page == "🏢 Institution Detail":
    render_institution_detail(df)
elif page == "📄 Reports":
    render_reports(df)
