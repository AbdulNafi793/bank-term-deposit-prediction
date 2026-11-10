from pathlib import Path

import html
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Term Deposit Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent


@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(BASE / "final_model.joblib")
    info = joblib.load(BASE / "model_info.joblib")
    return model, info


model, info = load_artifacts()
THRESHOLD = info["threshold"]
COLUMNS = info["columns"]


# ------------------------------------------------------------
# CSS: Dashboard styling
# ------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --navy: #0B2545;
  --blue: #13315C;
  --teal: #1FA2A6;
  --teal-dark: #168487;
  --bg: #F3F6FA;
  --card: #FFFFFF;
  --text: #1F2937;
  --muted: #64748B;
  --border: #E2E8F0;
  --good: #16A34A;
  --good-bg: #DCFCE7;
  --good-text: #166534;
  --warn: #D97706;
  --warn-bg: #FEF3C7;
  --warn-text: #92400E;
}

.stApp,
.stMarkdown,
p,
label,
button,
input,
h1,
h2,
h3 {
  font-family: 'Inter', 'Segoe UI', sans-serif;
}

.stApp {
  background: var(--bg);
  color: var(--text);
}

#MainMenu,
footer {
  visibility: hidden;
}

/* Main content: top space and centered maximum width */
.block-container {
  padding-top: 5rem;
  padding-bottom: 3rem;
  padding-left: 2rem;
  padding-right: 2rem;
  max-width: 1280px;
  margin-left: auto;
  margin-right: auto;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--card);
  border-right: 1px solid var(--border);
}

[data-testid="stWidgetLabel"] p {
  color: var(--text);
  font-weight: 500;
  font-size: 0.85rem;
}

[data-testid="stExpander"] {
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--card);
}

[data-testid="stForm"] {
  border: none;
  padding: 0;
}

button[data-baseweb="tab"] {
  font-weight: 600;
}

[data-baseweb="tab-highlight"] {
  background: var(--teal);
}

/* Prediction button */
[data-testid="stFormSubmitButton"] button {
  background: var(--teal);
  border: none;
  border-radius: 10px;
  padding: 0.7rem 1rem;
  width: 100%;
  transition: background 0.2s ease;
}

[data-testid="stFormSubmitButton"] button p {
  color: #FFFFFF;
  font-weight: 600;
}

[data-testid="stFormSubmitButton"] button:hover {
  background: var(--teal-dark);
}

/* Sidebar headings */
.side-title {
  font-weight: 700;
  color: var(--navy);
  font-size: 1.2rem;
}

.side-sub {
  color: var(--muted);
  font-size: 0.85rem;
  margin-bottom: 1rem;
  line-height: 1.6;
}

/* Header banner */
.hero {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1.5rem;

  background: linear-gradient(
    120deg,
    var(--navy),
    var(--blue) 55%,
    var(--teal)
  );

  color: #FFFFFF;
  padding: 1.9rem 2.2rem;
  border-radius: 18px;
  margin-bottom: 2rem;
  min-height: 125px;

  box-shadow: 0 8px 22px rgba(11, 37, 69, 0.16);
}

.hero-title {
  font-size: 1.8rem;
  font-weight: 700;
  line-height: 1.35;
}

.hero-sub {
  font-size: 0.95rem;
  opacity: 0.9;
  margin-top: 0.5rem;
  line-height: 1.7;
}

.hero-badge {
  background: rgba(255, 255, 255, 0.15);
  border: 1px solid rgba(255, 255, 255, 0.35);
  padding: 0.65rem 1rem;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 600;
  white-space: nowrap;
}

/* General cards */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.5rem 1.6rem;

  box-shadow: 0 3px 8px rgba(11, 37, 69, 0.06);

  height: 100%;
  min-height: 155px;
  margin-bottom: 1rem;

  transition:
    transform 0.2s ease,
    box-shadow 0.2s ease;
}

.card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 18px rgba(11, 37, 69, 0.10);
}

.card-title {
  font-weight: 700;
  color: var(--navy);
  font-size: 1.05rem;
  margin-bottom: 0.7rem;
  line-height: 1.5;
}

.card ul {
  margin: 0.3rem 0 0 1.1rem;
  padding: 0;
  font-size: 0.9rem;
  color: var(--text);
  line-height: 1.7;
}

.card li {
  margin-bottom: 0.5rem;
}

/* KPI cards */
.kpi-label {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--muted);
  font-weight: 600;
}

.kpi-value {
  font-size: 2rem;
  font-weight: 700;
  color: var(--navy);
  line-height: 1.5;
  margin: 0.25rem 0;
}

.kpi-note {
  font-size: 0.92rem;
  color: var(--muted);
  line-height: 1.7;
}

/* Recommendation pills */
.pill {
  display: inline-block;
  padding: 0.25rem 0.8rem;
  border-radius: 999px;
  font-size: 1rem;
  font-weight: 700;
}

.pill-good {
  background: var(--good-bg);
  color: var(--good-text);
}

.pill-warn {
  background: var(--warn-bg);
  color: var(--warn-text);
}

/* Score bar */
.track {
  position: relative;
  height: 16px;
  background: #E8EEF5;
  border-radius: 999px;
  margin: 1.1rem 0 2.2rem;
}

.fill {
  height: 100%;
  border-radius: 999px;
}

.marker {
  position: absolute;
  top: -6px;
  width: 3px;
  height: 28px;
  background: var(--navy);
  border-radius: 2px;
}

.marker span {
  position: absolute;
  top: 32px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--navy);
  white-space: nowrap;
}

.scale {
  display: flex;
  justify-content: space-between;
  font-size: 0.75rem;
  color: var(--muted);
}

/* Summary table: centered text, clean horizontal lines only */
.summ {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
}

.summ td {
  padding: 0.7rem 0.75rem;
  border: none !important;
  border-bottom: 1px solid var(--border) !important;
  font-size: 0.9rem;
  text-align: center !important;
  vertical-align: middle;
}

.summ tr:last-child td {
  border-bottom: none !important;
}

.summ .k {
  color: var(--muted);
}

.summ .v {
  font-weight: 600;
  color: var(--navy);
}

/* Instruction card numbers */
.step-num {
  display: inline-flex;
  align-items: center;
  justify-content: center;

  width: 34px;
  height: 34px;

  background: var(--teal);
  color: #FFFFFF;
  border-radius: 50%;
  font-weight: 700;
  font-size: 1rem;

  margin-bottom: 0.8rem;
}

/* Supporting text */
.note {
  font-size: 0.82rem;
  color: var(--muted);
  margin-top: 0.6rem;
  line-height: 1.6;
}

/* Footer */
.footer {
  text-align: center;
  color: var(--muted);
  font-size: 0.82rem;
  margin-top: 2rem;
  padding-bottom: 0.5rem;
}

/* Make cards fit on tablets and phones */
@media (max-width: 900px) {
  .block-container {
    padding-left: 1.25rem;
    padding-right: 1.25rem;
  }

  .hero {
    padding: 1.6rem;
    flex-wrap: wrap;
  }

  .hero-title {
    font-size: 1.55rem;
  }

  .hero-badge {
    white-space: normal;
  }
}

@media (max-width: 600px) {
  .block-container {
    padding-top: 1.5rem;
    padding-left: 1rem;
    padding-right: 1rem;
  }

  .hero {
    flex-direction: column;
    align-items: flex-start;
    padding: 1.4rem;
    border-radius: 14px;
  }

  .hero-title {
    font-size: 1.35rem;
  }

  .hero-sub {
    font-size: 0.88rem;
  }

  .card {
    padding: 1.2rem;
  }

  .kpi-value {
    font-size: 1.6rem;
  }
}
"""

st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def pretty(value):
    return str(value).replace(".", " ").replace("-", " ").strip().capitalize()


def safe_text(value):
    """Escape text before inserting it into custom HTML."""
    return html.escape(str(value))


def kpi(label, value_html, note=""):
    return (
        '<div class="card">'
        f'<div class="kpi-label">{safe_text(label)}</div>'
        f'<div class="kpi-value">{value_html}</div>'
        f'<div class="kpi-note">{safe_text(note)}</div>'
        "</div>"
    )


def prob_bar(prob, threshold):
    p = max(0.0, min(prob, 1.0)) * 100
    t = max(0.0, min(threshold, 1.0)) * 100

    color = "var(--good)" if prob >= threshold else "var(--warn)"

    return (
        '<div class="card">'
        '<div class="kpi-label">Score against the cutoff</div>'
        '<div class="track">'
        f'<div class="fill" style="width:{p:.1f}%; background:{color};"></div>'
        f'<div class="marker" style="left:{t:.1f}%;"><span>Cutoff {t:.0f}%</span></div>'
        "</div>"
        '<div class="scale"><span>0%</span><span>100%</span></div>'
        '<div class="note">Scores come from a model trained with balanced class weights, '
        "so they are not exact probabilities. Compare them with the cutoff instead of "
        "reading them as percentages.</div>"
        "</div>"
    )


def summary_card(title, pairs):
    rows = "".join(
        f'<tr><td class="k">{safe_text(k)}</td>'
        f'<td class="v">{safe_text(v)}</td></tr>'
        for k, v in pairs
    )

    return (
        '<div class="card">'
        f'<div class="card-title">{safe_text(title)}</div>'
        f'<table class="summ">{rows}</table>'
        "</div>"
    )


def list_card(title, items):
    li = "".join(f"<li>{safe_text(item)}</li>" for item in items)

    return (
        '<div class="card">'
        f'<div class="card-title">{safe_text(title)}</div>'
        f"<ul>{li}</ul>"
        "</div>"
    )


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.markdown(
    '<div class="hero">'
    "<div>"
    '<div class="hero-title">Term Deposit Subscription Predictor</div>'
    '<div class="hero-sub">Estimate which clients are worth a marketing call, '
    "based on the UCI Bank Marketing data.</div>"
    "</div>"
    '<div class="hero-badge">Tuned LightGBM &middot; ROC-AUC 0.816</div>'
    "</div>",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Sidebar: client form
# ------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="side-title">Client profile</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="side-sub">Fill in the details, then run the prediction.</div>',
        unsafe_allow_html=True,
    )

    with st.form("client_form"):

        with st.expander("Client details", expanded=True):
            age = st.number_input(
                "Age",
                min_value=17,
                max_value=100,
                value=40,
            )

            job = st.selectbox(
                "Job",
                [
                    "admin.",
                    "blue-collar",
                    "entrepreneur",
                    "housemaid",
                    "management",
                    "retired",
                    "self-employed",
                    "services",
                    "student",
                    "technician",
                    "unemployed",
                    "unknown",
                ],
                format_func=pretty,
            )

            marital = st.selectbox(
                "Marital status",
                ["married", "single", "divorced", "unknown"],
                format_func=pretty,
            )

            education = st.selectbox(
                "Education",
                [
                    "university.degree",
                    "high.school",
                    "professional.course",
                    "basic.9y",
                    "basic.6y",
                    "basic.4y",
                    "illiterate",
                    "unknown",
                ],
                format_func=pretty,
            )

            default = st.selectbox(
                "Credit in default?",
                ["no", "unknown", "yes"],
                format_func=pretty,
            )

            housing = st.selectbox(
                "Housing loan?",
                ["yes", "no", "unknown"],
                format_func=pretty,
            )

            loan = st.selectbox(
                "Personal loan?",
                ["no", "yes", "unknown"],
                format_func=pretty,
            )

        with st.expander("Campaign details"):
            contact = st.selectbox(
                "Contact type",
                ["cellular", "telephone"],
                format_func=pretty,
            )

            month = st.selectbox(
                "Month of contact",
                ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
                index=2,
                format_func=pretty,
            )

            day_of_week = st.selectbox(
                "Day of week",
                ["mon", "tue", "wed", "thu", "fri"],
                format_func=pretty,
            )

            campaign = st.number_input(
                "Contacts during this campaign",
                min_value=1,
                max_value=60,
                value=2,
            )

        with st.expander("Previous campaigns"):
            prev_contacted = st.checkbox(
                "Contacted in a previous campaign?",
                value=False,
            )

            pdays_input = st.number_input(
                "Days since last contact (only used if contacted before)",
                min_value=0,
                max_value=30,
                value=6,
            )

            previous = st.number_input(
                "Contacts before this campaign",
                min_value=0,
                max_value=10,
                value=0,
            )

            poutcome = st.selectbox(
                "Outcome of previous campaign",
                ["nonexistent", "failure", "success"],
                format_func=pretty,
            )

        with st.expander("Economic indicators"):
            st.caption("Defaults are typical values from the data.")

            emp_var_rate = st.number_input(
                "Employment variation rate",
                min_value=-3.5,
                max_value=1.5,
                value=1.1,
                step=0.1,
            )

            cons_price_idx = st.number_input(
                "Consumer price index",
                min_value=92.0,
                max_value=95.0,
                value=93.749,
                step=0.001,
                format="%.3f",
            )

            cons_conf_idx = st.number_input(
                "Consumer confidence index",
                min_value=-51.0,
                max_value=-26.0,
                value=-41.8,
                step=0.1,
            )

            euribor3m = st.number_input(
                "Euribor 3-month rate",
                min_value=0.5,
                max_value=5.1,
                value=4.857,
                step=0.001,
                format="%.3f",
            )

            nr_employed = st.number_input(
                "Number employed (thousands)",
                min_value=4900.0,
                max_value=5300.0,
                value=5191.0,
                step=1.0,
            )

        submitted = st.form_submit_button("Run prediction")


# ------------------------------------------------------------
# Run prediction  (the main part: build the row, score it)
# ------------------------------------------------------------

if submitted:
    row = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "campaign": campaign,
        "pdays": pdays_input if prev_contacted else 0,
        "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed,
        "previously_contacted": int(prev_contacted),
    }

    X_new = pd.DataFrame([row])[COLUMNS]

    prob = float(model.predict_proba(X_new)[0, 1])

    st.session_state["result"] = {
        "prob": prob,
        "row": row,
    }


res = st.session_state.get("result")


# ------------------------------------------------------------
# Main dashboard tabs
# ------------------------------------------------------------

tab_pred, tab_model = st.tabs(["Prediction", "About the model"])


# ------------------------------------------------------------
# Prediction tab
# ------------------------------------------------------------

with tab_pred:

    if res is None:

        s1, s2, s3 = st.columns(3, gap="large")

        with s1:
            st.markdown(
                '<div class="card">'
                '<div class="step-num">1</div>'
                '<div class="card-title">Enter the client profile</div>'
                '<div class="kpi-note">Use the panel on the left to describe '
                'the client and the campaign.</div>'
                "</div>",
                unsafe_allow_html=True,
            )

        with s2:
            st.markdown(
                '<div class="card">'
                '<div class="step-num">2</div>'
                '<div class="card-title">Run the prediction</div>'
                '<div class="kpi-note">Press the button at the bottom of the '
                'panel. The model scores the client in under a second.</div>'
                "</div>",
                unsafe_allow_html=True,
            )

        with s3:
            st.markdown(
                '<div class="card">'
                '<div class="step-num">3</div>'
                '<div class="card-title">Read the recommendation</div>'
                f'<div class="kpi-note">Clients scoring at or above '
                f'{THRESHOLD:.0%} are flagged as worth calling.</div>'
                "</div>",
                unsafe_allow_html=True,
            )

    else:
        prob = res["prob"]
        row = res["row"]

        call = prob >= THRESHOLD
        gap = (prob - THRESHOLD) * 100

        k1, k2, k3 = st.columns(3, gap="large")

        with k1:
            st.markdown(
                kpi(
                    "Likelihood score",
                    f"{prob:.1%}",
                    "Higher means more likely to subscribe",
                ),
                unsafe_allow_html=True,
            )

        pill = (
            '<span class="pill pill-good">Call this client</span>'
            if call
            else '<span class="pill pill-warn">Low priority</span>'
        )

        with k2:
            st.markdown(
                kpi(
                    "Recommendation",
                    pill,
                    f"Cutoff is {THRESHOLD:.0%}",
                ),
                unsafe_allow_html=True,
            )

        with k3:
            st.markdown(
                kpi(
                    "Distance from cutoff",
                    f"{gap:+.0f} pts",
                    "Above the cutoff" if call else "Below the cutoff",
                ),
                unsafe_allow_html=True,
            )

        st.markdown(
            prob_bar(prob, THRESHOLD),
            unsafe_allow_html=True,
        )

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown(
                summary_card(
                    "Client summary",
                    [
                        ("Age", row["age"]),
                        ("Job", pretty(row["job"])),
                        ("Education", pretty(row["education"])),
                        ("Contact type", pretty(row["contact"])),
                        ("Month of contact", pretty(row["month"])),
                        ("Contacts this campaign", row["campaign"]),
                        (
                            "Contacted before",
                            "Yes" if row["previously_contacted"] else "No",
                        ),
                        ("Previous outcome", pretty(row["poutcome"])),
                    ],
                ),
                unsafe_allow_html=True,
            )

        with right:
            if call:
                title = "Recommended action: call"
                body = (
                    f"This client's score is above the {THRESHOLD:.0%} cutoff. "
                    "The cutoff was chosen to reach most of the likely subscribers "
                    "without wasting too many calls."
                )
            else:
                title = "Recommended action: low priority"
                body = (
                    f"This client's score is below the {THRESHOLD:.0%} cutoff. "
                    "A call is still possible, but the model expects a lower return "
                    "than for clients above the cutoff."
                )

            st.markdown(
                '<div class="card">'
                f'<div class="card-title">{safe_text(title)}</div>'
                f'<div class="kpi-note" style="font-size:0.95rem;">'
                f'{safe_text(body)}</div>'
                '<div class="note">Try changing the contact type, the month or '
                'the previous outcome in the left panel to see how the score moves.'
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )


# ------------------------------------------------------------
# About the model tab
# ------------------------------------------------------------

with tab_model:

    m1, m2, m3, m4 = st.columns(4, gap="medium")

    with m1:
        st.markdown(
            kpi(
                "Training data",
                "41,188",
                "Calls, with about 11% subscribing",
            ),
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            kpi(
                "ROC-AUC",
                "0.816",
                "Test set, independent of the cutoff",
            ),
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            kpi(
                "Recall",
                "68%",
                f"Subscribers found at the {THRESHOLD:.0%} cutoff",
            ),
            unsafe_allow_html=True,
        )

    with m4:
        st.markdown(
            kpi(
                "Precision",
                "36%",
                "Flagged clients who subscribed (11% at random)",
            ),
            unsafe_allow_html=True,
        )

    a, b = st.columns(2, gap="large")

    with a:
        st.markdown(
            list_card(
                "How the model was built",
                [
                    "UCI Bank Marketing data: phone campaigns of a Portuguese bank.",
                    "Call duration was removed because it is only known after the call (data leakage).",
                    "Several models were compared (linear, distance-based, tree, bagging and boosting), and a tuned LightGBM was chosen using 5-fold cross-validation.",
                    "The cutoff was picked on training data with cross-validated predictions, assuming one subscription is worth about 10 calls.",
                ],
            ),
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            list_card(
                "Things to keep in mind",
                [
                    "Scores are not exact probabilities, because the model uses balanced class weights.",
                    "Economic indicators carry most of the model's importance, so it reflects the 2008 to 2010 period.",
                    "This is a portfolio project, not a production system or financial advice.",
                ],
            ),
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.markdown(
    '<div class="footer">Built with scikit-learn, LightGBM and Streamlit</div>',
    unsafe_allow_html=True,
)