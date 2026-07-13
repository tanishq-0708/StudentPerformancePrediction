from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="AI Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "student_model.pkl"

SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

FEATURES_PATH = BASE_DIR / "models" / "features.pkl"


@dataclass
class PredictionResult:
    """Structured container for a single prediction result."""
    predicted_score: float
    performance_category: str
    confidence_message: str
    recommendations: List[str]
    badge_color: str



def inject_custom_css() -> None:
    """Inject the premium black & gold glassmorphism theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }

        .stApp {
            background: radial-gradient(circle at 10% 10%, #1a1a1a 0%, #0d0d0d 45%, #000000 100%);
            color: #F5F5F5;
        }

        /* Hide default streamlit chrome */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* ---------- HERO ---------- */
        .hero-container {
            text-align: center;
            padding: 50px 20px 30px 20px;
            animation: fadeInDown 1s ease;
        }
        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(90deg, #FFD700, #FFF3B0, #D4AF37);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2em;
            letter-spacing: 1px;
        }
        .hero-subtitle {
            font-size: 1.15rem;
            color: #cfcfcf;
            font-weight: 300;
        }

        /* ---------- GLASS CARD ---------- */
        .glass-card {
            background: rgba(255, 255, 255, 0.045);
            border: 1px solid rgba(212, 175, 55, 0.35);
            border-radius: 20px;
            padding: 26px 28px;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.55);
            margin-bottom: 22px;
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }
        .glass-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(212, 175, 55, 0.25);
        }

        .section-heading {
            font-size: 1.6rem;
            font-weight: 700;
            color: #FFD700;
            margin-bottom: 14px;
            border-left: 4px solid #D4AF37;
            padding-left: 12px;
        }

        /* ---------- SCORE CARD ---------- */
        .score-card {
            text-align: center;
            padding: 34px 20px;
            border-radius: 24px;
            background: linear-gradient(145deg, rgba(212,175,55,0.14), rgba(255,255,255,0.02));
            border: 1px solid rgba(212, 175, 55, 0.5);
            box-shadow: 0 0 40px rgba(212, 175, 55, 0.18);
            animation: fadeInUp 0.8s ease;
        }
        .score-value {
            font-size: 4.2rem;
            font-weight: 800;
            color: #FFD700;
            text-shadow: 0 0 25px rgba(255, 215, 0, 0.45);
        }
        .score-label {
            font-size: 1rem;
            color: #cfcfcf;
            letter-spacing: 2px;
            text-transform: uppercase;
        }

        /* ---------- BADGE ---------- */
        .badge {
            display: inline-block;
            padding: 10px 26px;
            border-radius: 50px;
            font-weight: 700;
            font-size: 1.05rem;
            margin-top: 14px;
            letter-spacing: 1px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.4);
        }

        /* ---------- RECOMMENDATION CARD ---------- */
        .rec-card {
            background: rgba(255, 255, 255, 0.04);
            border-left: 4px solid #D4AF37;
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 12px;
            color: #eaeaea;
            font-size: 0.98rem;
            animation: fadeIn 0.6s ease;
        }

        /* ---------- PIPELINE ---------- */
        .pipeline-step {
            background: rgba(212, 175, 55, 0.08);
            border: 1px solid rgba(212, 175, 55, 0.4);
            border-radius: 14px;
            padding: 14px;
            text-align: center;
            font-weight: 600;
            color: #FFD700;
            margin-bottom: 6px;
        }
        .pipeline-arrow {
            text-align: center;
            color: #D4AF37;
            font-size: 1.4rem;
            margin: 2px 0;
        }

        /* ---------- FOOTER ---------- */
        .footer {
            text-align: center;
            color: #8a8a8a;
            padding: 30px 0 10px 0;
            font-size: 0.85rem;
            border-top: 1px solid rgba(255,255,255,0.08);
            margin-top: 40px;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d0d0d 0%, #1a1a1a 100%);
            border-right: 1px solid rgba(212, 175, 55, 0.25);
        }

        /* ---------- Buttons ---------- */
        div.stButton > button {
            background: linear-gradient(90deg, #D4AF37, #FFD700);
            color: #000000;
            font-weight: 700;
            border: none;
            border-radius: 50px;
            padding: 12px 30px;
            font-size: 1.05rem;
            transition: all 0.25s ease;
            box-shadow: 0 4px 20px rgba(212, 175, 55, 0.35);
        }
        div.stButton > button:hover {
            transform: scale(1.03);
            box-shadow: 0 6px 28px rgba(212, 175, 55, 0.55);
            color: #000000;
        }

        /* ---------- Animations ---------- */
        @keyframes fadeInDown {
            from {opacity: 0; transform: translateY(-25px);}
            to {opacity: 1; transform: translateY(0);}
        }
        @keyframes fadeInUp {
            from {opacity: 0; transform: translateY(25px);}
            to {opacity: 1; transform: translateY(0);}
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }

        hr {
            border-color: rgba(212, 175, 55, 0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



@st.cache_resource(show_spinner=False)
def load_artifacts() -> Dict[str, Any]:
    """
    Load the pre-trained Hybrid Stacking Regressor and StandardScaler
    from disk. Cached so the pickle files are only read once per
    session.

    Returns:
        Dictionary with keys 'model' and 'scaler'.

    Raises:
        FileNotFoundError: if the pickle artifacts are missing.
    """
    if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Ensure student_model.pkl and "
            "scaler.pkl exist in the project root. Run train_model.py "
            "to generate them."
        )

    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    with open(SCALER_PATH, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    with open(FEATURES_PATH, "rb") as features_file:
        features = pickle.load(features_file)

    return {"model": model, "scaler": scaler, "features": features}


# ======================================================================
# BUSINESS LOGIC
# ======================================================================

def classify_performance(score: float) -> Dict[str, str]:
    """
    Map a predicted numeric score to a performance category, along with
    a badge color used for UI styling.

    Args:
        score: Predicted exam score (0-100).

    Returns:
        Dictionary with 'category' and 'color' keys.
    """
    if score >= 90:
        return {"category": "Outstanding", "color": "#00E676"}
    elif score >= 80:
        return {"category": "Excellent", "color": "#FFD700"}
    elif score >= 70:
        return {"category": "Very Good", "color": "#40C4FF"}
    elif score >= 60:
        return {"category": "Good", "color": "#FFA726"}
    elif score >= 50:
        return {"category": "Average", "color": "#FF7043"}
    else:
        return {"category": "Needs Improvement", "color": "#EF5350"}


def build_confidence_message(score: float, category: str) -> str:
    """
    Build a natural-language confidence / summary message.

    Args:
        score: Predicted exam score.
        category: Performance category label.

    Returns:
        Human readable sentence describing the prediction.
    """
    return f"The student is likely to score {score:.2f} marks with {category} performance."


def generate_recommendations(
    attendance: float, hours_studied: float, previous_score: float, sleeping_hours: float, tutoring_sessions: float, predicted_score: float
) -> List[str]:
    """
    Generate personalized, rule-based recommendations from the raw
    inputs and the predicted score.

    Args:
        attendance: Attendance percentage (0-100).
        hours_studied: Daily study hours (0-24).
        previous_score: Previous / internal score (0-100).
        predicted_score: Model's predicted exam score.

    Returns:
        List of recommendation strings.
    """
    recommendations: List[str] = []

    if attendance < 75:
        recommendations.append(
            "📉 Low Attendance: Attendance is below the recommended threshold. "
            "Improving class attendance is strongly correlated with better outcomes."
        )

    if hours_studied < 3:
        recommendations.append(
            "⏱️ Low Study Hours: Consider increasing daily study time to at least "
            "3–4 focused hours to strengthen conceptual understanding."
        )

    if previous_score < 60:
        recommendations.append(
            "📚 Low Previous Score: Revising weak subjects and past mistakes will "
            "help build a stronger foundation for upcoming exams."
        )

    if predicted_score >= 85:
        recommendations.append(
            "🏆 High Predicted Score: Excellent trajectory! Keep up the consistency "
            "in attendance and study habits to maintain this performance."
        )

    if not recommendations:
        recommendations.append(
            "✅ Balanced Profile: Current habits look solid. Maintain consistency "
            "across attendance, study hours, and revision for continued growth."
        )

    return recommendations


def run_inference(
    attendance: float, hours_studied: float, previous_score: float, sleeping_hours: float, tutoring_sessions: float
) -> PredictionResult:
    """
    Full inference pipeline: scale inputs -> predict -> classify ->
    build message -> generate recommendations.

    Args:
        attendance: Attendance percentage input.
        hours_studied: Hours studied input.
        previous_score: Previous / internal score input.

    Returns:
        PredictionResult dataclass with all derived outputs.
    """
    artifacts = load_artifacts()
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    features = artifacts["features"]

    raw_features = pd.DataFrame([[attendance, hours_studied, previous_score, sleeping_hours, tutoring_sessions]], columns=features)
    scaled_features = scaler.transform(raw_features)

    raw_prediction = float(model.predict(scaled_features)[0])
    predicted_score = float(np.clip(raw_prediction, 0, 100))

    classification = classify_performance(predicted_score)
    category = classification["category"]
    badge_color = classification["color"]

    confidence_message = build_confidence_message(predicted_score, category)
    recommendations = generate_recommendations(
        attendance, hours_studied, previous_score, sleeping_hours, tutoring_sessions, predicted_score
    )

    return PredictionResult(
        predicted_score=round(predicted_score, 2),
        performance_category=category,
        confidence_message=confidence_message,
        recommendations=recommendations,
        badge_color=badge_color,
    )


def build_json_response(
    result: PredictionResult, attendance: float, hours_studied: float, previous_score: float, sleeping_hours: float, tutoring_sessions: float
) -> Dict[str, Any]:
    """
    Build the structured JSON-style response required by the backend
    specification.

    Args:
        result: PredictionResult produced by run_inference.
        attendance: Raw attendance input.
        hours_studied: Raw hours studied input.
        previous_score: Raw previous score input.

    Returns:
        Dictionary matching the required response schema.
    """
    return {
        "predicted_score": result.predicted_score,
        "performance_category": result.performance_category,
        "recommendation": result.recommendations,
        "chart_data": {
            "attendance": attendance,
            "hours_studied": hours_studied,
            "previous_score": previous_score,
            "sleeping_hours": sleeping_hours,
            "tutoring_sessions": tutoring_sessions,
            "predicted_score": result.predicted_score,
        },
    }


# ======================================================================
# CHART BUILDERS (PLOTLY)
# ======================================================================

PLOTLY_TEMPLATE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#F5F5F5", family="Poppins"),
    margin=dict(l=30, r=30, t=50, b=30),
)


def build_bar_chart(attendance: float, hours_studied: float, previous_score: float, sleeping_hours:float, tutoring_sessions:float,predicted_score: float) -> go.Figure:
    """Build a comparative bar chart of the four key metrics."""
    labels = ["Attendance (%)", "Hours Studied", "Previous Score", "Sleep Hours", "Tutoring Session","Predicted Score"]
    values = [attendance, hours_studied, previous_score, sleeping_hours, tutoring_sessions, predicted_score]
    display_values = [attendance, hours_studied, previous_score, sleeping_hours, tutoring_sessions, predicted_score]

    colors = ["#40C4FF", "#FFA726", "#AB47BC", "#FFD700", "#00E676", "#FF8800"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                text=[f"{v:.1f}" for v in display_values],
                textposition="outside",
                marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
            )
        ]
    )
    fig.update_layout(
        title="Input & Predicted Metrics Comparison",
        yaxis=dict(range=[0, 110], gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        **PLOTLY_TEMPLATE_LAYOUT,
    )
    return fig


def build_gauge_chart(predicted_score: float, badge_color: str) -> go.Figure:
    """Build a gauge chart representing the predicted score."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=predicted_score,
            number={"suffix": " / 100", "font": {"color": "#FFD700", "size": 42}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#F5F5F5"},
                "bar": {"color": badge_color},
                "bgcolor": "rgba(255,255,255,0.03)",
                "borderwidth": 1,
                "bordercolor": "rgba(212,175,55,0.4)",
                "steps": [
                    {"range": [0, 50], "color": "rgba(239,83,80,0.25)"},
                    {"range": [50, 60], "color": "rgba(255,112,67,0.25)"},
                    {"range": [60, 70], "color": "rgba(255,167,38,0.25)"},
                    {"range": [70, 80], "color": "rgba(64,196,255,0.25)"},
                    {"range": [80, 90], "color": "rgba(255,215,0,0.25)"},
                    {"range": [90, 100], "color": "rgba(0,230,118,0.25)"},
                ],
            },
        )
    )
    fig.update_layout(title="Predicted Exam Score", **PLOTLY_TEMPLATE_LAYOUT)
    return fig


def build_radar_chart(attendance: float, hours_studied: float, previous_score: float, predicted_score: float, sleeping_hours: float, tutoring_sessions: float) -> go.Figure:
    """Build a radar (spider) chart representing the student's overall profile."""
    categories = ["Attendance", "Hours Studied", "Previous Score", "Sleep Hours", "Tutoring Sessions","Predicted Score"]
    values = [attendance, hours_studied, previous_score, sleeping_hours, tutoring_sessions, predicted_score]
    values += values[:1]
    categories += categories[:1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            fillcolor="rgba(212,175,55,0.25)",
            line=dict(color="#FFD700", width=2),
            name="Student Profile",
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        ),
        showlegend=False,
        title="Student Profile Radar",
        **PLOTLY_TEMPLATE_LAYOUT,
    )
    return fig


def build_pie_chart(predicted_category: str) -> go.Figure:
    """Build a pie chart of performance category distribution, highlighting the predicted category."""
    categories = ["Outstanding", "Excellent", "Very Good", "Good", "Average", "Needs Improvement"]
    # Illustrative reference distribution across a typical cohort
    colors = ["#00E676", "#FFD700", "#40C4FF", "#FFA726", "#FF7043", "#EF5350"]

    pull = [0.12 if cat == predicted_category else 0 for cat in categories]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=categories,
                values=base_distribution,
                pull=pull,
                marker=dict(colors=colors, line=dict(color="#000000", width=2)),
                textinfo="label+percent",
                hole=0.35,
            )
        ]
    )
    fig.update_layout(
        title=f"Performance Category Distribution (Highlighted: {predicted_category})",
        showlegend=False,
        **PLOTLY_TEMPLATE_LAYOUT,
    )
    return fig


def render_sidebar() -> str:
    """
    Render the sidebar navigation and return the selected page.

    Returns:
        The selected navigation option as a string.
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 10px 0 20px 0;">
                <div style="font-size:3rem;">🎓</div>
                <div style="font-size:1.3rem; font-weight:700; color:#FFD700;">
                    AI Performance
                </div>
                <div style="font-size:0.85rem; color:#aaaaaa; letter-spacing:1px;">
                    PREDICTION SYSTEM
                </div>
            </div>
            <hr>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            options=["🏠 Predict", "📊 About Project", "👨‍💻 Developer Info"],
            label_visibility="collapsed",
        )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size:0.85rem; color:#999;">
            <b style="color:#FFD700;">Model:</b> Hybrid Stacking Ensemble<br>
            <b style="color:#FFD700;">Base Learners:</b> Linear Regression, XGBoost<br>
            <b style="color:#FFD700;">Meta Learner:</b> Linear Regression<br>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return page


def render_hero() -> None:
    """Render the hero header section."""
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">🎓 AI Student Performance Prediction</div>
            <div class="hero-subtitle">
                Predict academic performance using Hybrid Ensemble Machine Learning.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_form() -> Dict[str, float]:
    """
    Render the prediction input form (sliders) inside a glass card.

    Returns:
        Dictionary of the three raw feature inputs.
    """
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">📝 Student Prediction Form</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        attendance = st.slider(
            "Attendance (%)", min_value=0, max_value=100, value=85, step=1,
            help="Overall class attendance percentage.",
        )

    with col2:
        hours_studied = st.slider(
            "Hours Studied (per day)", min_value=0, max_value=24, value=4, step=1,
            help="Average number of hours studied per day.",
        )

    with col3:
        previous_score = st.slider(
            "Previous Academic Score Avg", min_value=0, max_value=100, value=75, step=1,
            help="Most recent internal and assessment scores average. i.e. avg=(total sum of internal exam scores + total sum of assessment scores)/Number of subjects.",
        )

    with col4:
        sleeping_hours = st.slider(
            "Sleeping Hours (per day)", min_value=0, max_value=24, value=8, step=1,
            help="Total Sleeping hours per day. Adequate sleep is crucial for cognitive function and academic performance.",
        )

    with col5:
        tutoring_sessions = st.slider(
            "Tutoring Session  Hours (per day)", min_value=0, max_value=24, value=9, step=1,
            help="Hours spent in overall tutoring sessions in a day. Additionally tutoring can help clarify difficult concepts and improve understanding.",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    return {
        "attendance": float(attendance),
        "hours_studied": float(hours_studied),
        "previous_score": float(previous_score),
        "sleeping_hours": float(sleeping_hours),
        "tutoring_sessions": float(tutoring_sessions)
    }


def render_result_section(result: PredictionResult, inputs: Dict[str, float]) -> None:
    st.markdown("<br>", unsafe_allow_html=True)

    score_col, badge_col = st.columns([1, 1])

    with score_col:
        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-label">Predicted Exam Score</div>
                <div class="score-value">{result.predicted_score:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with badge_col:
        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-label">Performance Category</div><br>
                <span class="badge" style="background:{result.badge_color}22; color:{result.badge_color}; border: 1px solid {result.badge_color};">
                    {result.performance_category}
                </span>
                <p style="margin-top:18px; color:#dddddd; font-size:0.98rem;">
                    {result.confidence_message}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Recommendations
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">💡 Personalized Recommendations</div>', unsafe_allow_html=True)
    for rec in result.recommendations:
        st.markdown(f'<div class="rec-card">{rec}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Visualizations
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">📊 Visual Analytics</div>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(
            build_bar_chart(
                inputs["attendance"], inputs["hours_studied"], inputs["previous_score"], inputs["sleeping_hours"], inputs["tutoring_sessions"],result.predicted_score
            ),
            use_container_width=True,
        )
    with chart_col2:
        st.plotly_chart(
            build_gauge_chart(result.predicted_score, result.badge_color),
            use_container_width=True,
        )

    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        st.plotly_chart(
            build_radar_chart(
                inputs["attendance"], inputs["hours_studied"], inputs["previous_score"], inputs["sleeping_hours"], inputs["tutoring_sessions"], result.predicted_score
            ),
            use_container_width=True,
        )
    with chart_col4:
        st.plotly_chart(
            build_pie_chart(result.performance_category),
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # JSON Response (developer view)
    with st.expander("🧩 View Raw JSON Response (Backend Output)"):
        st.json(build_json_response(result, inputs["attendance"], inputs["hours_studied"], inputs["previous_score"], inputs["sleeping_hours"], inputs["tutoring_sessions"]))


def render_pipeline_diagram() -> None:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">🧠 Machine Learning Pipeline</div>', unsafe_allow_html=True)

    steps = [
        "🧑‍🎓 Student Inputs (Attendance, Hours Studied, Previous Score, Sleeping Hours, Tutoring Sessions)",
        "⚙️ Preprocessing (Standard Scaling)",
        "📈 Base Learner 1: Linear Regression",
        "🌲 Base Learner 2: XGBoost Regressor",
        "🧮 Meta Learner: Linear Regression (Stacking)",
        "🎯 Predicted Exam Score",
    ]

    for i, step in enumerate(steps):
        st.markdown(f'<div class="pipeline-step">{step}</div>', unsafe_allow_html=True)
        if i != len(steps) - 1:
            st.markdown('<div class="pipeline-arrow">↓</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_about_section() -> None:
    render_hero()

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">📌 Problem Statement</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Educational institutions often struggle to identify students who may
        underperform in final examinations until it is too late for
        meaningful intervention. This system uses historical academic
        indicators — **attendance**, **study hours**, and **previous
        performance** — to proactively predict a student's likely exam
        score, enabling timely, personalized academic support.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">🗂️ Dataset</div>', unsafe_allow_html=True)
    st.markdown(
        """
        The model is trained on structured academic records containing three
        core numerical features per student:

        - **Attendance (%)** — ranges from 0 to 100
        - **Hours Studied** — average daily study hours, 0 to 24
        - **Previous Academic Score Avgerage** — most recent internal exam scores and assessment score average, 0 to 100
        - **Sleeping Hours** — average daily sleeping hours, 0 to 24
        - **Tutoring Sessions Hours** — average daily tutoring session hours, 0 to 24

        The target variable is the **Final Exam Score** (0–100), which the
        model learns to predict from the above features.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">🔬 Machine Learning Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Raw student inputs are first standardized using a fitted
        **StandardScaler**, ensuring all features contribute proportionally
        regardless of their original scale. The scaled features are then
        passed through the Hybrid Stacking Ensemble described below.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    render_pipeline_diagram()

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">🧬 Hybrid Stacking Model</div>', unsafe_allow_html=True)
    st.markdown(
        """
        The predictive core of this system is a **Stacking Regressor**
        combining the complementary strengths of two base learners:

        1. **Linear Regression** — captures straightforward, additive
           relationships between attendance, study hours, previous score,
           and the final exam score.
        2. **XGBoost Regressor** — a gradient-boosted tree ensemble that
           captures non-linear interactions and saturation effects (e.g.
           diminishing returns from excessive study hours).

        The predictions of both base learners are combined by a **Linear
        Regression meta-learner**, which learns the optimal weighting
        between the two models to produce the final, more robust
        prediction than either model alone.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">🚀 Deployment</div>', unsafe_allow_html=True)
    st.markdown(
        """
        The trained model and scaler are serialized using **Pickle** and
        loaded once at application startup. The interactive dashboard is
        built entirely with **Streamlit**, styled with a custom black &
        gold glassmorphism theme, and visualized using **Plotly** for rich,
        interactive charts. The application performs inference only — no
        training occurs at runtime, ensuring fast, consistent, and
        reproducible predictions.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_developer_info() -> None:
    """Render the Developer Info page."""
    render_hero()

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">👨‍💻 Developer Information</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="line-height:2;">
        <b style="color:#FFD700;">Project:</b> AI-Based Student Performance Prediction System<br>
        <b style="color:#FFD700;">Architecture:</b> Hybrid Stacking Ensemble Learning<br>
        <b style="color:#FFD700;">Frontend:</b> Streamlit (Custom Glassmorphism UI)<br>
        <b style="color:#FFD700;">Visualization:</b> Plotly<br>
        <b style="color:#FFD700;">Backend / ML:</b> Python, Scikit-learn, XGBoost<br>
        <b style="color:#FFD700;">Deployment Ready:</b> Streamlit Cloud / Docker / On-Prem<br>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">🛠️ Tech Stack</div>', unsafe_allow_html=True)
    tech_cols = st.columns(4)
    tech_items = [
        ("🐍", "Python 3.10+"),
        ("⚡", "Streamlit"),
        ("🌲", "XGBoost"),
        ("📈", "Plotly"),
    ]
    for col, (icon, label) in zip(tech_cols, tech_items):
        with col:
            st.markdown(
                f"""
                <div class="pipeline-step">{icon}<br>{label}</div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def render_footer() -> None:
    """Render the global footer."""
    st.markdown(
        """
        <div class="footer">
            © 2026 AI Student Performance Prediction System · Built with Streamlit, Plotly & Scikit-learn<br>
            Powered by Hybrid Stacking Ensemble Learning
        </div>
        """,
        unsafe_allow_html=True,
    )


# ======================================================================
# MAIN APPLICATION
# ======================================================================

def main() -> None:
    """Application entry point."""
    inject_custom_css()
    page = render_sidebar()

    if page == "🏠 Predict":
        render_hero()
        inputs = render_prediction_form()

        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        predict_clicked = st.button("🔮 Predict Performance", use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

        if predict_clicked:
            try:
                with st.spinner("Running inference through the Hybrid Stacking Ensemble..."):
                    result = run_inference(
                        inputs["attendance"], inputs["hours_studied"], inputs["previous_score"], inputs["sleeping_hours"], inputs["tutoring_sessions"]
                    )
                render_result_section(result, inputs)
            except FileNotFoundError as err:
                st.error(str(err))

        render_pipeline_diagram()

    elif page == "📊 About Project":
        render_about_section()

    elif page == "👨‍💻 Developer Info":
        render_developer_info()

    render_footer()


if __name__ == "__main__":
    main()
