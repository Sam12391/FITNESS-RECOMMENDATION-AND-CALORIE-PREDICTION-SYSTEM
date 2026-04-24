# ============================================================
#  AI Fitness Recommendation & Calorie Prediction System
#  Built with Streamlit, Scikit-learn & Plotly
#  Author: AI Fitness Lab
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fitness System",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  – dark athletic theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── root / page ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0d0f14;
    color: #e8eaf0;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: #13161e !important;
    border-right: 1px solid #1f2330;
}
[data-testid="stSidebar"] * { color: #c8cad4 !important; }

/* ── headings ── */
h1, h2, h3 {
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 2px;
}

/* ── metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #191d28 0%, #1e2333 100%);
    border: 1px solid #2a2f42;
    border-radius: 16px;
    padding: 22px 26px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: transform .2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: #6b7191;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.6rem;
    letter-spacing: 1px;
    line-height: 1;
}
.metric-sub {
    font-size: 0.82rem;
    color: #8890aa;
    margin-top: 4px;
}

/* ── accent colours ── */
.accent-green  { color: #34d399; }
.accent-orange { color: #fb923c; }
.accent-blue   { color: #60a5fa; }
.accent-red    { color: #f87171; }
.accent-yellow { color: #fbbf24; }

/* ── info / plan boxes ── */
.plan-box {
    background: #191d28;
    border-left: 4px solid #34d399;
    border-radius: 0 12px 12px 0;
    padding: 18px 22px;
    margin: 10px 0;
    font-size: 0.95rem;
    line-height: 1.7;
}
.plan-box.orange { border-left-color: #fb923c; }
.plan-box.blue   { border-left-color: #60a5fa; }
.plan-box.yellow { border-left-color: #fbbf24; }

/* ── section header ── */
.section-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5rem;
    letter-spacing: 3px;
    color: #34d399;
    border-bottom: 1px solid #1f2a38;
    padding-bottom: 6px;
    margin: 28px 0 14px 0;
}

/* ── badge ── */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.badge-green  { background:#064e3b; color:#34d399; }
.badge-orange { background:#431407; color:#fb923c; }
.badge-red    { background:#450a0a; color:#f87171; }
.badge-blue   { background:#1e3a5f; color:#60a5fa; }

/* ── r2 bar ── */
.r2-bar-bg {
    background: #1e2333;
    border-radius: 999px;
    height: 10px;
    width: 100%;
    overflow: hidden;
    margin-top: 6px;
}
.r2-bar-fill {
    height: 10px;
    border-radius: 999px;
    background: linear-gradient(90deg, #34d399, #60a5fa);
}

/* ── tip pill ── */
.tip-pill {
    background: #191d28;
    border: 1px solid #2a2f42;
    border-radius: 10px;
    padding: 10px 16px;
    margin: 6px 0;
    font-size: 0.9rem;
}

/* ── hide streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER: BMI classification
# ─────────────────────────────────────────────
def classify_bmi(bmi: float) -> tuple[str, str, str]:
    """Return (label, badge_class, accent_class) for a BMI value."""
    if bmi < 18.5:
        return "Underweight", "badge-blue", "accent-blue"
    elif bmi < 25:
        return "Normal", "badge-green", "accent-green"
    elif bmi < 30:
        return "Overweight", "badge-orange", "accent-orange"
    else:
        return "Obese", "badge-red", "accent-red"


# ─────────────────────────────────────────────
# ML MODEL  – trained once and cached
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model():
    """
    Generate a synthetic dataset and train a LinearRegression model
    to predict daily calorie requirements.
    """
    np.random.seed(42)
    n = 2000

    age    = np.random.randint(18, 61,  n).astype(float)
    weight = np.random.uniform(40, 120, n)
    height = np.random.uniform(140, 200, n)
    gender = np.random.randint(0, 2, n)          # 0 = female, 1 = male
    # activity: 0=sedentary(1.2), 1=moderate(1.55), 2=active(1.725)
    activity = np.random.randint(0, 3, n)
    activity_mult = np.where(activity == 0, 1.2,
                    np.where(activity == 1, 1.55, 1.725))

    # BMR via Mifflin-St Jeor
    bmr = np.where(
        gender == 1,
        10*weight + 6.25*height - 5*age + 5,    # male
        10*weight + 6.25*height - 5*age - 161   # female
    )
    calories = bmr * activity_mult + np.random.normal(0, 30, n)  # small noise

    df = pd.DataFrame({
        "age": age, "weight": weight, "height": height,
        "gender": gender, "activity": activity, "calories": calories
    })

    X = df[["age", "weight", "height", "gender", "activity"]]
    y = df["calories"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    return model, r2


# ─────────────────────────────────────────────
# BMI GAUGE  (Plotly speedometer)
# ─────────────────────────────────────────────
def bmi_gauge(bmi_val: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bmi_val,
        number={"font": {"size": 38, "family": "Bebas Neue", "color": "#e8eaf0"},
                "suffix": " kg/m²"},
        gauge={
            "axis": {
                "range": [10, 40],
                "tickwidth": 1,
                "tickcolor": "#3a3f52",
                "tickfont": {"color": "#6b7191", "size": 11},
            },
            "bar": {"color": "#34d399", "thickness": 0.28},
            "bgcolor": "#191d28",
            "borderwidth": 0,
            "steps": [
                {"range": [10, 18.5], "color": "#1e3a5f"},   # underweight – blue
                {"range": [18.5, 25], "color": "#064e3b"},   # normal – green
                {"range": [25, 30],   "color": "#431407"},   # overweight – orange
                {"range": [30, 40],   "color": "#450a0a"},   # obese – red
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 3},
                "thickness": 0.85,
                "value": bmi_val,
            },
        },
        title={"text": "YOUR BMI STATUS",
               "font": {"size": 13, "family": "DM Sans",
                        "color": "#6b7191"},
               "align": "center"},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        paper_bgcolor="#0d0f14",
        font_color="#e8eaf0",
        height=280,
        margin=dict(t=50, b=10, l=20, r=20),
    )
    # Legend annotations
    for x, label, color in [
        (0.10, "Under-\nweight", "#60a5fa"),
        (0.35, "Normal",        "#34d399"),
        (0.65, "Over-\nweight", "#fb923c"),
        (0.88, "Obese",         "#f87171"),
    ]:
        fig.add_annotation(
            x=x, y=-0.08, xref="paper", yref="paper",
            text=f"<b style='color:{color}'>{label}</b>",
            showarrow=False,
            font=dict(size=9, color=color),
            align="center",
        )
    return fig


# ─────────────────────────────────────────────
# WORKOUT & DIET CONTENT
# ─────────────────────────────────────────────
WORKOUT = {
    "Fat Loss": {
        "icon": "🔥",
        "exercises": ["Running / Outdoor Jogging", "Cycling (stationary or road)",
                      "HIIT Circuit Training", "Jump Rope", "Swimming"],
        "frequency": "4 – 5 days / week",
        "tip": "Keep heart rate at 65–85 % of max during cardio sessions.",
    },
    "Muscle Gain": {
        "icon": "💪",
        "exercises": ["Barbell Squats & Deadlifts", "Bench Press & Overhead Press",
                      "Pull-ups & Rows", "Progressive Overload Lifting", "Compound Movements"],
        "frequency": "4 – 6 days / week",
        "tip": "Increase weight or reps every 1–2 weeks to ensure progressive overload.",
    },
    "Maintain": {
        "icon": "⚖️",
        "exercises": ["Moderate Cardio (30 min)", "Bodyweight Circuits",
                      "Yoga / Mobility Work", "Light Dumbbell Training", "Recreational Sports"],
        "frequency": "3 – 5 days / week",
        "tip": "Consistency matters more than intensity when maintaining fitness.",
    },
}

DIET = {
    "Fat Loss": {
        "icon": "🥗",
        "strategy": "Calorie Deficit  (−300 to −500 kcal/day)",
        "tips": [
            "🥩 High protein: 1.6–2 g per kg bodyweight",
            "🚫 Minimise added sugars & refined carbs",
            "🥦 Fill half your plate with vegetables",
            "💧 Drink 2.5–3 L of water daily",
            "🕗 Avoid eating 2 hrs before bed",
        ],
    },
    "Muscle Gain": {
        "icon": "🍗",
        "strategy": "Calorie Surplus  (+200 to +400 kcal/day)",
        "tips": [
            "🥩 Protein-rich meals every 3–4 hours",
            "🍚 Complex carbs: oats, brown rice, sweet potato",
            "🥑 Healthy fats from nuts, avocado, olive oil",
            "🥛 Post-workout shake within 45 min",
            "😴 Sleep 7–9 hrs – growth hormone peaks at night",
        ],
    },
    "Maintain": {
        "icon": "🥙",
        "strategy": "Balanced Calories  (match TDEE)",
        "tips": [
            "⚖️ Equal balance of proteins, carbs & fats",
            "🥦 Eat a rainbow of vegetables & fruits",
            "🌾 Prefer whole grains over refined carbs",
            "🐟 Include fatty fish 2× per week (omega-3)",
            "☕ Limit alcohol & ultra-processed foods",
        ],
    },
}


# ─────────────────────────────────────────────
# SIDEBAR  – USER INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <div style='font-family:"Bebas Neue",sans-serif; font-size:1.8rem;
                    letter-spacing:3px; color:#34d399;'>🏋️ AI FITNESS</div>
        <div style='font-size:0.75rem; color:#6b7191; letter-spacing:2px;
                    text-transform:uppercase;'>Calorie Prediction System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 👤 Personal Details")
    age    = st.slider("Age",    18, 80, 25)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    height = st.slider("Height (cm)", 140, 220, 170)
    weight = st.slider("Weight (kg)",  40, 150,  70)

    st.markdown("---")
    st.markdown("##### 🎯 Fitness Profile")
    activity = st.selectbox("Activity Level",
                            ["Sedentary", "Moderate", "Active"])
    goal     = st.selectbox("Fitness Goal",
                            ["Fat Loss", "Muscle Gain", "Maintain"])

    st.markdown("---")
    predict_btn = st.button("⚡  Generate My Plan", use_container_width=True)

    st.markdown("""
    <div style='margin-top:24px; padding:14px; background:#0d0f14;
                border-radius:10px; border:1px solid #1f2330; font-size:0.78rem;
                color:#6b7191; line-height:1.6;'>
        <b style='color:#34d399;'>ℹ️ About the ML Model</b><br>
        Linear Regression trained on 2 000 synthetic records using age, weight,
        height, gender, and activity level to predict daily calorie requirements.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD / TRAIN MODEL
# ─────────────────────────────────────────────
with st.spinner("🤖 Training AI model on synthetic dataset…"):
    model, r2 = train_model()


# ─────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────

# ── Hero Title ──────────────────────────────
st.markdown("""
<div style='padding: 10px 0 4px 0;'>
    <h1 style='font-family:"Bebas Neue",sans-serif; font-size:3rem;
               letter-spacing:4px; margin:0; line-height:1;'>
        🏋️ AI Fitness Recommendation
        <span style='color:#34d399;'>&</span> Calorie Prediction
    </h1>
    <p style='color:#6b7191; font-size:0.92rem; margin-top:6px; letter-spacing:1px;'>
        Powered by Machine Learning  ·  Mifflin-St Jeor Equation  ·  Plotly Visualisation
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# CALCULATIONS
# ─────────────────────────────────────────────
height_m = height / 100
bmi = weight / (height_m ** 2)

if gender == "Male":
    bmr = 10 * weight + 6.25 * height - 5 * age + 5
else:
    bmr = 10 * weight + 6.25 * height - 5 * age - 161

activity_map = {"Sedentary": 1.2, "Moderate": 1.55, "Active": 1.725}
tdee_formula = bmr * activity_map[activity]

# ML prediction
gender_enc   = 1 if gender == "Male" else 0
activity_enc = {"Sedentary": 0, "Moderate": 1, "Active": 2}[activity]
X_input      = np.array([[age, weight, height, gender_enc, activity_enc]])
ml_calories  = float(model.predict(X_input)[0])

bmi_label, badge_cls, accent_cls = classify_bmi(bmi)


# ─────────────────────────────────────────────
# SECTION 1 – Health Summary
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📊 YOUR HEALTH SUMMARY</div>',
            unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Body Mass Index</div>
        <div class="metric-value {accent_cls}">{bmi:.1f}</div>
        <div class="metric-sub">
            <span class="badge {badge_cls}">{bmi_label}</span>
        </div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Basal Metabolic Rate</div>
        <div class="metric-value accent-yellow">{bmr:.0f}</div>
        <div class="metric-sub">kcal / day (at rest)</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Formula TDEE</div>
        <div class="metric-value accent-orange">{tdee_formula:.0f}</div>
        <div class="metric-sub">kcal / day</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🤖 ML Prediction</div>
        <div class="metric-value accent-green">{ml_calories:.0f}</div>
        <div class="metric-sub">kcal / day</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SECTION 2 – BMI Gauge + Interpretation
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📈 BMI VISUALISATION</div>',
            unsafe_allow_html=True)

gcol, icol = st.columns([1.1, 0.9])

with gcol:
    st.plotly_chart(bmi_gauge(bmi), use_container_width=True, config={"displayModeBar": False})

with icol:
    # Interpretation message
    if bmi < 18.5:
        interp_msg = "You are <b class='accent-blue'>underweight</b>. Focus on a calorie surplus with nutrient-dense foods and strength training to build lean mass."
        interp_icon = "⚠️"
    elif bmi < 25:
        interp_msg = "You are in a <b class='accent-green'>healthy range</b>. Maintain your current habits with a balanced diet and consistent exercise."
        interp_icon = "✅"
    elif bmi < 30:
        interp_msg = "You are <b class='accent-orange'>overweight</b>. A moderate calorie deficit combined with regular cardio can help achieve a healthier weight."
        interp_icon = "⚠️"
    else:
        interp_msg = "Your BMI falls in the <b class='accent-red'>obese</b> category. It is recommended to consult a healthcare professional alongside following this plan."
        interp_icon = "🚨"

    st.markdown(f"""
    <div style='margin-top:30px;'>
        <div style='font-family:"Bebas Neue",sans-serif; font-size:1.1rem;
                    letter-spacing:2px; color:#6b7191; margin-bottom:10px;'>
            {interp_icon} BMI INTERPRETATION
        </div>
        <div class='plan-box' style='font-size:1rem; line-height:1.75;'>
            {interp_msg}
        </div>
    </div>

    <div style='margin-top:22px;'>
        <div style='font-family:"Bebas Neue",sans-serif; font-size:1.1rem;
                    letter-spacing:2px; color:#6b7191; margin-bottom:10px;'>
            🤖 MODEL PERFORMANCE
        </div>
        <div class='plan-box blue'>
            <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
                <span>R² Score</span>
                <b class='accent-blue'>{r2:.4f}</b>
            </div>
            <div class='r2-bar-bg'>
                <div class='r2-bar-fill' style='width:{r2*100:.1f}%'></div>
            </div>
            <div style='font-size:0.8rem; color:#6b7191; margin-top:8px;'>
                Model explains <b>{r2*100:.1f}%</b> of variance in calorie requirements.
                Trained on 1 600 records · Tested on 400 records.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 3 – Recommended Plan
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">🎯 RECOMMENDED PLAN</div>',
            unsafe_allow_html=True)

# Adjust target calories based on goal
if goal == "Fat Loss":
    target_cal  = ml_calories - 400
    cal_note    = "−400 kcal deficit applied"
    cal_accent  = "accent-orange"
elif goal == "Muscle Gain":
    target_cal  = ml_calories + 300
    cal_note    = "+300 kcal surplus applied"
    cal_accent  = "accent-green"
else:
    target_cal  = ml_calories
    cal_note    = "Maintenance calories"
    cal_accent  = "accent-blue"

pcol1, pcol2, pcol3 = st.columns(3)

with pcol1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Target Calories</div>
        <div class="metric-value {cal_accent}">{target_cal:.0f}</div>
        <div class="metric-sub">{cal_note}</div>
    </div>""", unsafe_allow_html=True)

with pcol2:
    protein_g = round(weight * 1.8)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Daily Protein</div>
        <div class="metric-value accent-yellow">{protein_g}</div>
        <div class="metric-sub">grams / day (1.8 g/kg)</div>
    </div>""", unsafe_allow_html=True)

with pcol3:
    water_l = round(weight * 0.033, 1)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Daily Water</div>
        <div class="metric-value accent-blue">{water_l}</div>
        <div class="metric-sub">litres / day</div>
    </div>""", unsafe_allow_html=True)

# Safe weight change message
st.markdown(f"""
<div style='background:linear-gradient(135deg,#064e3b,#065f46); border-radius:12px;
            padding:16px 22px; margin:18px 0; border:1px solid #059669;'>
    <span style='font-size:1.2rem;'>📅</span>
    <b style='color:#34d399; margin-left:8px;'>Estimated Progress</b>
    <span style='color:#a7f3d0; margin-left:6px; font-size:0.92rem;'>
        If you follow this plan, your estimated safe weight change is
        <b>~0.5 – 1 kg per week</b>. Consistency over 8–12 weeks yields the best results.
    </span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 4 – Workout Routine
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">🏋️ WORKOUT ROUTINE</div>',
            unsafe_allow_html=True)

wdata = WORKOUT[goal]
wcol1, wcol2 = st.columns([1, 1])

with wcol1:
    exercises_html = "".join(
        f"<div class='tip-pill'>{wdata['icon']} {ex}</div>"
        for ex in wdata["exercises"]
    )
    st.markdown(f"""
    <div class='plan-box orange'>
        <b>Recommended Exercises</b><br><br>
        {exercises_html}
    </div>""", unsafe_allow_html=True)

with wcol2:
    st.markdown(f"""
    <div class='plan-box'>
        <b>📆 Frequency</b><br>
        <span style='font-family:"Bebas Neue",sans-serif; font-size:1.6rem;
                     color:#34d399; letter-spacing:2px;'>{wdata['frequency']}</span>
        <br><br>
        <b>💡 Pro Tip</b><br>
        <span style='color:#8890aa; font-size:0.92rem;'>{wdata['tip']}</span>
        <br><br>
        <b>🎯 Your Goal</b><br>
        <span class='badge {"badge-orange" if goal=="Fat Loss" else "badge-green" if goal=="Muscle Gain" else "badge-blue"}'>{goal}</span>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 5 – Diet Tips
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">🥗 DIET TIPS & NUTRITION</div>',
            unsafe_allow_html=True)

ddata = DIET[goal]
dcol1, dcol2 = st.columns([1, 1])

with dcol1:
    tips_html = "".join(
        f"<div class='tip-pill'>{tip}</div>"
        for tip in ddata["tips"]
    )
    st.markdown(f"""
    <div class='plan-box yellow'>
        <b>{ddata['icon']} Nutrition Strategy</b><br>
        <span style='color:#fbbf24; font-size:0.88rem;'>{ddata['strategy']}</span>
        <br><br>
        {tips_html}
    </div>""", unsafe_allow_html=True)

with dcol2:
    # Simple macro breakdown
    p_cal = protein_g * 4
    if goal == "Fat Loss":
        carb_pct, fat_pct = 0.35, 0.25
    elif goal == "Muscle Gain":
        carb_pct, fat_pct = 0.45, 0.20
    else:
        carb_pct, fat_pct = 0.40, 0.25
    carb_g = round((target_cal * carb_pct) / 4)
    fat_g  = round((target_cal * fat_pct)  / 9)

    fig_macro = go.Figure(go.Pie(
        labels=["Protein", "Carbohydrates", "Fat"],
        values=[protein_g * 4, carb_g * 4, fat_g * 9],
        hole=0.55,
        marker=dict(colors=["#34d399", "#60a5fa", "#fb923c"]),
        textinfo="label+percent",
        textfont=dict(size=12, family="DM Sans", color="#e8eaf0"),
    ))
    fig_macro.update_layout(
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#0d0f14",
        font_color="#e8eaf0",
        height=260,
        margin=dict(t=20, b=10, l=10, r=10),
        showlegend=False,
        annotations=[dict(
            text=f"<b>{target_cal:.0f}</b><br>kcal",
            x=0.5, y=0.5, font_size=15, showarrow=False,
            font_color="#e8eaf0", font_family="Bebas Neue",
        )],
    )
    st.markdown("<div style='margin-top:8px; color:#6b7191; font-size:0.78rem;"
                "text-transform:uppercase; letter-spacing:2px;'>"
                "📊 Macro Distribution</div>", unsafe_allow_html=True)
    st.plotly_chart(fig_macro, use_container_width=True,
                    config={"displayModeBar": False})
    st.markdown(f"""
    <div style='display:flex; gap:12px; justify-content:center; margin-top:-10px;'>
        <div class='tip-pill' style='flex:1; text-align:center;'>
            🥩 <b>{protein_g}g</b><br><span style='color:#6b7191;font-size:0.78rem;'>Protein</span>
        </div>
        <div class='tip-pill' style='flex:1; text-align:center;'>
            🍚 <b>{carb_g}g</b><br><span style='color:#6b7191;font-size:0.78rem;'>Carbs</span>
        </div>
        <div class='tip-pill' style='flex:1; text-align:center;'>
            🥑 <b>{fat_g}g</b><br><span style='color:#6b7191;font-size:0.78rem;'>Fat</span>
        </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#3a3f52; font-size:0.78rem;
            letter-spacing:1px; padding: 8px 0 16px 0;'>
    AI Fitness Recommendation &amp; Calorie Prediction System &nbsp;·&nbsp;
    Built with Streamlit, Scikit-learn &amp; Plotly &nbsp;·&nbsp;
    <span style='color:#34d399;'>ML College Project</span>
</div>
""", unsafe_allow_html=True)
