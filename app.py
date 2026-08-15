import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Environmental & Pollution Risk Screening",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ============================================================
# LOAD EXTERNAL CSS
# ============================================================

css_path = Path(__file__).with_name("style.css")
with open(css_path, "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



# ============================================================
# LOAD EXTERNAL CSS
# ============================================================

css_path = Path(__file__).with_name("style.css")

with open(css_path, "r", encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ============================================================
# LOAD MODEL FILES
# ============================================================

model = joblib.load("final_random_forest_model.pkl")
# scaler = joblib.load("scaler.pkl")
selected_features = list(joblib.load("model_features.pkl"))


# ============================================================
# SESSION STATE
# ============================================================

if "step" not in st.session_state:
    st.session_state.step = 1

if "input_values" not in st.session_state:
    st.session_state.input_values = {}

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ============================================================
# CUSTOM CSS
# ============================================================

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    '<div class="sidebar-title">🌍 Environmental<br>Analytics</div>',
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔬 CIscore Prediction",
        "ℹ️ About the Model",
    ],
)

st.sidebar.divider()

st.sidebar.markdown("**Model**")
st.sidebar.write("Tuned Random Forest Regressor")

st.sidebar.markdown("**Target**")
st.sidebar.write("CIscore")

st.sidebar.markdown("**Selected Features**")
st.sidebar.write(str(len(selected_features)))


# ============================================================
# FEATURE GROUPING
#
# The screenshots supplied with the project show these groups:
# Pollution, Population, Socioeconomic, Health and Other.
#
# The four-step application requested by the user combines:
#   1. Environmental & Pollution
#   2. Population & Socioeconomic
#   3. Health
#   4. Other
# ============================================================

# Exact/known names visible in the supplied screenshots.
# These mappings take priority over keyword matching.

pollution_exact = {
    "Pollution",
    "PollutionScore",
    "PollutionP",
    "lead",
    "leadP",
    "pm",
    "pmP",
    "dieselP",
}

population_exact = {
    "PopChar",
    "PopCharScore",
    "PopCharP",
    "Hispanic.pct",
    "White.pct",
    "White",
    "Hispanic",
}

socioeconomic_exact = {
    "eduP",
    "edu",
    "popP",
    "pop",
    "povP",
    "pov",
    "housingBP",
    "housingB",
}

health_exact = {
    "asthmaP",
    "asthma",
    "cvdP",
    "cvd",
    "lbwP",
    "lbw",
}

other_exact = {
    "lingP",
    "ling",
    "unempP",
    "unemp",
    "drinkP",
    "drink",
    "RSEIhazP",
    "Elderly_65over.pct",
    "cleanupsP",
}


def classify_feature(feature):
    """
    Classify a selected feature for the website.

    Exact names from the supplied screenshots are handled first.
    Remaining features use conservative keyword matching.
    """

    name = str(feature)
    lower = name.lower()

    if name in pollution_exact:
        return "environmental"

    if name in population_exact:
        return "population"

    if name in socioeconomic_exact:
        return "population"

    if name in health_exact:
        return "health"

    if name in other_exact:
        return "other"

    # ---- Pollution / environmental fallback ----
    pollution_words = [
        "pollution",
        "pm2",
        "pm",
        "ozone",
        "traffic",
        "diesel",
        "pest",
        "pesticide",
        "lead",
        "tox",
        "groundwater",
        "impaired",
        "solid",
        "waste",
        "hazard",
        "air",
    ]

    # ---- Health fallback ----
    health_words = [
        "asthma",
        "cvd",
        "cardio",
        "cardiovascular",
        "lbw",
        "birth",
        "health",
    ]

    # ---- Population / socioeconomic fallback ----
    population_words = [
        "popchar",
        "hispanic",
        "white",
        "poverty",
        "pov",
        "education",
        "edu",
        "unemployment",
        "unemp",
        "housing",
        "linguistic",
        "ling",
        "isolation",
        "income",
        "race",
    ]

    if any(word in lower for word in health_words):
        return "health"

    if any(word in lower for word in pollution_words):
        return "environmental"

    if any(word in lower for word in population_words):
        return "population"

    return "other"


groups = {
    "environmental": [],
    "population": [],
    "health": [],
    "other": [],
}

for feature in selected_features:
    groups[classify_feature(feature)].append(feature)


# ============================================================
# INPUT FUNCTION
# ============================================================

def render_feature_inputs(features, key_prefix):

    if not features:
        st.info("No selected indicators were assigned to this section.")
        return

    columns = st.columns(3)

    for index, feature in enumerate(features):

        with columns[index % 3]:

            current_value = st.session_state.input_values.get(
                feature,
                0.0
            )

            # Simple text box
            value = st.text_input(
                str(feature),
                value=f"{current_value:.4f}",
                placeholder="Enter value",
                key=f"{key_prefix}_{feature}"
            )

            # Convert the entered value to numeric
            try:
                st.session_state.input_values[feature] = float(value)

            except ValueError:
                st.session_state.input_values[feature] = 0.0
# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                🌍 Environmental &amp; Pollution Risk Screening
            </div>
            <div class="hero-subtitle">
                Machine Learning based screening of environmental
                and pollution indicators
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
            <div class="card-title">About the Project</div>
            <div class="card-text">
                This project uses Machine Learning to estimate the
                <b>CIscore</b> based on selected environmental,
                pollution, population and socioeconomic indicators.
                The application provides a model-based estimate that
                can support environmental risk screening.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">🌱 Environmental</div>
                <div class="card-text">
                    Analyze environmental indicators that may
                    contribute to community risk.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">🏭 Pollution</div>
                <div class="card-text">
                    Consider pollution-related indicators when
                    estimating environmental burden.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
        """
            <div class="card">
                <div class="card-title">👥 Population</div>
                <div class="card-text">
                    Consider population and socioeconomic
                    indicators that may influence community
                    vulnerability.
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )

 

    st.markdown(
        """
        <div class="card">
            <div class="card-title">⚙️ How It Works</div>
            <div class="card-text">
                <b>1.</b> Enter environmental and pollution indicators.<br>
                <b>2.</b> Enter population and socioeconomic indicators.<br>
                <b>3.</b> Enter health indicators.<br>
                <b>4.</b> Enter the remaining selected indicators.<br>
                <b>5.</b> The trained Random Forest model processes the inputs.<br>
                <b>6.</b> The model predicts the CIscore.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "👉 Use 'CIscore Prediction' from the sidebar to make a prediction."
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "🔬 CIscore Prediction":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                🔬 Environmental Risk Prediction
            </div>
            <div class="hero-subtitle">
                Enter the required indicators to estimate the CIscore.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # STEP INFORMATION
    # --------------------------------------------------------

    step_names = {
        1: "🌱 Environmental & Pollution Indicators",
        2: "👥 Population & Socioeconomic Indicators",
        3: "🏥 Health Indicators",
        4: "📊 Other Indicators",
    }

    current_step = st.session_state.step

    st.markdown(
        f"""
        <div class="step-box">
            <div class="step-label">STEP {current_step} OF 4</div>
            <div class="step-name">{step_names[current_step]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(current_step / 4)

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    if current_step == 1:

        st.markdown(
            '<div class="section-title">🌫️ Environmental & Pollution Indicators</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Enter the environmental and pollution-related indicator values.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"**{len(groups['environmental'])} selected indicators**"
        )

        render_feature_inputs(
            groups["environmental"],
            "environmental",
        )

        st.divider()

        if st.button(
            "Continue to Population & Socioeconomic ➡️",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.step = 2
            st.rerun()

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    elif current_step == 2:

        st.markdown(
            '<div class="section-title">👥 Population & Socioeconomic Indicators</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Enter the population and socioeconomic indicator values.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"**{len(groups['population'])} selected indicators**"
        )

        render_feature_inputs(
            groups["population"],
            "population",
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "⬅️ Back to Environmental",
                use_container_width=True,
            ):
                st.session_state.step = 1
                st.rerun()

        with col2:
            if st.button(
                "Continue to Health ➡️",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.step = 3
                st.rerun()

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    elif current_step == 3:

        st.markdown(
            '<div class="section-title">🏥 Health Indicators</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Enter the selected health indicator values.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"**{len(groups['health'])} selected indicators**"
        )

        render_feature_inputs(
            groups["health"],
            "health",
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "⬅️ Back to Population",
                use_container_width=True,
            ):
                st.session_state.step = 2
                st.rerun()

        with col2:
            if st.button(
                "Continue to Other ➡️",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.step = 4
                st.rerun()

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    elif current_step == 4:

        st.markdown(
            '<div class="section-title">📊 Other Indicators</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Enter the remaining selected indicator values.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"**{len(groups['other'])} selected indicators**"
        )

        render_feature_inputs(
            groups["other"],
            "other",
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "⬅️ Back to Health",
                use_container_width=True,
            ):
                st.session_state.step = 3
                st.rerun()

        with col2:

            predict_clicked = st.button(
                "🔎 Predict CIscore",
                type="primary",
                use_container_width=True,
            )

            if predict_clicked:

                missing_features = [
                    feature
                    for feature in selected_features
                    if feature not in st.session_state.input_values
                ]

                if missing_features:

                    st.error(
                        f"{len(missing_features)} indicator value(s) "
                        "are still missing."
                    )

                    with st.expander("Show missing indicators"):
                        for feature in missing_features:
                            st.write(f"• {feature}")

                else:

                    try:

                        # Create the input dataframe in exactly the
                        # same feature order used during training.
                        input_data = pd.DataFrame(
                            [
                                {
                                    feature:
                                    st.session_state.input_values[feature]
                                    for feature in selected_features
                                }
                            ],
                            columns=selected_features,
                        )

                        # The current deployment setup uses the saved
                        # scaler because the existing app/model files
                        # were prepared with scaler.pkl.
                        input_scaled = scaler.transform(input_data)

                        predicted_ciscore = float(
                            model.predict(input_scaled)[0]
                        )

                        st.session_state.prediction = predicted_ciscore

                    except Exception as error:

                        st.error(
                            "Prediction could not be generated."
                        )

                        st.exception(error)

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if st.session_state.prediction is not None:

        prediction = st.session_state.prediction

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-caption">
                    Predicted Environmental Burden
                </div>
                <div class="result-number">
                    {prediction:.2f}
                </div>
                <div class="result-unit">
                    CIscore
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.success(
            f"Prediction generated successfully. Estimated CIscore: {prediction:.2f}"
        )

        st.write(
            "The predicted value is the output of the trained "
            "Random Forest regression model using the selected "
            "community indicators."
        )

        if st.button(
            "🔄 Start New Prediction",
            use_container_width=True,
        ):
            st.session_state.step = 1
            st.session_state.input_values = {}
            st.session_state.prediction = None
            st.rerun()


# ============================================================
# ABOUT MODEL PAGE
# ============================================================

elif page == "ℹ️ About the Model":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">
                ℹ️ About the Model
            </div>
            <div class="hero-subtitle">
                Environmental & Pollution Analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Dataset",
            "CalEnviroScreen 4.0",
        )

    with col2:
        st.metric(
            "Selected Features",
            len(selected_features),
        )

    with col3:
        st.metric(
            "Target",
            "CIscore",
        )

    st.markdown(
        """
        <div class="card">
            <div class="card-title">🎯 Feature Selection</div>
            <div class="card-text">
                Numeric features were correlated with CIscore and
                features meeting the selected correlation threshold
                were considered. CIscore-derived variables were
                removed during the target-leakage check.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
            <div class="card-title">⚙️ Feature Scaling</div>
            <div class="card-text">
                The saved scaler is applied to prediction inputs using
                the same preprocessing object used by the current
                deployment setup.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
            <div class="card-title">🌳 Random  Forest Regression</div>
            <div class="card-text">
                The final application uses the trained Random Forest
                regression model. Random Forest can capture
                non-linear relationships and interactions between
                predictors.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
            <div class="card-title">🔧 Hyperparameter Tuning</div>
            <div class="card-text">
                GridSearchCV with cross-validation was used to search
                for suitable Random Forest configurations. Parameters
                considered included max_depth, min_samples_split and
                min_samples_leaf.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("📈 Evaluation Metrics")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("MAE", "Lower is better")
        st.caption("Mean Absolute Error")

    with c2:
        st.metric("RMSE", "Lower is better")
        st.caption("Root Mean Squared Error")

    with c3:
        st.metric("R²", "Higher is better")
        st.caption("Coefficient of Determination")

    st.info(
        "The application is intended as a model-based analytical "
        "tool to support environmental monitoring and data-driven planning."
    )

