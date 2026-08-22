import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


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
# LOAD DATASET
# ============================================================

BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "CalEnviroScreen_4_0_Results__-6724038633628611498.csv"

try:
    df = pd.read_csv(DATASET_PATH)
except Exception as error:
    st.error(
        "Dataset could not be loaded. Please keep the CSV file in the "
        "same folder as app.py."
    )
    st.exception(error)
    st.stop()


# ============================================================
# LOAD EXTERNAL CSS
# ============================================================

css_path = Path(__file__).with_name("style.css")

with open(css_path, "r", encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD MODEL FILES
# ============================================================

model = joblib.load("final_random_forest_model.pkl")
# scaler = joblib.load("scaler.pkl")
selected_features = list(joblib.load("model_features.pkl"))
test_data_path = Path(__file__).with_name(
    "test_data.pkl"
)

test_df = joblib.load(
    test_data_path
)

# ============================================================
# SESSION STATE
# ============================================================

if "step" not in st.session_state:
    st.session_state.step = 1

if "input_values" not in st.session_state:
    st.session_state.input_values = {}

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "dataset_row_loaded" not in st.session_state:
    st.session_state.dataset_row_loaded = False

if "actual_ciscore" not in st.session_state:
    st.session_state.actual_ciscore = None


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

            saved_value = st.session_state.input_values.get(feature, "")

            value = st.text_input(
                str(feature),
                value=str(saved_value),
                placeholder="Enter value",
                key=f"{key_prefix}_{feature}",
            )

            if value.strip():
                try:
                    st.session_state.input_values[feature] = float(value)
                except ValueError:
                    st.session_state.input_values.pop(feature, None)
            else:
                st.session_state.input_values.pop(feature, None)


def get_missing_features(features):
    return [
        feature
        for feature in features
        if (
            feature not in st.session_state.input_values
            or str(st.session_state.input_values.get(feature, "")).strip() == ""
        )
    ]


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

    st.header("📊 CIscore Prediction")

    st.subheader("Choose Input Method")

    input_method = st.radio(
        "How would you like to enter community information?",
        [
            "✏️ Manual Entry",
            "📂 Select from Dataset",
             "🧪 Test 50 Samples",
        ],
        horizontal=True,
    )

    # ========================================================
    # SELECT FROM DATASET
    # ========================================================

    if input_method == "📂 Select from Dataset":

        st.subheader("📂 Select Community from Dataset")

        st.write(
            "Select a row number from the CalEnviroScreen dataset "
            "to automatically load all selected indicator values."
        )

        row_number = st.number_input(
            "Select Dataset Row Number",
            min_value=1,
            max_value=len(df),
            value=1,
            step=1,
        )

        if st.button(
            "📥 Load Community Data",
            type="primary",
            use_container_width=True,
        ):

            selected_row = df.iloc[int(row_number) - 1]

            # Get the actual CIscore from the selected dataset row.
            if "CIscore" in df.columns:
                try:
                    st.session_state.actual_ciscore = float(
                        selected_row["CIscore"]
                    )
                except (ValueError, TypeError):
                    st.session_state.actual_ciscore = None
            else:
                st.session_state.actual_ciscore = None

            loaded_features = []
            missing_features = []

            for feature in selected_features:

                if feature not in df.columns:
                    missing_features.append(feature)
                    continue

                value = selected_row[feature]

                if pd.isna(value):
                    missing_features.append(feature)
                    continue

                try:
                    numeric_value = float(value)
                    st.session_state.input_values[feature] = numeric_value
                    loaded_features.append(feature)

                except (ValueError, TypeError):
                    missing_features.append(feature)

            st.session_state.dataset_row_loaded = True

            if missing_features:

                st.warning(
                    f"{len(loaded_features)} indicators loaded successfully. "
                    f"{len(missing_features)} indicators could not be loaded."
                )

                with st.expander("Show unavailable indicators"):
                    for feature in missing_features:
                        st.write(f"• {feature}")

            else:

                st.success(
                    f"✅ All {len(loaded_features)} selected indicators "
                    f"were loaded from dataset row {int(row_number)}."
                )

            st.rerun()

        # Show all loaded values in the text fields.
        if st.session_state.dataset_row_loaded:

            st.markdown(
                """
                <div class="hero">
                    <div class="hero-title">
                        🔬 Loaded Community Indicators
                    </div>
                    <div class="hero-subtitle">
                        Values below were loaded from the selected
                        CalEnviroScreen dataset row.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("### 🌫️ Environmental & Pollution Indicators")
            render_feature_inputs(
                groups["environmental"],
                "dataset_environmental",
            )

            st.markdown("### 👥 Population & Socioeconomic Indicators")
            render_feature_inputs(
                groups["population"],
                "dataset_population",
            )

            st.markdown("### 🏥 Health Indicators")
            render_feature_inputs(
                groups["health"],
                "dataset_health",
            )

            st.markdown("### 📊 Other Indicators")
            render_feature_inputs(
                groups["other"],
                "dataset_other",
            )

            st.divider()

            if st.button(
                "🔎 Predict CIscore",
                type="primary",
                use_container_width=True,
            ):

                missing_features = get_missing_features(selected_features)

                if missing_features:

                    st.error(
                        "Please make sure all indicators have valid values "
                        "before prediction."
                    )

                    with st.expander("Show missing indicators"):
                        for feature in missing_features:
                            st.write(f"• {feature}")

                else:

                    try:

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

                        predicted_ciscore = float(
                            model.predict(input_data)[0]
                        )

                        st.session_state.prediction = predicted_ciscore

                        # ====================================================
                        # ACTUAL VS PREDICTED CIscore GRAPH
                        # ====================================================

                        actual_ciscore = st.session_state.actual_ciscore

                        if actual_ciscore is not None:

                            st.markdown(
                                "### 📈 Actual vs Predicted CIscore"
                            )

                            comparison = pd.DataFrame(
                                {
                                    "Type": [
                                        "Actual CIscore",
                                        "Predicted CIscore",
                                    ],
                                    "CIscore": [
                                        actual_ciscore,
                                        predicted_ciscore,
                                    ],
                                }
                            )

                            fig, ax = plt.subplots(figsize=(8, 4))

                            bars = ax.bar(
                                comparison["Type"],
                                comparison["CIscore"],
                            )

                            ax.set_ylabel("CIscore")
                            ax.set_title(
                                "Actual vs Predicted CIscore"
                            )

                            ax.bar_label(
                                bars,
                                fmt="%.2f",
                                padding=3,
                            )

                            maximum = float(comparison["CIscore"].max())
                            ax.set_ylim(
                                0,
                                maximum * 1.20 if maximum > 0 else 1,
                            )

                            plt.tight_layout()
                            st.pyplot(fig)

                            difference = (
                                predicted_ciscore - actual_ciscore
                            )

                            c1, c2, c3 = st.columns(3)

                            with c1:
                                st.metric(
                                    "Actual CIscore",
                                    f"{actual_ciscore:.2f}",
                                )

                            with c2:
                                st.metric(
                                    "Predicted CIscore",
                                    f"{predicted_ciscore:.2f}",
                                )

                            with c3:
                                st.metric(
                                    "Difference",
                                    f"{difference:.2f}",
                                )

                        else:

                            st.warning(
                                "Actual CIscore is not available in the "
                                "selected dataset row."
                            )

                    except Exception as error:

                        st.error("Prediction could not be generated.")
                        st.exception(error)

    # ========================================================
    # MANUAL ENTRY
    # ========================================================

    elif input_method == "✏️ Manual Entry":

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

        # ----------------------------------------------------
        # STEP 1
        # ----------------------------------------------------

        if current_step == 1:

            st.markdown(
                '<div class="section-title">'
                '🌫️ Environmental & Pollution Indicators'
                '</div>',
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

                missing = get_missing_features(
                    groups["environmental"]
                )

                if missing:

                    st.error(
                        "Please fill in all Environmental Indicators "
                        "before continuing."
                    )

                    with st.expander("Show missing indicators"):
                        for feature in missing:
                            st.write(f"• {feature}")

                else:

                    st.session_state.step = 2
                    st.rerun()

        # ----------------------------------------------------
        # STEP 2
        # ----------------------------------------------------

        elif current_step == 2:

            st.markdown(
                '<div class="section-title">'
                '👥 Population & Socioeconomic Indicators'
                '</div>',
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

                    missing = get_missing_features(
                        groups["population"]
                    )

                    if missing:

                        st.error(
                            "Please fill in all Population & Socioeconomic "
                            "Indicators before continuing."
                        )

                        with st.expander("Show missing indicators"):
                            for feature in missing:
                                st.write(f"• {feature}")

                    else:

                        st.session_state.step = 3
                        st.rerun()

        # ----------------------------------------------------
        # STEP 3
        # ----------------------------------------------------

        elif current_step == 3:

            st.markdown(
                '<div class="section-title">'
                '🏥 Health Indicators'
                '</div>',
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

                    missing = get_missing_features(
                        groups["health"]
                    )

                    if missing:

                        st.error(
                            "Please fill in all Health Indicators "
                            "before continuing."
                        )

                        with st.expander("Show missing indicators"):
                            for feature in missing:
                                st.write(f"• {feature}")

                    else:

                        st.session_state.step = 4
                        st.rerun()

        # ----------------------------------------------------
        # STEP 4
        # ----------------------------------------------------

        elif current_step == 4:

            st.markdown(
                '<div class="section-title">'
                '📊 Other Indicators'
                '</div>',
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

                if st.button(
                    "🔎 Predict CIscore",
                    type="primary",
                    use_container_width=True,
                ):

                    # Final validation: every selected feature is required.
                    missing_features = get_missing_features(
                        selected_features
                    )

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

                            predicted_ciscore = float(
                                model.predict(input_data)[0]
                            )

                            st.session_state.prediction = predicted_ciscore

                        except Exception as error:

                            st.error(
                                "Prediction could not be generated."
                            )

                            st.exception(error)

    elif input_method == "🧪 Test 50 Samples":

        st.subheader("🧪 Test Model Using Unseen Test Data")

        st.write(
        "The following test samples are taken from the "
        "20% testing dataset that was not used during model training."
        )

        st.info(
        f"Available testing samples: {len(test_df)}"
        )

        if st.button(
            "🎲 Select Random 50 Test Samples",
            type="primary",
            use_container_width=True
        ):

            test_50 = test_df.sample(
                n=50,
                random_state=42
            )

            X_test_50 = test_50[selected_features]

            y_actual = test_50["CIscore"]

            y_predicted = model.predict(
            X_test_50
            )

            results = pd.DataFrame({
                "Actual CIscore": y_actual.values,
                "Predicted CIscore": y_predicted
            })

            results["Difference"] = (
                results["Predicted CIscore"]
                - results["Actual CIscore"]
            )

            st.session_state.test_results = results 
    if "test_results" in st.session_state:

        results = st.session_state.test_results

        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score
        )

        actual = results["Actual CIscore"]

        predicted = results["Predicted CIscore"]

        mae = mean_absolute_error(
            actual,
            predicted
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        )

        r2 = r2_score(
            actual,
            predicted
        )

        st.subheader("📈 Testing Performance")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "MAE",
                f"{mae:.4f}"
            )

        with col2:
            st.metric(
                "RMSE",
                f"{rmse:.4f}"
            )

        with col3:
            st.metric(
                "R²",
                f"{r2:.4f}"
            )

        st.subheader(
            "📋 Actual vs Predicted CIscore"
        )

        st.dataframe(
            results,
            use_container_width=True
        )
        
        st.subheader(
            "📊 Actual vs Predicted CIscore"
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            range(1, len(actual) + 1),
            actual.values,
            marker="o",
            label="Actual CIscore"
        )

        ax.plot(
            range(1, len(predicted) + 1),
            predicted.values,
            marker="x",
            label="Predicted CIscore"
        )

        ax.set_xlabel("Testing Sample")
        ax.set_ylabel("CIscore")

        ax.set_title(
            "Actual vs Predicted CIscore - 50 Test Samples"
        )

        ax.legend()

        ax.grid(True)

        st.pyplot(fig)

# ============================================================
# RESULT
# ============================================================

if (
    page == "🔬 CIscore Prediction"
    and st.session_state.prediction is not None
):

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
        f"Prediction generated successfully. Estimated CIscore: "
        f"{prediction:.2f}"
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
        st.session_state.dataset_row_loaded = False
        st.session_state.actual_ciscore = None
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
                The current application passes the selected indicator
                values directly to the trained Random Forest model.
                No scaler is applied in this deployment.
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
