import streamlit as st
import joblib
import numpy as np
import pandas as pd
import io
import plotly.express as px
import time

# Set page configuration
st.set_page_config(page_title="Certificate Classifier", page_icon="📜", layout="wide")

# Load trained model
try:
    model = joblib.load("model_trainer/certificate_model.pkl")
    st.success("ML model loaded successfully!", icon="✅")
except Exception as e:
    st.error(f"Failed to load model: {str(e)}", icon="❌")
    st.stop()

# Initialize session state for input persistence
if "features" not in st.session_state:
    st.session_state.features = [0.5] * 11
if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ---- Helper Functions ----
def predict_certificate(data):
    """Predict certificate status using the loaded model."""
    try:
        data_array = np.array(data).reshape(1, -1)
        prediction = model.predict(data_array)[0]
        return "Pass" if prediction == 1 else "Fail"
    except Exception as e:
        return f"Error: {str(e)}"


def validate_features(features):
    """Validate that all features are between 0 and 1."""
    return all(0 <= f <= 1 for f in features)


# ---- Custom CSS for Styling ----
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stNumberInput input {
        border: 2px solid #e0e0e0;
        border-radius: 5px;
    }
    .stNumberInput input:focus {
        border-color: #4CAF50;
    }
    .result-box {
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
    }
    .pass {
        background-color: #d4edda;
        color: #155724;
    }
    .fail {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Streamlit UI ----
st.title("📜 Certificate Classifier Web App")
st.markdown(
    "Classify certificates interactively using a machine learning model. Enter feature values manually or upload a CSV for batch processing.")

# Tabs for Manual Input and CSV Upload
tab1, tab2 = st.tabs(["Manual Input", "CSV Upload"])

# Tab 1: Manual Input
with tab1:
    st.header("Manual Certificate Evaluation")
    st.markdown("Enter 11 feature values (0 to 1) to evaluate a single certificate.")

    # Create 11 input fields in a grid layout
    cols = st.columns(4)
    features = []
    for i in range(11):
        with cols[i % 4]:
            value = st.number_input(
                f"Feature {i + 1}",
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                value=st.session_state.features[i],
                key=f"feature_{i}",
                help=f"Enter a value between 0 and 1 for Feature {i + 1}"
            )
            features.append(value)

    # Update session state
    st.session_state.features = features

    # Real-time validation feedback
    if not validate_features(features):
        st.warning("All feature values must be between 0 and 1.", icon="⚠️")
    elif len(features) != 11:
        st.warning("Please provide exactly 11 feature values.", icon="⚠️")
    else:
        st.info("Feature values are valid. Ready to submit!", icon="✔️")

    # Submit button
    if st.button("Submit Certificate", key="submit_manual"):
        if len(features) == 11 and validate_features(features):
            with st.spinner("Processing prediction..."):
                time.sleep(1)  # Simulate processing delay for UX
                result = predict_certificate(features)
                st.session_state.prediction = result
        else:
            st.error("Invalid input. Ensure all 11 features are between 0 and 1.", icon="❌")

    # Display result with styled box
    if st.session_state.prediction:
        result_class = "pass" if st.session_state.prediction == "Pass" else "fail"
        st.markdown(
            f'<div class="result-box {result_class}">Prediction: <b>{st.session_state.prediction}</b></div>',
            unsafe_allow_html=True
        )

# Tab 2: CSV Upload
with tab2:
    st.header("Batch Processing via CSV Upload")
    st.markdown("Upload a CSV file with 11 columns (no headers required) to process multiple certificates.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_upload")

    if uploaded_file:
        with st.spinner("Processing CSV file..."):
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file, header=None)
                if df.shape[1] != 11:
                    st.error("CSV must have exactly 11 columns.", icon="❌")
                else:
                    # Make predictions
                    predictions = model.predict(df.values)
                    results = ["Pass" if p == 1 else "Fail" for p in predictions]
                    df['Prediction'] = results

                    # Rename columns for display
                    display_df = df.rename(
                        columns={i: f"Feature_{i + 1}" for i in range(11)} | {"Prediction": "Prediction"})

                    # Display results
                    st.write("### Prediction Results")
                    st.dataframe(display_df, use_container_width=True)

                    # Interactive visualization: Bar chart of Pass/Fail counts
                    st.write("### Prediction Summary")
                    summary = display_df["Prediction"].value_counts().reset_index()
                    summary.columns = ["Prediction", "Count"]
                    fig = px.bar(
                        summary,
                        x="Prediction",
                        y="Count",
                        color="Prediction",
                        title="Pass/Fail Distribution",
                        color_discrete_map={"Pass": "#4CAF50", "Fail": "#EF5350"}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Provide download link for results
                    output = io.BytesIO()
                    df.to_csv(output, index=False, header=[f"Feature_{i + 1}" for i in range(11)] + ["Prediction"])
                    output.seek(0)
                    st.download_button(
                        label="Download Results CSV",
                        data=output,
                        file_name="certificate_results.csv",
                        mime="text/csv",
                        key="download_csv"
                    )
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}", icon="❌")

# Sidebar for Documentation and Reset
st.sidebar.title("About")
st.sidebar.markdown("""
This app classifies certificates using a pre-trained ML model. Enhanced features include:
- **Interactive Manual Input**: Real-time validation and styled result display.
- **CSV Upload with Visualization**: Process multiple certificates and view Pass/Fail distribution.
- **Downloadable Results**: Get predictions as a CSV file.

For more details, see the [GitHub repository](https://github.com/vinaykumarkv/Machine-Learning-Projects/tree/main/ML_apps/ml_cert_app).
""")

# Reset button in sidebar
if st.sidebar.button("Reset Inputs", key="reset"):
    st.session_state.features = [0.5] * 11
    st.session_state.prediction = None
    st.experimental_rerun()