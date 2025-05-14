import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
import os
from jsonCsvConverter import convert_json_to_csv
from cleanData import load_and_clean
from predictionPipeline import run_prediction_pipeline_with_progress


# path to the default dataset
DATA_PATH = Path("data/processed/reviews_with_scores.csv")

# page title
st.title("🛒 AI Sentiment & Trust Dashboard")

# tabs
tab1, tab2, tab3 = st.tabs(["📤 Data", "📊 Analysis", "🤖 Fine-Tune AI"])

#tab 1 - Data Upload & Cleaning
with tab1:
    st.header("📤 Upload & Convert JSON File")

    uploaded_json = st.file_uploader("Upload a JSON file", type=["json"])

    if uploaded_json is not None:
        output_csv_name = uploaded_json.name.replace(".json", ".csv")
        output_csv_path = f"data/raw/{output_csv_name}"

        if st.button("Convert JSON ➔ CSV"):
            temp_json_path = "temp_upload.json"

            # Save uploaded JSON temporarily
            with open(temp_json_path, "wb") as f:
                f.write(uploaded_json.getbuffer())

            # Convert JSON to CSV
            convert_json_to_csv(temp_json_path, output_csv_path)

            # Success message and download button
            st.success(f"✅ Saved CSV as {output_csv_path}")
            with open(output_csv_path, "rb") as f:
                st.download_button(
                    label="Download Converted CSV",
                    data=f,
                    file_name=output_csv_name,
                    mime="text/csv"
                )

            # clean up temp file
            try:
                os.remove(temp_json_path)
            except FileNotFoundError:
                pass

    st.divider()

    st.subheader("📤 Upload a CSV file for Cleaning")

    uploaded_csv = st.file_uploader("Upload a CSV file to clean", type=["csv"], key="csv")

    if uploaded_csv is not None:
        temp_csv_path = "temp_uploaded.csv"
        with open(temp_csv_path, "wb") as f:
            f.write(uploaded_csv.getbuffer())

        if st.button("Clean Uploaded CSV"):
            try:
                # Call your cleaning function
                train_df, test_df = load_and_clean(temp_csv_path)

                # Save outputs temporarily
                train_output_path = "data/processed/train_clean.csv"
                test_output_path = "data/processed/test_clean.csv"

                train_df.to_csv(train_output_path, index=False)
                test_df.to_csv(test_output_path, index=False)

                st.success(
                    f"✅ Data cleaned successfully! Saved CSV as {train_output_path}  & {test_output_path}! "
                    "Download your files below:"
                )

                # Download buttons
                with open(train_output_path, "rb") as f:
                    st.download_button(
                        label="Download Train Split",
                        data=f,
                        file_name="train_clean.csv",
                        mime="text/csv"
                    )
                with open(test_output_path, "rb") as f:
                    st.download_button(
                        label="Download Test Split",
                        data=f,
                        file_name="test_clean.csv",
                        mime="text/csv"
                    )

                # clean up temp file
                try:
                    os.remove(temp_csv_path)
                except FileNotFoundError:
                    pass

            except Exception as e:
                st.error(f"⚠️ Error cleaning data: {e}")


# Tab 2 - Analysis Section
with tab2:
    st.subheader("🔮 Run Sentiment + Trust Analysis on Current Dataset")

    processed_csvs = sorted([p.name for p in Path("data/processed").glob("*.csv")])
    dataset_choice = st.selectbox("Select a processed CSV to analyze:", processed_csvs)

    if st.button("Run AI Sentiment Analysis"):
        from predictionPipeline import run_prediction_pipeline_with_progress

        # Setup Streamlit placeholders
        progress_bar = st.progress(0)
        status_placeholder = st.empty()

        try:
            run_prediction_pipeline_with_progress(
                input_csv_path=f"data/processed/{dataset_choice}",
                output_csv_path="data/processed/reviews_with_scores.csv",
                progress_bar=progress_bar,
                status_placeholder=status_placeholder
            )
            df = pd.read_csv("data/processed/reviews_with_scores.csv")
            st.success("✅ Sentiment Analysis complete! Dashboard refreshed.")

            st.session_state.sentiment_filter = ["positive", "neutral", "negative"]

        except Exception as e:
            st.error(f"⚠️ Error during Sentiment Analysis: {e}")

    st.divider()

    if "sentiment_filter" not in st.session_state:
        st.session_state.sentiment_filter = []

    # function to load cleaned data
    @st.cache_data
    def load_data(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        # ensure expected columns exist
        expected = {"review_text", "sentiment", "confidence",
                    "trust_score", "asin", "reviewTime"}
        missing = expected - set(df.columns)
        if missing:
            st.error(f"Missing columns: {missing}")
        return df

    # loads dataset
    df = load_data(DATA_PATH)

    # sidebar filters
    st.sidebar.header("🔍 Filters")
    sentiment_filter = st.sidebar.multiselect(
        "Sentiment", options=["positive", "neutral", "negative"],
        default=st.session_state.sentiment_filter  
    )
    trust_min, trust_max = st.sidebar.slider(
        "Trust-score range", 0, 100, (0, 100), step=1)
    asin_filter = st.sidebar.text_input("Filter by ASIN (optional)").strip()

    # apply filters
    mask = (
        df["sentiment"].isin(sentiment_filter) &
        df["trust_score"].between(trust_min, trust_max)
    )
    if asin_filter:
        mask &= df["asin"].str.contains(asin_filter, case=False, na=False)
    filtered_df = df[mask]

    st.subheader(f"Dataset overview — {len(filtered_df):,} reviews selected")

    # visuals
    col1, col2 = st.columns(2)

    with col1:
        # sentiment distribution
        sent_fig = px.histogram(filtered_df, x="sentiment", color="sentiment",
                                category_orders={"sentiment": ["positive", "neutral", "negative"]},
                                title="Sentiment distribution", text_auto=True)
        st.plotly_chart(sent_fig, use_container_width=True)

    with col2:
        # trust score histogram
        trust_fig = px.histogram(filtered_df, x="trust_score", nbins=20,
                                 title="Trust-score histogram", labels={"trust_score": "Score"})
        st.plotly_chart(trust_fig, use_container_width=True)

    # review table
    st.markdown("### 📄 Review Table")
    st.dataframe(
        filtered_df[["asin", "sentiment", "trust_score", "review_text"]]
                 .sort_values("trust_score"),
        use_container_width=True,
        height=400
    )

    # export filtered data
    csv = filtered_df.to_csv(index=False)
    st.download_button("Download filtered CSV", data=csv,
                       file_name="filtered_reviews.csv",
                       mime="text/csv")

# tab 3 - Fine-tune AI Model
with tab3:
    st.header("🤖 Fine-Tune Your Sentiment AI Model")

    csv_files = list(Path("data/processed").glob("*.csv"))
    csv_files = [f.name for f in csv_files]  # Get only the filenames, not full path

    dataset_option = st.selectbox(
        "Choose training dataset:",
        csv_files
    )

    epochs = st.slider("Number of Training Epochs", 1, 10, 3)
    batch_size = st.selectbox("Batch Size", options=[4, 8, 16], index=0)

    # Placeholders for live updates
    training_log = st.empty()
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    if st.button("🚀 Start Fine-Tuning"):
        st.info("Initializing model training...")

        from trainModel import main as train_model

        try:
            train_model(
                progress_placeholder=training_log,
                progress_bar_placeholder=progress_bar,
                status_placeholder=status_placeholder,
                num_epochs=epochs,
                batch_size=batch_size
            )
            st.success("✅ Fine-tuning complete!")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
