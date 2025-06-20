import streamlit as st
import pandas as pd
from ml_pipeline import train_and_evaluate
from PIL import Image
import io

st.set_page_config(page_title="Universal Supervised ML App", layout="wide")
st.title("🤖 Universal Supervised ML App")

st.markdown("Upload a dataset and select the target column to train an ML model automatically.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    target = st.selectbox("Select the target (dependent) column", df.columns)

    future_file = st.file_uploader("Optional: Upload future data CSV", type=["csv"])
    future_df = pd.read_csv(future_file) if future_file else None

    if st.button("🚀 Train Model"):
        with st.spinner("Training model, please wait..."):
            result, plots, predictions = train_and_evaluate(df, target, future_df)

        st.subheader("✅ Model Result")
        st.text(result)

        if plots:
            st.subheader("📊 Visualizations")
            for p in plots:
                st.image(p)

        if predictions is not None:
            st.subheader("🔮 Future Predictions")
            st.dataframe(predictions.head())
            csv = predictions.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
