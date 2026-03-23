import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock Market Prediction", layout="wide")
st.title("Stock Market Prediction — LSTM")

st.sidebar.header("Upload Data")
uploaded_files = st.sidebar.file_uploader(
    "Upload 1 to 4 company CSV files (with columns: Date, Open, High, Low, Close, Adj Close, Volume)",
    type="csv",
    accept_multiple_files=True,
)

BETA = 0.93
LOOKBACK = 60
EPOCHS = 50
BATCH_SIZE = 32


def load_and_clean(file):
    df = pd.read_csv(file)
    df.dropna(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    df = df.sort_values("Date", ignore_index=True)
    return df


def apply_ema_smoothing(df, beta=BETA):
    prices = np.array(df["Close"])
    smoothed = [prices[0]]
    for i in range(1, len(prices)):
        smoothed.append(smoothed[-1] * beta + (1 - beta) * prices[i])
    df = df.copy()
    df["Close"] = smoothed
    return df


def add_moving_averages(df):
    for window in [10, 20, 50]:
        df[f"MA_{window}"] = df["Close"].rolling(window).mean()
    return df


def build_sequences(scaled, lookback=LOOKBACK):
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y)


def train_lstm(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE):
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model, history


if not uploaded_files:
    st.info("Upload CSV files from the sidebar to get started.")
    st.stop()

companies = []
company_names = []

for i, f in enumerate(uploaded_files[:4]):
    df = load_and_clean(f)
    if i == 3:
        df = apply_ema_smoothing(df, BETA)
    df = add_moving_averages(df)
    companies.append(df)
    company_names.append(f.name.replace(".csv", ""))

tabs = st.tabs(["Overview", "EDA", "Moving Averages", "Train & Predict"])

with tabs[0]:
    st.subheader("Dataset Overview")
    for name, df in zip(company_names, companies):
        st.markdown(f"**{name}**")
        st.dataframe(df.head(5), use_container_width=True)
        st.caption(f"Shape: {df.shape}")

with tabs[1]:
    st.subheader("Closing Price — All Companies")
    n = len(companies)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = np.array(axes).flatten()
    for i, (name, df) in enumerate(zip(company_names, companies)):
        axes[i].plot(df["Date"], df["Close"], linewidth=1.2)
        axes[i].set_title(name)
        axes[i].set_ylabel("Close")
        axes[i].tick_params(axis="x", rotation=30)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Descriptive Statistics")
    selected = st.selectbox("Select company", company_names, key="desc")
    idx = company_names.index(selected)
    st.dataframe(companies[idx].describe(), use_container_width=True)

with tabs[2]:
    st.subheader("Moving Averages")
    selected_ma = st.selectbox("Select company", company_names, key="ma")
    idx = company_names.index(selected_ma)
    df = companies[idx]
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.plot(df["Date"], df["Close"], label="Close", linewidth=1)
    ax2.plot(df["Date"], df["MA_10"], label="MA 10", linewidth=1)
    ax2.plot(df["Date"], df["MA_20"], label="MA 20", linewidth=1)
    ax2.plot(df["Date"], df["MA_50"], label="MA 50", linewidth=1)
    ax2.set_title(f"{selected_ma} — Moving Averages")
    ax2.legend()
    ax2.tick_params(axis="x", rotation=30)
    fig2.tight_layout()
    st.pyplot(fig2)

with tabs[3]:
    st.subheader("Train LSTM & Predict")

    selected_train = st.selectbox("Select company to train", company_names, key="train")
    train_ratio = st.slider("Train split (%)", 60, 90, 80)
    epochs_input = st.number_input("Epochs", min_value=5, max_value=200, value=EPOCHS, step=5)

    if st.button("Train Model"):
        idx = company_names.index(selected_train)
        df = companies[idx]

        trainset = df[["Open"]].values
        split = int(len(trainset) * train_ratio / 100)
        train_data = trainset[:split]
        test_data = trainset[split:]

        sc = MinMaxScaler(feature_range=(0, 1))
        train_scaled = sc.fit_transform(train_data)

        if len(train_scaled) <= LOOKBACK:
            st.error(f"Not enough training rows. Need more than {LOOKBACK}.")
            st.stop()

        x_train, y_train = build_sequences(train_scaled, LOOKBACK)

        with st.spinner("Training LSTM..."):
            model, history = train_lstm(x_train, y_train, epochs=int(epochs_input))

        st.success("Training complete.")

        fig_loss, ax_loss = plt.subplots(figsize=(10, 3))
        ax_loss.plot(history.history["loss"])
        ax_loss.set_title("Training Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("MSE")
        fig_loss.tight_layout()
        st.pyplot(fig_loss)

        dataset_total = df["Open"].values
        inputs = dataset_total[len(dataset_total) - len(test_data) - LOOKBACK:]
        inputs = inputs.reshape(-1, 1)
        inputs = sc.transform(inputs)

        x_test = []
        for i in range(LOOKBACK, len(inputs)):
            x_test.append(inputs[i - LOOKBACK:i, 0])
        x_test = np.array(x_test).reshape(-1, LOOKBACK, 1)

        predicted = model.predict(x_test)
        predicted = sc.inverse_transform(predicted)

        real = test_data
        n_pred = min(len(real), len(predicted))

        rmse = math.sqrt(mean_squared_error(real[:n_pred], predicted[:n_pred]))
        mae = mean_absolute_error(real[:n_pred], predicted[:n_pred])
        col1, col2 = st.columns(2)
        col1.metric("RMSE", f"{rmse:.4f}")
        col2.metric("MAE", f"{mae:.4f}")

        test_dates = df["Date"].values[split:split + n_pred]
        fig_pred, ax_pred = plt.subplots(figsize=(14, 5))
        ax_pred.plot(df["Date"].values[:split], df["Open"].values[:split], label="Train", linewidth=0.8)
        ax_pred.plot(test_dates, real[:n_pred], label="Actual", linewidth=1.2)
        ax_pred.plot(test_dates, predicted[:n_pred], label="Predicted", linewidth=1.2, linestyle="--")
        ax_pred.set_title(f"{selected_train} — Actual vs Predicted (Open Price)")
        ax_pred.legend()
        ax_pred.tick_params(axis="x", rotation=30)
        fig_pred.tight_layout()
        st.pyplot(fig_pred)

        result_df = pd.DataFrame({
            "Date": test_dates,
            "Actual": real[:n_pred].flatten(),
            "Predicted": predicted[:n_pred].flatten(),
        })
        st.dataframe(result_df, use_container_width=True)

        csv = result_df.to_csv(index=False).encode()
        st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")
