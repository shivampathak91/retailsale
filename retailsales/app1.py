import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from fpdf import FPDF
from datetime import datetime

# ML
from prophet import Prophet

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI BI SaaS", layout="wide")
st.title("📊 AI + BI Analytics SaaS (Power BI Style)")

# ---------------- SESSION INIT ----------------
if "datasets" not in st.session_state:
    st.session_state["datasets"] = {}

# ---------------- LOGIN ----------------
st.sidebar.header("Login")
email = st.sidebar.text_input("Enter Email")

if st.sidebar.button("Login"):
    if email:
        st.session_state["user"] = email

if "user" not in st.session_state:
    st.warning("Please login to continue")
    st.stop()

st.sidebar.success(f"User: {st.session_state['user']}")

# ---------------- FILE LOADER ----------------
def load_file(file):
    try:
        return pd.read_csv(file, encoding="utf-8")
    except:
        return pd.read_csv(file, encoding="latin1", on_bad_lines="skip")

# ---------------- UPLOAD ----------------
st.sidebar.header("📤 Upload CSV")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:
    df = load_file(file)
    st.session_state["datasets"][file.name] = df
    st.sidebar.success("✅ File uploaded successfully!")

# ---------------- DATA SELECT ----------------
if not st.session_state["datasets"]:
    st.info("Upload dataset first")
    st.stop()

dataset_name = st.selectbox("Select Dataset", list(st.session_state["datasets"].keys()))
df = st.session_state["datasets"][dataset_name]

st.subheader("📊 Data Preview")
st.dataframe(df.head())

# ---------------- CLEAN DATA ----------------
if "Order Date" in df.columns:
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df = df.dropna(subset=["Order Date"])

df = df.dropna()

# ---------------- FILTER PANEL ----------------
st.sidebar.header("🎛 Filters")

if "Region" in df.columns:
    region_filter = st.sidebar.multiselect("Region", df["Region"].unique(), df["Region"].unique())
    df = df[df["Region"].isin(region_filter)]

if "Category" in df.columns:
    category_filter = st.sidebar.multiselect("Category", df["Category"].unique(), df["Category"].unique())
    df = df[df["Category"].isin(category_filter)]

# ---------------- KPIs ----------------
st.subheader("📌 KPIs")

col1, col2, col3 = st.columns(3)

if "Sales" in df.columns:
    col1.metric("Total Sales", f"{df['Sales'].sum():,.0f}")

if "Profit" in df.columns:
    col2.metric("Total Profit", f"{df['Profit'].sum():,.0f}")

col3.metric("Rows", len(df))

# ---------------- DASHBOARD ----------------
st.subheader("📊 Dashboard")

if "Sales" in df.columns:
    fig1 = px.histogram(df, x="Sales", title="Sales Distribution")
    st.plotly_chart(fig1, use_container_width=True)

if "Category" in df.columns and "Sales" in df.columns:
    fig2 = px.bar(
        df.groupby("Category")["Sales"].sum().reset_index(),
        x="Category", y="Sales", title="Sales by Category"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- TIME SERIES ----------------
if "Order Date" in df.columns and "Sales" in df.columns:
    ts = df.groupby("Order Date")["Sales"].sum().reset_index()

    fig3 = px.line(ts, x="Order Date", y="Sales", title="Sales Trend")
    st.plotly_chart(fig3, use_container_width=True)

# ---------------- AI INSIGHTS ----------------
st.subheader("🤖 AI Insights (Why Sales Dropped?)")

insights = []

if "Sales" in df.columns:

    if "Order Date" in df.columns:
        daily = df.groupby("Order Date")["Sales"].sum()

        if len(daily) > 5:
            drop = daily.pct_change().min()

            if pd.notna(drop) and drop < -0.2:
                insights.append(f"⚠️ Sales dropped by {drop:.2%} at some point.")

    if df["Sales"].std() > df["Sales"].mean() * 0.5:
        insights.append("📉 Sales are highly unstable.")

    if "Discount" in df.columns and df["Discount"].mean() > 0.2:
        insights.append("💸 High discounts may be hurting profits.")

if not insights:
    insights.append("✅ No major issues detected.")

for i in insights:
    st.success(i)

# ---------------- FORECASTING ----------------
st.subheader("🔮 Forecasting Engine (Select Model)")

if "Order Date" in df.columns and "Sales" in df.columns:

    import plotly.graph_objects as go

    # ---------------- CLEAN DATA ----------------
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales"])

    ts = df.groupby("Order Date")["Sales"].sum().reset_index()
    ts = ts.sort_values("Order Date")

    ts.columns = ["ds", "y"]

    st.write("📊 Data points:", len(ts))

    if len(ts) < 10:
        st.warning("Need at least 10 data points for forecasting")
        st.stop()

    # ---------------- MODEL SELECTOR ----------------
    model_choice = st.selectbox(
        "Choose Forecasting Model",
        ["Prophet", "ARIMA", "Random Forest"]
    )

    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]

    forecast_values = None

    # ---------------- PROPHET ----------------
    if model_choice == "Prophet":
        try:
            from prophet import Prophet

            model = Prophet()
            model.fit(train)

            future = model.make_future_dataframe(periods=len(test))
            forecast = model.predict(future)

            forecast_values = forecast["yhat"].iloc[-len(test):].values

            st.success("Prophet model used")

        except Exception as e:
            st.error(f"Prophet error: {e}")

    # ---------------- ARIMA ----------------
    elif model_choice == "ARIMA":
        try:
            from statsmodels.tsa.arima.model import ARIMA

            model = ARIMA(train["y"], order=(5,1,0))
            model_fit = model.fit()

            forecast_values = model_fit.forecast(len(test))

            st.success("ARIMA model used")

        except Exception as e:
            st.error(f"ARIMA error: {e}")

    # ---------------- RANDOM FOREST ----------------
    elif model_choice == "Random Forest":
        try:
            from sklearn.ensemble import RandomForestRegressor

            ts["month"] = pd.to_datetime(ts["ds"]).dt.month
            ts["year"] = pd.to_datetime(ts["ds"]).dt.year

            X = ts[["month", "year"]]
            y = ts["y"]

            X_train = X[:train_size]
            X_test = X[train_size:]

            model = RandomForestRegressor(n_estimators=200)
            model.fit(X_train, y_train := y[:train_size])

            forecast_values = model.predict(X_test)

            st.success("Random Forest model used")

        except Exception as e:
            st.error(f"RF error: {e}")

    # ---------------- PLOT ----------------
    if forecast_values is not None:

        fig = go.Figure()

        # Actual
        fig.add_trace(go.Scatter(
            x=test["ds"],
            y=test["y"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="blue")
        ))

        # Forecast
        fig.add_trace(go.Scatter(
            x=test["ds"],
            y=forecast_values,
            mode="lines+markers",
            name=f"Predicted ({model_choice})",
            line=dict(color="red")
        ))

        fig.update_layout(
            title=f"Forecasting using {model_choice}",
            xaxis_title="Date",
            yaxis_title="Sales",
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No forecast generated")

else:
    st.warning("Order Date and Sales columns required")

# ---------------- PDF REPORT ----------------
st.subheader("📄 Auto Report Generator")

def create_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="AI BI Report", ln=True)
    pdf.cell(200, 10, txt=f"User: {st.session_state['user']}", ln=True)
    pdf.cell(200, 10, txt=f"Generated: {datetime.now()}", ln=True)

    pdf.cell(200, 10, txt="Insights:", ln=True)

    for i in insights:
        pdf.cell(200, 10, txt=str(i), ln=True)

    file_path = "report.pdf"
    pdf.output(file_path)
    return file_path

if st.button("Generate PDF Report"):
    with st.spinner("Generating report..."):
        path = create_pdf()

        with open(path, "rb") as f:
            st.download_button(
                "⬇️ Download Report",
                f,
                file_name="AI_Report.pdf",
                mime="application/pdf"
            )