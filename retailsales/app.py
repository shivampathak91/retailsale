import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from supabase import create_client
import openai
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from io import BytesIO
import uuid


# ---------------- CONFIG ----------------
st.set_page_config(page_title="Retail AI SaaS", layout="wide")

SUPABASE_URL = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")



if not SUPABASE_URL or not SUPABASE_KEY or not OPENAI_API_KEY:
    st.error("Missing SUPABASE or OPENAI configuration. Set st.secrets or environment variables.")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai.api_key = OPENAI_API_KEY

# ---------------- AUTH ----------------
def login(email, password):
    if not email or not password:
        return False, "Please provide both email and password."
    try:
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if getattr(response, "user", None) or getattr(response, "session", None):
            return True, response
        return False, "Invalid credentials or user not found."
    except Exception as e:
        return False, str(e)

def signup(email, password):
    if not email or not password:
        return False, "Please provide both email and password."
    try:
        response = supabase.auth.sign_up({"email": email, "password": password})
        if getattr(response, "user", None) or getattr(response, "session", None):
            return True, response
        return False, "Signup failed. Please check email and password."
    except Exception as e:
        return False, str(e)

# ---------------- LOGIN UI ----------------
menu = st.sidebar.selectbox("Menu", ["Login", "Signup"])

if menu == "Signup":
    st.title("Signup")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Create Account"):
        success, result = signup(email, password)
        if success:
            st.success("Signup request succeeded. If email confirmation is required, please check your inbox.")
            st.write(result)
        else:
            st.error(f"Signup failed: {result}")

elif menu == "Login":
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        success, result = login(email, password)
        if success:
            st.session_state["user"] = result.user.id
            st.session_state["session"] = result.session
            st.success("Login successful.")
            st.rerun()
        else:
            st.error(f"Login failed: {result}")

if "user" not in st.session_state:
    st.stop()

if "session" in st.session_state:
    supabase.auth.set_session(st.session_state["session"].access_token, st.session_state["session"].refresh_token)

user_id = st.session_state["user"]

# ---------------- DATA STORAGE ----------------
def save_dataset(file, user_id):
    file_id = str(uuid.uuid4())
    supabase.table("datasets").insert({
        "id": file_id,
        "user": user_id,
        "filename": file.name,
        "content": file.getvalue().decode("ISO-8859-1")
    }).execute()
    return file_id

def load_datasets(user_id):
    res = supabase.table("datasets").select("*").eq("user", user_id).execute()
    return res.data

# ---------------- UI ----------------
st.title("📊 AI Retail SaaS Dashboard")

uploaded_file = st.file_uploader("Upload Dataset")

if uploaded_file:
    save_dataset(uploaded_file, user_id)
    st.success("Dataset saved!")

datasets = load_datasets(user_id)

dataset_names = [d["filename"] for d in datasets]

selected = st.selectbox("Select Dataset", dataset_names)

if selected:
    selected_data = next(d for d in datasets if d["filename"] == selected)
    try:
        data = pd.read_csv(pd.io.common.StringIO(selected_data["content"]))
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()
else:
    st.stop()

# ---------------- CLEAN ----------------
required_columns = ["Order Date", "Sales", "Profit", "Customer ID"]
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    st.error(f"Dataset missing required columns: {', '.join(missing_columns)}")
    st.stop()

data.dropna(inplace=True)
data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
data = data.dropna(subset=['Order Date'])
if data.empty:
    st.error("Dataset has no valid order dates after parsing.")
    st.stop()

# ---------------- KPI ----------------
col1, col2, col3 = st.columns(3)
col1.metric("Sales", int(data['Sales'].sum()))
col2.metric("Profit", int(data['Profit'].sum()))
col3.metric("Orders", len(data))

# ---------------- TIME SERIES ----------------
data['month'] = data['Order Date'].dt.month
data['year'] = data['Order Date'].dt.year
monthly_sales = data.groupby(['year','month'])['Sales'].sum().reset_index()

fig = px.line(monthly_sales, x="month", y="Sales", color="year")
st.plotly_chart(fig)

# ---------------- FORECAST ----------------
# ---------------- ELITE FORECAST ENGINE ----------------
st.header("🚀 Smart AutoML Forecasting Engine")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet

# Prepare data
ts_data = data[['Order Date', 'Sales']].copy()
ts_data = ts_data.sort_values('Order Date')

# Train-test split
train_size = int(len(ts_data) * 0.8)
train = ts_data[:train_size]
test = ts_data[train_size:]

results = {}

# ---------------- PROPHET ----------------
try:
    prophet_train = train.rename(columns={'Order Date':'ds','Sales':'y'})
    model_p = Prophet(interval_width=0.95)
    model_p.fit(prophet_train)

    future = model_p.make_future_dataframe(periods=len(test))
    forecast_p = model_p.predict(future)

    y_pred_p = forecast_p['yhat'][-len(test):].values
    rmse_p = np.sqrt(mean_squared_error(test['Sales'], y_pred_p))

    results["Prophet"] = {
        "pred": y_pred_p,
        "rmse": rmse_p,
        "model": model_p,
        "forecast": forecast_p
    }
except:
    pass

# ---------------- ARIMA ----------------
try:
    model_a = ARIMA(train['Sales'], order=(5,1,2))
    model_a_fit = model_a.fit()

    y_pred_a = model_a_fit.forecast(steps=len(test))
    rmse_a = np.sqrt(mean_squared_error(test['Sales'], y_pred_a))

    results["ARIMA"] = {
        "pred": y_pred_a,
        "rmse": rmse_a
    }
except:
    pass

# ---------------- RANDOM FOREST ----------------
try:
    rf_data = ts_data.copy()
    rf_data['month'] = rf_data['Order Date'].dt.month
    rf_data['year'] = rf_data['Order Date'].dt.year

    X = rf_data[['month','year']]
    y = rf_data['Sales']

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model_rf = RandomForestRegressor(n_estimators=200)
    model_rf.fit(X_train, y_train)

    y_pred_rf = model_rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    results["Random Forest"] = {
        "pred": y_pred_rf,
        "rmse": rmse_rf
    }
except:
    pass

if not results:
    st.error("Forecasting failed: no model produced valid predictions. Check dataset size/format and package installation.")
    st.stop()

# ---------------- MODEL COMPARISON ----------------
st.subheader("📊 Model Performance")

rmse_df = pd.DataFrame([
    {"Model": k, "RMSE": v["rmse"]}
    for k,v in results.items()
])

import plotly.express as px
fig_rmse = px.bar(rmse_df, x="Model", y="RMSE", color="Model",
                 title="Model Accuracy (Lower RMSE = Better)")
st.plotly_chart(fig_rmse)

# ---------------- BEST MODEL ----------------
best_model = min(results.items(), key=lambda x: x[1]["rmse"])
best_name = best_model[0]
best_data = best_model[1]

st.success(f"🏆 Best Model Selected: {best_name}")

# ---------------- FORECAST VISUAL ----------------
fig_compare = px.line(title="Forecast Comparison")

fig_compare.add_scatter(y=test['Sales'].values, mode='lines', name='Actual')

for name, val in results.items():
    fig_compare.add_scatter(y=val["pred"], mode='lines', name=name)

st.plotly_chart(fig_compare)

# ---------------- CONFIDENCE INTERVAL (PROPHET ONLY) ----------------
if best_name == "Prophet":
    st.subheader("📉 Forecast Confidence Interval")

    fig_ci = px.line(best_data["forecast"], x="ds", y="yhat",
                     title="Forecast with Confidence Interval")

    fig_ci.add_scatter(
        x=best_data["forecast"]["ds"],
        y=best_data["forecast"]["yhat_upper"],
        mode='lines',
        name='Upper Bound',
        line=dict(dash='dot')
    )

    fig_ci.add_scatter(
        x=best_data["forecast"]["ds"],
        y=best_data["forecast"]["yhat_lower"],
        mode='lines',
        name='Lower Bound',
        line=dict(dash='dot')
    )

    st.plotly_chart(fig_ci)

# ---------------- SEASONAL DECOMPOSITION ----------------
st.subheader("📊 Time Series Decomposition")

try:
    decomposition = seasonal_decompose(ts_data['Sales'], model='additive', period=12)

    fig_decomp = px.line(title="Trend / Seasonality")

    fig_decomp.add_scatter(y=decomposition.trend, name="Trend")
    fig_decomp.add_scatter(y=decomposition.seasonal, name="Seasonality")
    fig_decomp.add_scatter(y=decomposition.resid, name="Residual")

    st.plotly_chart(fig_decomp)
except:
    st.warning("Not enough data for decomposition")

# ---------------- AI EXPLANATION ----------------
st.subheader("🤖 AI Insight on Forecast")

def explain_model(results):
    summary = "\n".join([f"{k}: RMSE={v['rmse']}" for k,v in results.items()])
    
    prompt = f"""
    These are forecasting model results:
    {summary}

    Explain:
    - Which model is best
    - Why it performed better
    - Business implications
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}]
        )
        return response.choices[0].message.content
    except openai.RateLimitError:
        return "⚠️ OpenAI quota exceeded. Please check your OpenAI billing at https://platform.openai.com/account/billing and add credits to continue using AI features."
    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"

if st.button("Explain Results"):
    with st.spinner("AI analyzing..."):
        explanation = explain_model(results)
        st.info(explanation)

# ---------------- CLUSTER ----------------
customer = data.groupby('Customer ID').agg({'Sales':'sum','Profit':'sum'}).reset_index()

scaler = StandardScaler()
X = scaler.fit_transform(customer[['Sales','Profit']])

kmeans = KMeans(n_clusters=3)
customer['Cluster'] = kmeans.fit_predict(X)

fig3 = px.scatter(customer, x="Sales", y="Profit", color="Cluster")
st.plotly_chart(fig3)

# ---------------- GPT ANOMALY DETECTION ----------------
st.header("🧠 AI Anomaly Detection")

def detect_anomalies(data):
    sample = data.describe().to_string()

    prompt = f"""
    Analyze this business data summary:
    {sample}

    Find anomalies, unusual trends, or risks.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}]
        )
        return response.choices[0].message.content
    except openai.RateLimitError:
        return "⚠️ OpenAI quota exceeded. Please check your OpenAI billing at https://platform.openai.com/account/billing and add credits to continue using AI features."
    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"

if st.button("Detect Anomalies"):
    with st.spinner("Analyzing..."):
        result = detect_anomalies(data)
        st.warning(result)

# ---------------- AI CHAT ----------------
st.header("🤖 AI Chatbot")

question = st.text_input("Ask about your data")

def ask_ai(data, q):
    sample = data.head(50).to_csv()

    prompt = f"""
    Data:
    {sample}

    Question: {q}
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}]
        )
        return response.choices[0].message.content
    except openai.RateLimitError:
        return "⚠️ OpenAI quota exceeded. Please check your OpenAI billing at https://platform.openai.com/account/billing and add credits to continue using AI features."
    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"

if st.button("Ask"):
    st.info(ask_ai(data, question))

# ---------------- PDF REPORT ----------------
st.header("🧾 Generate Report")

def create_pdf(data, results, best_model, customer_clusters):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=1  # Center
    )

    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=15,
        textColor=colors.darkblue
    )

    normal_style = styles['Normal']

    story = []

    # Title
    story.append(Paragraph("Retail Sales Analytics Report", title_style))
    story.append(Spacer(1, 12))

    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    story.append(Paragraph(f"Dataset Records: {len(data)}", normal_style))
    story.append(Paragraph(f"Date Range: {data['Order Date'].min()} to {data['Order Date'].max()}", normal_style))
    story.append(Spacer(1, 12))

    # Key Metrics
    story.append(Paragraph("Key Performance Metrics", heading_style))
    metrics_data = [
        ["Metric", "Value"],
        ["Total Sales", f"USD {data['Sales'].sum():,.2f}"],
        ["Total Profit", f"USD {data['Profit'].sum():,.2f}"],
        ["Average Order Value", f"USD {data['Sales'].mean():,.2f}"],
        ["Profit Margin", f"{(data['Profit'].sum() / data['Sales'].sum() * 100):,.1f}%"],
        ["Total Customers", str(data['Customer ID'].nunique())]
    ]

    metrics_table = Table(metrics_data)
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 12))

    # Forecasting Results
    if results:
        story.append(Paragraph("Sales Forecasting Analysis", heading_style))
        story.append(Paragraph(f"Best Performing Model: {best_model[0]} (RMSE: {best_model[1]['rmse']:.2f})", normal_style))
        story.append(Spacer(1, 6))

        # Model Comparison Table
        model_data = [["Model", "RMSE Score"]]
        for model_name, model_data_item in results.items():
            model_data.append([model_name, f"{model_data_item['rmse']:.2f}"])

        model_table = Table(model_data)
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(model_table)
        story.append(Spacer(1, 12))

    # Customer Segmentation
    if not customer_clusters.empty:
        story.append(Paragraph("Customer Segmentation Analysis", heading_style))

        cluster_summary = customer_clusters.groupby('Cluster').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Customer ID': 'count'
        }).round(2)

        cluster_data = [["Cluster", "Customers", "Total Sales", "Total Profit"]]
        for cluster_id, row in cluster_summary.iterrows():
            cluster_data.append([
                f"Cluster {cluster_id}",
                str(int(row['Customer ID'])),
                f"USD {row['Sales']:,.2f}",
                f"USD {row['Profit']:,.2f}"
            ])

        cluster_table = Table(cluster_data)
        cluster_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(cluster_table)
        story.append(Spacer(1, 12))

    # Top Products
    story.append(Paragraph("Top Performing Products", heading_style))
    top_products = data.groupby('Product Name').agg({
        'Sales': 'sum',
        'Quantity': 'sum'
    }).sort_values('Sales', ascending=False).head(5)

    product_data = [["Rank", "Product", "Sales", "Quantity"]]
    for idx, (product, row) in enumerate(top_products.iterrows(), 1):
        product_data.append([
            str(idx),
            str(product)[:50],  # Truncate long names
            f"USD {row['Sales']:,.2f}",
            str(int(row['Quantity']))
        ])

    product_table = Table(product_data, colWidths=[40, 200, 80, 60])
    product_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (1, 1), (1, -1), 'LEFT'),  # Left align product names
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(product_table)
    story.append(Spacer(1, 12))

    # Regional Performance
    story.append(Paragraph("Regional Performance", heading_style))
    region_perf = data.groupby('Region').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).sort_values('Sales', ascending=False)

    region_data = [["Region", "Sales", "Profit"]]
    for region, row in region_perf.iterrows():
        region_data.append([
            str(region),
            f"USD {row['Sales']:,.2f}",
            f"USD {row['Profit']:,.2f}"
        ])

    region_table = Table(region_data)
    region_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(region_table)
    story.append(Spacer(1, 12))

    # Recommendations
    story.append(Paragraph("Key Recommendations", heading_style))
    recommendations = [
        "Focus on high-performing products and regions",
        "Implement targeted marketing for different customer segments",
        "Monitor forecast accuracy and adjust models quarterly",
        "Consider inventory optimization based on demand patterns"
    ]

    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", normal_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

if st.button("Generate PDF"):
    pdf_bytes = create_pdf(data, results, best_model, customer)
    st.download_button("Download PDF", pdf_bytes, file_name="retail_analytics_report.pdf")

# ---------------- REAL-TIME REFRESH ----------------
st.sidebar.header("Live Mode")

if st.sidebar.button("Refresh Data"):
    st.rerun()

# ---------------- STYLE ----------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)