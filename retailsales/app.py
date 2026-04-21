import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from groq import Groq
from supabase import create_client
from io import BytesIO
import tempfile
from prophet import Prophet

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# ================= CONFIG =================
st.set_page_config(page_title="InSightX", layout="wide")

# 🔑 ADD YOUR KEYS HERE
SUPABASE_URL = ""
SUPABASE_KEY = ""
GROQ_API_KEY = ""

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
client = Groq(api_key=GROQ_API_KEY)

# ================= SESSION =================
for k in ["user", "session", "df"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ================= AI =================
def ai_explain(prompt, data):
    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """
"You are a Retail Data Analyst.

Rules:
- NEVER return Python code
- ALWAYS return final numeric answers when asked (like total sales, profit)
- Be direct and business-friendly
- If user asks for totals → calculate from given data summary
- Give short answer + insight

Format:
Answer: <value>
Insight: <short business meaning>"

Explain the visualization in 3 parts:
1. Trend
2. Retail Insight
3. Actionable Takeaway

Keep it short and professional.
"""
                },
                {"role": "user", "content": f"{prompt}\n\nData:\n{data}"}
            ]
        )
        return res.choices[0].message.content
    except:
        return "AI unavailable"

# ================= LOGIN =================
if not st.session_state["user"]:

    # 🎯 ADD TITLE HERE
    st.title("Welcome To 🚀 InSightX")
    st.markdown("### Smart Retail Analytics & AI Insights Platform")
    st.markdown("---")


    mode = st.selectbox("Mode", ["Login", "Signup"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Submit", key="auth_btn"):
        try:
            if mode == "Signup":
                supabase.auth.sign_up({"email": email, "password": password})
                st.success("Account created")
            else:
                res = supabase.auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })
                st.session_state["user"] = email
                st.session_state["session"] = res.session
                st.rerun()
        except Exception as e:
            st.error(str(e))

    st.stop()

# ================= FILE =================
st.title("📊 Turn Data into Decisions Instantly")

file = st.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)
    df.columns = df.columns.str.lower()
    st.session_state["df"] = df

if st.session_state["df"] is None:
    st.stop()

df = st.session_state["df"]

num_cols = df.select_dtypes(include="number").columns
cat_cols = df.select_dtypes(include="object").columns

# ================= KPI =================
total_sales = df[num_cols[0]].sum()
total_profit = df[num_cols[1]].sum() if len(num_cols) > 1 else 0

st.metric("Total Sales", total_sales)
st.metric("Total Profit", total_profit)

# ================= CATEGORY =================
st.subheader("📊 Category Sales")

cat_fig = None
if len(cat_cols) > 0:
    cat = df.groupby(cat_cols[0])[num_cols[0]].sum().reset_index()

    cat_fig = px.bar(cat, x=cat_cols[0], y=num_cols[0])
    st.plotly_chart(cat_fig)

    if st.button("Explain Category Sales", key="cat_btn_unique"):

        # ✅ LIMIT DATA (CRITICAL FIX)
        sample_data = cat.head(10).to_string()

        st.info(
            ai_explain(
                "Bar chart showing category-wise sales performance",
                sample_data
            )
        )

# ================= REGION PIE =================
st.subheader("🌍 Regional Sales Distribution")

region_fig = None

if "region" in df.columns and len(num_cols) > 0:
    region_data = df.groupby("region")[num_cols[0]].sum().reset_index()

    region_fig = px.pie(
        region_data,
        names="region",
        values=num_cols[0],
        title="Sales Distribution by Region"
    )

    st.plotly_chart(region_fig, use_container_width=True)

    # ✅ UNIQUE BUTTON KEY (important fix)
    if st.button("Explain Regional Sales", key="region_btn"):
        st.info(
            ai_explain(
                "Pie chart showing regional sales distribution",
                region_data.to_string()
            )
        )
# ================= CORRELATION =================
st.subheader("📉 Correlation")
corr_fig = None
if len(num_cols) > 1:
    corr = df[num_cols].corr()
    corr_fig = px.imshow(corr)
    st.plotly_chart(corr_fig)

    if st.button("Explain Correlation", key="corr_btn"):
        st.info(ai_explain("Correlation matrix", corr.to_string()))

# ================= SALES VS PROFIT =================
st.subheader("💰 Sales vs Profit")
sv_fig = None
if len(cat_cols) > 0 and len(num_cols) >= 2:
    seg = df.groupby(cat_cols[0])[[num_cols[0], num_cols[1]]].sum().reset_index()
    sv_fig = px.scatter(seg, x=num_cols[0], y=num_cols[1], color=cat_cols[0])
    st.plotly_chart(sv_fig)

    if st.button("Explain Sales vs Profit", key="seg_btn"):
        st.info(ai_explain("Sales vs profit scatter", seg.to_string()))

# ================= FORECAST =================
st.subheader("🔮 AI Forecast (ML Powered)")
forecast_fig = None

metric_option = st.selectbox("Select Metric", ["Sales", "Profit"])
choice = st.selectbox("Forecast Range", ["7 Days", "30 Days", "6 Months", "1 Year"])

days_map = {"7 Days":7,"30 Days":30,"6 Months":180,"1 Year":365}
periods = days_map[choice]

value_col = num_cols[0] if metric_option=="Sales" else (num_cols[1] if len(num_cols)>1 else num_cols[0])

date_cols = [c for c in df.columns if "date" in c]

if date_cols:
    date_col = date_cols[0]
    ts = df[[date_col, value_col]].copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts[value_col] = pd.to_numeric(ts[value_col], errors="coerce")
    ts = ts.dropna()
    ts = ts.groupby(date_col, as_index=False)[value_col].sum()
    ts.columns = ["ds", "y"]

    if len(ts) >= 10:
        model = Prophet()
        model.fit(ts)

        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        forecast_fig = px.line(forecast, x="ds", y="yhat")
        st.plotly_chart(forecast_fig)

        if st.button("Explain Forecast", key="forecast_btn"):
            st.info(ai_explain(f"{metric_option} forecast", forecast.tail(20).to_string()))


# ================= MONTHLY =================
st.subheader("📅 Monthly Sales")

monthly_fig = None

# 🔥 AUTO DETECT DATE COLUMN
date_cols = [col for col in df.columns if "date" in col.lower()]

if len(date_cols) == 0:
    st.warning("No date column found for monthly analysis")
else:
    date_col = date_cols[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    monthly = df.groupby(df[date_col].dt.to_period("M"))[num_cols[0]].sum().reset_index()
    monthly[date_col] = monthly[date_col].astype(str)

    monthly_fig = px.bar(monthly, x=date_col, y=num_cols[0], title="Monthly Sales")

    st.plotly_chart(monthly_fig, use_container_width=True)

    # ✅ EXPLAIN BUTTON
    if st.button("Explain Monthly", key="monthly_btn"):
        st.info(ai_explain("Monthly sales trend", monthly.to_string()))

# ================= TREEMAP =================
st.subheader("📊 Profitability by Category & Sub-Category")

treemap_fig = None
if len(cat_cols) >= 2 and len(num_cols) >= 2:
    treemap_fig = px.treemap(
        df,
        path=[cat_cols[0], cat_cols[1]],
        values=num_cols[0],  # Sales
        color=num_cols[1],   # Profit
        title="Sales Size vs Profit Color"
    )
    st.plotly_chart(treemap_fig, use_container_width=True)

    if st.button("Explain Treemap", key="tree_btn"):
        st.info(ai_explain(
            "Treemap showing category and sub-category sales vs profit",
            df[[cat_cols[0], cat_cols[1], num_cols[0], num_cols[1]]].head().to_string()
        ))

# ================= SHIPPING =================
st.subheader("🚚 Shipping Efficiency")

ship_fig = None

date_cols = [col for col in df.columns if "date" in col]

if len(date_cols) >= 2:
    order_col = date_cols[0]
    ship_col = date_cols[1]

    temp = df[[order_col, ship_col]].copy()
    temp[order_col] = pd.to_datetime(temp[order_col], errors="coerce")
    temp[ship_col] = pd.to_datetime(temp[ship_col], errors="coerce")

    temp["lead_time"] = (temp[ship_col] - temp[order_col]).dt.days

    ship_data = df.copy()
    ship_data["lead_time"] = temp["lead_time"]

    ship_avg = ship_data.groupby("ship mode")["lead_time"].mean().reset_index()

    ship_fig = px.bar(ship_avg, x="ship mode", y="lead_time", title="Avg Shipping Time")
    st.plotly_chart(ship_fig, use_container_width=True)

    if st.button("Explain Shipping Efficiency", key="ship_btn"):
        st.info(ai_explain("Shipping lead time analysis", ship_avg.to_string()))

# ================= SEGMENT PIE =================
st.subheader("👥 Customer Segment Distribution")

segment_fig = None

if "segment" in df.columns:
    seg_data = df.groupby("segment")[num_cols[0]].sum().reset_index()

    segment_fig = px.pie(seg_data, names="segment", values=num_cols[0])
    st.plotly_chart(segment_fig, use_container_width=True)

    if st.button("Explain Segment Distribution", key="segpie_btn"):
        st.info(ai_explain("Customer segment revenue share", seg_data.to_string()))

# ================= GEO MAP =================
st.subheader("🌍 Geographical Sales Map")

geo_fig = None

if "city" in df.columns and "sales" in df.columns:
    geo_data = df.groupby("city")["sales"].sum().reset_index()

    geo_fig = px.scatter_geo(
        geo_data,
        locations="city",
        locationmode="country names",
        size="sales",
        title="Sales by City"
    )

    st.plotly_chart(geo_fig, use_container_width=True)

    if st.button("Explain Geography", key="geo_btn"):
        st.info(ai_explain("Geographical sales distribution", geo_data.to_string()))



# ================= AI CHATBOT =================
st.subheader("🤖 AI Chatbot (Ask Your Data)")

# store chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_query = st.text_input("Ask something about your dataset")

if st.button("Ask AI", key="chat_btn"):

    if user_query:

        # 🔥 CONTEXT FOR AI
        context = f"""
Columns: {list(df.columns)}

Sample Data:
{df.head(5).to_string()}

Stats:
{df.describe().to_string()}
"""

        answer = ai_explain(user_query, context)

        # save history
        st.session_state["chat_history"].append(("You", user_query))
        st.session_state["chat_history"].append(("AI", answer))

# ================= SHOW CHAT =================
for role, msg in st.session_state["chat_history"]:
    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 AI:** {msg}")


# ================= PDF =================
def fig_to_img(fig):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.write_image(tmp.name)
    return tmp.name

def generate_pdf(metric_option, choice):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Retail AI Report", styles["Title"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Total Sales: {total_sales}", styles["Normal"]))
    story.append(Paragraph(f"Total Profit: {total_profit}", styles["Normal"]))

    charts = [
        ("Category", cat_fig, "Category sales"),
        ("Regional Sales", region_fig, "Sales distribution across regions"),
        ("Correlation", corr_fig, "Correlation"),
        ("Sales vs Profit", sv_fig, "Segmentation"),
        ("Forecast", forecast_fig, f"{metric_option} forecast for {choice}"),
        ("Monthly", monthly_fig, "Monthly sales"),
        ("Treemap", treemap_fig, "Category vs sub-category profitability"),
        ("Shipping Efficiency", ship_fig, "Shipping lead time analysis"),
        ("Segment Distribution", segment_fig, "Customer segment revenue"),
        ("Geographical Map", geo_fig, "Sales distribution by city")
    ]

    for title, fig, desc in charts:
        if fig:
            story.append(Spacer(1, 10))
            story.append(Paragraph(title, styles["Heading2"]))
            img = fig_to_img(fig)
            story.append(Image(img, width=400, height=250))

            explanation = ai_explain(desc, df.describe().to_string())
            story.append(Paragraph(explanation.replace("\n","<br/>"), styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ================= DOWNLOAD =================

if "pdf_ready" not in st.session_state:
    st.session_state["pdf_ready"] = None

# STEP 1: Generate
if st.button("Generate Report", key="gen_btn"):
    try:
        with st.spinner("Generating AI Report..."):
            pdf = generate_pdf(metric_option, choice)

            st.session_state["pdf_ready"] = pdf
            st.success("✅ Report Ready!")

    except Exception as e:
        st.error(f"❌ Failed: {e}")

# STEP 2: Download (ONLY if ready)
if st.session_state["pdf_ready"] is not None:
    st.download_button(
        label="📄 Download Full AI Report",
        data=st.session_state["pdf_ready"],
        file_name="Retail_AI_Report.pdf",
        mime="application/pdf",
        key="download_pdf"
    )
