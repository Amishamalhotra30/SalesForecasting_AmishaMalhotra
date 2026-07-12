import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from xgboost import XGBRegressor

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# GLOBAL STYLING
# ==========================================================

st.markdown("""
<style>
    /* Overall page */
    .main {
        padding-top: 1rem;
    }

    /* Headings */
    h1 {
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        font-weight: 700;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f5f7fb 100%);
        border: 1px solid #e6e9ef;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label {
        font-weight: 600;
        color: #5b6472;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border: 1px solid #e6e9ef;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #10131a;
    }
    section[data-testid="stSidebar"] * {
        color: #f0f1f5 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #2a2e3a;
    }

    /* Sidebar metric cards need their own dark styling, otherwise they
       inherit the light metric-card background from the main-content
       rule above and become unreadable against the forced white text. */
    section[data-testid="stSidebar"] div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 0.7rem 1rem;
        box-shadow: none;
    }
    section[data-testid="stSidebar"] div[data-testid="stMetric"] label {
        color: #a7adba !important;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700;
    }

    /* Chart containers */
    div[data-testid="stPlotlyChart"], div[data-testid="stPyplotGlobalUseContainerWidth"] {
        background: #ffffff;
        border-radius: 14px;
        padding: 0.5rem;
        border: 1px solid #eef0f4;
    }

    /* Section divider spacing */
    hr {
        margin: 1.6rem 0;
    }

    /* Badge-style caption */
    .section-caption {
        color: #6b7280;
        font-size: 0.92rem;
        margin-top: -0.4rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_white"
ACCENT_SEQUENCE = px.colors.qualitative.Set2

# ==========================================================
# HEADER
# ==========================================================

title_col, badge_col = st.columns([5, 1])
with title_col:
    st.title("📈 End-to-End Sales Forecasting & Demand Intelligence System")
    st.markdown(
        '<div class="section-caption">Interactive analytics · forecasting · anomaly detection · demand segmentation</div>',
        unsafe_allow_html=True
    )
with badge_col:
    st.markdown(
        "<div style='text-align:right; padding-top:1.6rem;'>"
        "<span style='background:#eef2ff;color:#4338ca;padding:6px 12px;border-radius:999px;"
        "font-size:0.8rem;font-weight:600;'>Live Dashboard</span></div>",
        unsafe_allow_html=True
    )

st.markdown("---")

# ==========================================================
# DATA LOADING
# ==========================================================

@st.cache_data
def load_data():

    BASE_DIR = Path(__file__).parent
    data_path = BASE_DIR / "train.csv"

    df = pd.read_csv(data_path)

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        dayfirst=True
    )

    return df
    
with st.spinner("Loading sales data..."):
    df = load_data()

# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.title("🧭 Navigation")
st.sidebar.caption("Choose a view to explore")

page = st.sidebar.radio(

    "Select Page",

    [

        "Sales Overview",

        "Forecast Explorer",

        "Anomaly Report",

        "Demand Segments"

    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Dataset Snapshot")
st.sidebar.metric("Total Records", f"{len(df):,}")
st.sidebar.metric("Date Range", f"{df['Order Date'].dt.year.min()}–{df['Order Date'].dt.year.max()}")
st.sidebar.metric("Total Sales", f"${df['Sales'].sum():,.0f}")

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · scikit-learn · XGBoost · Plotly")

# ==========================================================
# PAGE 1 : SALES OVERVIEW
# ==========================================================

if page == "Sales Overview":

    st.header("📊 Sales Overview Dashboard")
    st.markdown(
        '<div class="section-caption">High-level trends across years, months, regions and categories</div>',
        unsafe_allow_html=True
    )

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
    kpi2.metric("Total Orders", f"{df['Order ID'].nunique():,}")
    kpi3.metric("Avg. Order Value", f"${df['Sales'].sum() / df['Order ID'].nunique():,.2f}")

    st.markdown("---")

    # Yearly Sales
    yearly_sales = (
        df.groupby(df["Order Date"].dt.year)["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        yearly_sales,
        x="Order Date",
        y="Sales",
        title="Total Sales by Year",
        text_auto=".2s",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=ACCENT_SEQUENCE
    )
    fig.update_layout(xaxis_title="Year", yaxis_title="Sales", title_font_size=18)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Monthly Sales Trend

    monthly_sales = (
        df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
        .sum()
        .reset_index()
    )

    fig2 = px.line(
        monthly_sales,
        x="Order Date",
        y="Sales",
        title="Monthly Sales Trend",
        markers=True,
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=ACCENT_SEQUENCE
    )
    fig2.update_layout(xaxis_title="Month", yaxis_title="Sales", title_font_size=18)

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.subheader("🔍 Sales by Region and Category")

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        region = st.selectbox(
            "Select Region",
            sorted(df["Region"].unique())
        )

    with filter_col2:
        category = st.selectbox(
            "Select Category",
            sorted(df["Category"].unique())
        )

    filtered = df[
        (df["Region"] == region) &
        (df["Category"] == category)
    ]

    if filtered.empty:

        st.warning("No data available.")

    else:

        chart = (
            filtered
            .groupby("Sub-Category")["Sales"]
            .sum()
            .reset_index()
        )

        fig3 = px.bar(

            chart,

            x="Sub-Category",

            y="Sales",

            color="Sales",

            title=f"{category} Sales in {region}",
            template=PLOTLY_TEMPLATE,
            color_continuous_scale="Blues"

        )
        fig3.update_layout(title_font_size=18)

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        with st.expander("📄 View raw filtered records (first 20)", expanded=False):
            st.dataframe(filtered.head(20), use_container_width=True)

# ==========================================================
# PAGE 2 : FORECAST EXPLORER
# ==========================================================

elif page == "Forecast Explorer":

    st.header("📈 Forecast Explorer")
    st.markdown(
        '<div class="section-caption">XGBoost-powered short-term sales forecasting</div>',
        unsafe_allow_html=True
    )

    control_col1, control_col2, control_col3 = st.columns([1.2, 1.5, 1.5])

    with control_col1:
        option = st.radio(

            "Forecast By",

            ["Category", "Region"]

        )

    if option == "Category":

        with control_col2:
            selected = st.selectbox(

                "Choose Category",

                sorted(df["Category"].unique())

            )

        data = df[df["Category"] == selected]

    else:

        with control_col2:
            selected = st.selectbox(

                "Choose Region",

                sorted(df["Region"].unique())

            )

        data = df[df["Region"] == selected]

    with control_col3:
        horizon = st.slider(

            "Forecast Horizon (Months)",

            min_value=1,

            max_value=3,

            value=3

        )

    st.markdown("---")

    monthly = (
        data.groupby(
            pd.Grouper(
                key="Order Date",
                freq="ME"
            )
        )["Sales"]
        .sum()
    )

    ts = monthly.to_frame(name="Sales")

    ts["Lag1"] = ts["Sales"].shift(1)
    ts["Lag2"] = ts["Sales"].shift(2)
    ts["Lag3"] = ts["Sales"].shift(3)

    ts["RollingMean"] = (
        ts["Sales"]
        .rolling(3)
        .mean()
    )

    ts["Month"] = ts.index.month
    ts["Quarter"] = ts.index.quarter

    ts = ts.dropna()

    X = ts.drop(columns="Sales")
    y = ts["Sales"]

    X_train = X[:-3]
    X_test = X[-3:]

    y_train = y[:-3]
    y_test = y[-3:]

    model = XGBRegressor(

        n_estimators=300,

        learning_rate=0.05,

        max_depth=4,

        random_state=42,

        objective="reg:squarederror"

    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    mae = np.mean(np.abs(y_test - prediction))

    rmse = np.sqrt(
        np.mean((y_test - prediction)**2)
    )

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("MAE", f"{mae:,.2f}")
    metric_col2.metric("RMSE", f"{rmse:,.2f}")

    st.markdown("---")

    plt.style.use("seaborn-v0_8-whitegrid") if "seaborn-v0_8-whitegrid" in plt.style.available else None

    plt.figure(figsize=(10,5))

    plt.plot(

        y_train.index,

        y_train,

        label="Training",
        color="#6b7280"

    )

    plt.plot(

        y_test.index,

        y_test,

        marker="o",

        label="Actual",
        color="#2563eb"

    )

    plt.plot(

        y_test.index,

        prediction,

        marker="o",

        linestyle="--",

        label="Forecast",
        color="#f97316"

    )

    plt.title(f"{selected} Sales Forecast", fontsize=14, fontweight="bold")

    plt.legend()

    st.pyplot(plt)

    st.subheader("📋 Forecast Values")

    forecast_table = pd.DataFrame({

        "Month": y_test.index,

        "Predicted Sales": prediction.round(2)

    })

    st.dataframe(forecast_table, use_container_width=True)

# ==========================================================
# PAGE 3 : ANOMALY REPORT
# ==========================================================

elif page == "Anomaly Report":

    st.header("🚨 Sales Anomaly Report")
    st.markdown(
        '<div class="section-caption">Isolation Forest-based detection of unusual weekly sales activity</div>',
        unsafe_allow_html=True
    )

    weekly_sales = (
        df.groupby(
            pd.Grouper(
                key="Order Date",
                freq="W"
            )
        )["Sales"]
        .sum()
        .reset_index()
    )

    iso = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    weekly_sales["Isolation"] = iso.fit_predict(
        weekly_sales[["Sales"]]
    )

    weekly_sales["Anomaly"] = (
        weekly_sales["Isolation"] == -1
    )

    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(
        weekly_sales["Order Date"],
        weekly_sales["Sales"],
        linewidth=2,
        label="Weekly Sales",
        color="#2563eb"
    )

    ax.scatter(
        weekly_sales.loc[
            weekly_sales["Anomaly"],
            "Order Date"
        ],
        weekly_sales.loc[
            weekly_sales["Anomaly"],
            "Sales"
        ],
        color="#dc2626",
        s=80,
        label="Anomaly",
        zorder=5
    )

    ax.set_title("Weekly Sales Anomaly Detection", fontsize=14, fontweight="bold")

    ax.set_xlabel("Date")

    ax.set_ylabel("Sales")

    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    st.markdown("---")

    result_col1, result_col2 = st.columns([1, 2])

    with result_col1:
        st.metric(
            "Total Anomalies Detected",
            len(weekly_sales[weekly_sales["Anomaly"]])
        )

    with result_col2:
        st.subheader("Detected Anomalies")

        anomalies = weekly_sales[
            weekly_sales["Anomaly"]
        ][["Order Date","Sales"]]

        st.dataframe(anomalies, use_container_width=True)

    st.markdown("---")

    with st.container():
        st.markdown("""
### 💡 Business Interpretation

- Sudden spikes may indicate festive sales, discounts or promotional campaigns.
- Sudden drops may indicate supply shortages, logistics issues or seasonal demand changes.
- Early anomaly detection helps improve inventory planning and business decision making.
""")

# ==========================================================
# PAGE 4 : PRODUCT DEMAND SEGMENTS
# ==========================================================

elif page == "Demand Segments":

    st.header("📦 Product Demand Segmentation")
    st.markdown(
        '<div class="section-caption">KMeans clustering of sub-categories by sales, growth, volatility and order value</div>',
        unsafe_allow_html=True
    )

    # Aggregate data at Sub-Category level
    segment = df.groupby("Sub-Category").agg({

        "Sales": "sum",
        "Order ID": "count"

    }).rename(columns={"Order ID": "Orders"})

    segment["Average_Order_Value"] = (
        segment["Sales"] /
        segment["Orders"]
    )

    # Monthly Sales
    monthly = df.groupby([
        pd.Grouper(key="Order Date", freq="ME"),
        "Sub-Category"
    ])["Sales"].sum().reset_index()

    growth = []
    volatility = []

    for sub in segment.index:

        temp = monthly[
            monthly["Sub-Category"] == sub
        ]

        growth.append(
            temp["Sales"].pct_change().mean()
        )

        volatility.append(
            temp["Sales"].std()
        )

    segment["Growth_Rate"] = growth
    segment["Volatility"] = volatility

    segment = segment.fillna(0)

    features = segment[[
        "Sales",
        "Growth_Rate",
        "Volatility",
        "Average_Order_Value"
    ]]

    scaler = StandardScaler()

    scaled = scaler.fit_transform(features)

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    segment["Cluster"] = kmeans.fit_predict(scaled)

    pca = PCA(n_components=2)

    pca_result = pca.fit_transform(scaled)

    segment["PCA1"] = pca_result[:,0]
    segment["PCA2"] = pca_result[:,1]

    labels = {
        0: "High Volume Stable Demand",
        1: "Growing Demand",
        2: "Low Volume High Volatility",
        3: "Declining Demand"
    }

    segment["Demand Segment"] = (
        segment["Cluster"].map(labels)
    )

    fig = px.scatter(

        segment,

        x="PCA1",

        y="PCA2",

        color="Demand Segment",

        text=segment.index,

        title="Product Demand Segments",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=ACCENT_SEQUENCE

    )
    fig.update_traces(textposition="top center", marker=dict(size=12, line=dict(width=1, color="white")))
    fig.update_layout(title_font_size=18)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("📋 Demand Segment Details")

    st.dataframe(

        segment[[
            "Sales",
            "Growth_Rate",
            "Volatility",
            "Average_Order_Value",
            "Demand Segment"
        ]],
        use_container_width=True

    )

    st.markdown("---")

    strategy = pd.DataFrame({

        "Demand Segment":[

            "High Volume Stable Demand",

            "Growing Demand",

            "Low Volume High Volatility",

            "Declining Demand"

        ],

        "Recommended Stocking Strategy":[

            "Maintain high inventory with regular replenishment.",

            "Increase inventory gradually based on demand.",

            "Maintain safety stock and monitor frequently.",

            "Reduce inventory and review product assortment."

        ]

    })

    st.subheader("📌 Recommended Stocking Strategies")
    st.table(strategy)
