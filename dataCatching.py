import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import xgboost as xgb


st.set_page_config(page_title="Earthquake Dashboard", layout="wide")
st.title("🌎 Earthquake Dashboard (HDX / USGS CSV)")

# --- Load data ---
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")

    # Convert types
    for col in ["latitude", "longitude", "depth", "mag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse time if present
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

    df = df.dropna(subset=["latitude", "longitude", "depth", "mag"])
    return df

df = load_data("2.5_week.csv")

# --- Sidebar filters ---
st.sidebar.header("Filters")
mag_min, mag_max = float(df["mag"].min()), float(df["mag"].max())
mag_range = st.sidebar.slider("Magnitude range", mag_min, mag_max, (mag_min, mag_max))

depth_min, depth_max = float(df["depth"].min()), float(df["depth"].max())
depth_range = st.sidebar.slider("Depth range (km)", depth_min, depth_max, (depth_min, depth_max))

df_f = df[(df["mag"].between(*mag_range)) & (df["depth"].between(*depth_range))].copy()

# Target definition (same as your ML task)
threshold = st.sidebar.slider("Big quake threshold (mag)", 2.5, 6.0, 3.0, 0.1)
df_f["target"] = (df_f["mag"] > threshold).astype(int)

# --- KPIs ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Events", f"{len(df_f):,}")
col2.metric("Avg Mag", f"{df_f['mag'].mean():.2f}")
col3.metric("Avg Depth (km)", f"{df_f['depth'].mean():.1f}")
col4.metric(f"% > {threshold}", f"{100*df_f['target'].mean():.1f}%")

st.divider()

# --- Charts row 1 ---
c1, c2 = st.columns(2)

with c1:
    st.subheader("Magnitude distribution")
    fig = px.histogram(df_f, x="mag", nbins=30, color="target",
                       color_discrete_map={0:"#636EFA", 1:"#EF553B"})
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Depth vs Magnitude")
    fig = px.scatter(df_f, x="depth", y="mag", color="target",
                     hover_data=[col for col in ["place", "magType"] if col in df_f.columns])
    st.plotly_chart(fig, use_container_width=True)

# --- Map ---
st.subheader("Map of events")
# Streamlit’s built-in map expects columns named lat/lon
map_df = df_f.rename(columns={"latitude": "lat", "longitude": "lon"})
st.map(map_df[["lat", "lon"]])

# --- Time series if time exists ---
if "time" in df_f.columns and df_f["time"].notna().any():
    st.subheader("Events over time")
    ts = df_f.dropna(subset=["time"]).set_index("time").resample("1H").size().reset_index(name="count")
    fig = px.line(ts, x="time", y="count")
    st.plotly_chart(fig, use_container_width=True)

st.divider()


st.subheader("Data preview")
st.dataframe(df_f.head(50))

# Logging Plot to MLflow 

# plt.savefig("feature_importance.png")

# with mlflow.start_run():
    # fig, ax = plt.subplots()
    # ax.hist(df["mag"], bins=30)
    # ax.set_title("Magnitude Distribution")
    # ax.set_xlabel("Magnitude")
    # ax.set_ylabel("Frequency")

    # fig.savefig("mag_dist.png")
    # mlflow.log_artifact(fig, "mag_dist.png")

    # plt.close(fig)








