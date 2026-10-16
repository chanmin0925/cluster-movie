import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.caption("영화의 특성을 바탕으로 비슷한 영화끼리 세 유형으로 묶습니다.")

FEATURE_LABELS = {
    "log_first_scrn": "스크린 수(상용로그)",
    "log_total_audi": "누적 관객(상용로그)",
    "days_in_top10": "10위권 일수",
    "long_run_index": "롱런 지수",
}
FEATURES = list(FEATURE_LABELS.keys())

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    required = [
        "movieCd", "movieNm", "first_scrn", "first_week_audi",
        "total_audi", "days_in_top10"
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("필요한 열이 없습니다: " + ", ".join(missing))

    for col in ["first_scrn", "first_week_audi", "total_audi", "days_in_top10"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["log_first_scrn"] = np.where(df["first_scrn"] > 0, np.log10(df["first_scrn"]), np.nan)
    df["log_total_audi"] = np.where(df["total_audi"] > 0, np.log10(df["total_audi"]), np.nan)
    df["long_run_index"] = np.where(
        df["first_week_audi"] > 0,
        (df["total_audi"] / df["first_week_audi"]).clip(upper=20),
        np.nan
    )
    return df

try:
    original = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

valid = original.dropna(subset=FEATURES).copy()
valid = valid[
    (valid["first_scrn"] > 0)
    & (valid["total_audi"] > 0)
    & (valid["first_week_audi"] > 0)
    & (valid["days_in_top10"].notna())
    & (valid["long_run_index"].notna())
].copy()

st.markdown("### 분류에 사용할 속성")
selected_features = st.multiselect(
    "두 개 이상 선택하세요. 기본값은 네 속성 모두입니다.",
    options=FEATURES,
    default=FEATURES,
    format_func=lambda x: FEATURE_LABELS[x]
)

if len(selected_features) < 2:
    st.warning("K-평균으로 유형을 나누려면 속성을 두 개 이상 선택하세요.")
    st.stop()

if len(valid) < 3:
    st.error("유형을 세 개로 나누기에는 유효한 영화 데이터가 부족합니다.")
    st.stop()

scaler = StandardScaler()
scaled = scaler.fit_transform(valid[selected_features])
model = KMeans(n_clusters=3, random_state=42, n_init=10)
valid["cluster_raw"] = model.fit_predict(scaled)

# 누적 관객 평균이 큰 군집부터 ㉮, ㉯, ㉰를 부여
cluster_order = (
    valid.groupby("cluster_raw")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
cluster_labels = {cluster_id: ["㉮", "㉯", "㉰"][i] for i, cluster_id in enumerate(cluster_order)}
valid["유형"] = valid["cluster_raw"].map(cluster_labels)

st.markdown(f"**전체 영화 {len(original):,}편 · 유형 분류에 사용한 영화 {len(valid):,}편**")
