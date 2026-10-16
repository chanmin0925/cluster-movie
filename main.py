
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.caption("영화의 특성을 분석해 비슷한 영화끼리 세 유형으로 묶습니다.")

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

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
        "movieCd", "movieNm", "first_scrn",
        "first_week_audi", "total_audi", "days_in_top10"
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("필요한 열이 없습니다: " + ", ".join(missing))

    for col in [
        "first_scrn", "first_week_audi",
        "total_audi", "days_in_top10"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["log_first_scrn"] = np.where(
        df["first_scrn"] > 0,
        np.log10(df["first_scrn"]),
        np.nan
    )

    df["log_total_audi"] = np.where(
        df["total_audi"] > 0,
        np.log10(df["total_audi"]),
        np.nan
    )

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

st.subheader("분류에 사용할 속성")

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

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)
valid["cluster_raw"] = model.fit_predict(scaled)

cluster_order = (
    valid.groupby("cluster_raw")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

cluster_labels = {
    cluster_id: ["㉮", "㉯", "㉰"][i]
    for i, cluster_id in enumerate(cluster_order)
}
valid["유형"] = valid["cluster_raw"].map(cluster_labels)

st.markdown(
    f"**전체 영화 {len(original):,}편 · "
    f"유형 분류에 사용한 영화 {len(valid):,}편**"
)

st.divider()
st.subheader("2차원 산점도")

col_x, col_y = st.columns(2)

with col_x:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        format_func=lambda x: FEATURE_LABELS[x],
        key="scatter_x"
    )

with col_y:
    y_feature = st.selectbox(
        "세로축 속성",
        options=selected_features,
        index=1,
        format_func=lambda x: FEATURE_LABELS[x],
        key="scatter_y"
    )

fig2 = px.scatter(
    valid,
    x=x_feature,
    y=y_feature,
    color="유형",
    hover_name="movieNm",
    category_orders={"유형": ["㉮", "㉯", "㉰"]},
    labels={
        **FEATURE_LABELS,
        "유형": "영화 유형"
    },
    title="선택한 속성에 따른 영화 유형"
)

fig2.update_traces(
    marker={"size": 7, "opacity": 0.8}
)
fig2.update_layout(
    legend_title_text="영화 유형",
    height=560
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("3차원 산점도")

if len(selected_features) < 3:
    st.info(
        "3차원 산점도를 보려면 분류에 사용할 속성을 "
        "세 개 이상 선택하세요."
    )
else:
    c1, c2, c3 = st.columns(3)

    with c1:
        x3 = st.selectbox(
            "x축 속성",
            selected_features,
            index=0,
            format_func=lambda x: FEATURE_LABELS[x],
            key="x3"
        )

    with c2:
        y3 = st.selectbox(
            "y축 속성",
            selected_features,
            index=1,
            format_func=lambda x: FEATURE_LABELS[x],
            key="y3"
        )

    with c3:
        z3 = st.selectbox(
            "z축 속성",
            selected_features,
            index=2,
            format_func=lambda x: FEATURE_LABELS[x],
            key="z3"
        )

    fig3 = px.scatter_3d(
        valid,
        x=x3,
        y=y3,
        z=z3,
        color="유형",
        hover_name="movieNm",
        category_orders={"유형": ["㉮", "㉯", "㉰"]},
        labels={
            **FEATURE_LABELS,
            "유형": "영화 유형"
        },
        title="선택한 세 속성에 따른 영화 유형"
    )

    fig3.update_traces(
        marker={"size": 3, "opacity": 0.8}
    )
    fig3.update_layout(
        legend_title_text="영화 유형",
        height=680
    )
    st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.subheader("유형별 요약")

summary = (
    valid.groupby("유형")
    .agg(
        편수=("movieNm", "size"),
        스크린수_평균=("first_scrn", "mean"),
        누적관객_평균=("total_audi", "mean"),
        십위권일수_평균=("days_in_top10", "mean"),
        롱런지수_평균=("long_run_index", "mean"),
    )
    .reindex(["㉮", "㉯", "㉰"])
    .reset_index()
)

summary = summary.rename(columns={
    "유형": "영화 유형",
    "스크린수_평균": "스크린 수 평균",
    "누적관객_평균": "누적 관객 평균",
    "십위권일수_평균": "10위권 일수 평균",
    "롱런지수_평균": "롱런 지수 평균",
})

summary["스크린 수 평균"] = summary["스크린 수 평균"].round(1)
summary["누적 관객 평균"] = summary["누적 관객 평균"].round(0)
summary["10위권 일수 평균"] = summary["10위권 일수 평균"].round(1)
summary["롱런 지수 평균"] = summary["롱런 지수 평균"].round(2)

st.dataframe(
    summary,
    hide_index=True,
    use_container_width=True
)

st.divider()
st.subheader("유형별 누적 관객 TOP 5")

for label in ["㉮", "㉯", "㉰"]:
    group = (
        valid[valid["유형"] == label]
        .sort_values("total_audi", ascending=False)
        .head(5)
    )

    with st.expander(
        f"{label} 유형 — 누적 관객이 많은 영화",
        expanded=True
    ):
        if group.empty:
            st.write("해당 유형에 영화가 없습니다.")
        else:
            top5 = group[["movieNm", "total_audi"]].copy()
            top5.columns = ["영화 제목", "누적 관객"]
            top5["누적 관객"] = top5["누적 관객"].map(
                lambda n: f"{int(n):,}"
            )
            top5.insert(
                0,
                "순위",
                range(1, len(top5) + 1)
            )
            st.dataframe(
                top5,
                hide_index=True,
                use_container_width=True
            )

st.caption(
    "스크린 수와 누적 관객은 상용로그(log₁₀)를 적용했습니다. "
    "롱런 지수는 누적 관객을 첫 주 관객으로 나눈 값이며 "
    "최대 20으로 제한했습니다. "
    "선택한 속성을 표준화한 뒤 K-평균으로 세 유형을 분류했습니다. "
    "난수 시드는 42로 고정했습니다."
)
