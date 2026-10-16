
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.caption("영화의 특성을 분석해 비슷한 영화끼리 유형으로 묶습니다.")

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

FEATURE_LABELS = {
    "log_first_scrn": "스크린 수(상용로그)",
    "log_total_audi": "누적 관객(상용로그)",
    "days_in_top10": "10위권 일수",
    "long_run_index": "롱런 지수",
}

FEATURES = list(FEATURE_LABELS.keys())

CLUSTER_SYMBOLS = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required = [
        "movieCd",
        "movieNm",
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            "필요한 열이 없습니다: " + ", ".join(missing)
        )

    for col in [
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 스크린 수와 누적 관객의 상용로그
    df["log_first_scrn"] = np.where(
        df["first_scrn"] > 0,
        np.log10(df["first_scrn"]),
        np.nan,
    )

    df["log_total_audi"] = np.where(
        df["total_audi"] > 0,
        np.log10(df["total_audi"]),
        np.nan,
    )

    # 롱런 지수: 누적 관객 / 첫 주 관객, 최대 20
    df["long_run_index"] = np.where(
        df["first_week_audi"] > 0,
        (
            df["total_audi"] / df["first_week_audi"]
        ).clip(upper=20),
        np.nan,
    )

    return df


try:
    original = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


# 네 속성 모두 값이 있는 영화만 사용
valid = original.dropna(subset=FEATURES).copy()

valid = valid[
    (valid["first_scrn"] > 0)
    & (valid["total_audi"] > 0)
    & (valid["first_week_audi"] > 0)
    & (valid["days_in_top10"].notna())
    & (valid["long_run_index"].notna())
].copy()

st.subheader("1. 분류 설정")

col_features, col_clusters = st.columns([3, 1])

with col_features:
    selected_features = st.multiselect(
        "묶는 데 사용할 속성 (두 개 이상 선택)",
        options=FEATURES,
        default=FEATURES,
        format_func=lambda x: FEATURE_LABELS[x],
    )

with col_clusters:
    n_clusters = st.selectbox(
        "묶음 수",
        options=list(range(2, 8)),
        index=1,
        help="2개부터 7개까지 선택할 수 있습니다.",
    )

if len(selected_features) < 2:
    st.warning("유형을 나누려면 속성을 두 개 이상 선택하세요.")
    st.stop()

if len(valid) < 7:
    st.error("묶음 수를 최대 7개까지 분석하려면 유효한 영화가 7편 이상 필요합니다.")
    st.stop()


# 선택한 속성을 표준화
scaler = StandardScaler()
scaled = scaler.fit_transform(valid[selected_features])


# 1~7개 묶음에 대한 중심 거리 제곱합 계산
inertia_values = []

for k in range(1, 8):
    inertia_model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    inertia_model.fit(scaled)
    inertia_values.append(inertia_model.inertia_)


# 사용자가 선택한 묶음 수로 최종 분류
model = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10,
)

valid["cluster_raw"] = model.fit_predict(scaled)


# 누적 관객 평균이 높은 유형부터 기호 지정
cluster_order = (
    valid.groupby("cluster_raw")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

cluster_labels = {
    cluster_id: CLUSTER_SYMBOLS[i]
    for i, cluster_id in enumerate(cluster_order)
}

valid["유형"] = valid["cluster_raw"].map(cluster_labels)

ordered_labels = [
    cluster_labels[cluster_id]
    for cluster_id in cluster_order
]


# 전체 편수와 분석 편수
st.markdown(
    f"**전체 영화 {len(original):,}편 · "
    f"유형 분류에 사용한 영화 {len(valid):,}편 · "
    f"현재 묶음 수 {n_clusters}개**"
)


# 실루엣 점수
silhouette = silhouette_score(
    scaled,
    valid["cluster_raw"],
)

st.metric(
    label=f"현재 {n_clusters}개 묶음의 실루엣 점수",
    value=f"{silhouette:.4f}",
)

st.caption(
    "실루엣 점수는 -1부터 1까지입니다. "
    "1에 가까울수록 같은 묶음 안의 영화는 서로 가깝고, "
    "다른 묶음의 영화와는 잘 구분되는 경향이 있습니다."
)


# --------------------------------------------------
# 2. 2차원 산점도
# --------------------------------------------------
st.divider()
st.subheader("2. 2차원 산점도")

col_x, col_y = st.columns(2)

with col_x:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        format_func=lambda x: FEATURE_LABELS[x],
        key="scatter_x",
    )

with col_y:
    y_feature = st.selectbox(
        "세로축 속성",
        options=selected_features,
        index=1,
        format_func=lambda x: FEATURE_LABELS[x],
        key="scatter_y",
    )

fig2 = px.scatter(
    valid,
    x=x_feature,
    y=y_feature,
    color="유형",
    hover_name="movieNm",
    category_orders={"유형": ordered_labels},
    labels={
        **FEATURE_LABELS,
        "유형": "영화 유형",
    },
    title=f"영화 유형 산점도 ({n_clusters}개 묶음)",
)

fig2.update_traces(
    marker={
        "size": 7,
        "opacity": 0.8,
    }
)

fig2.update_layout(
    legend_title_text="영화 유형",
    height=560,
)

st.plotly_chart(fig2, use_container_width=True)


# --------------------------------------------------
# 3. 3차원 산점도
# --------------------------------------------------
st.divider()
st.subheader("3. 3차원 산점도")

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
            key="x3",
        )

    with c2:
        y3 = st.selectbox(
            "y축 속성",
            selected_features,
            index=1,
            format_func=lambda x: FEATURE_LABELS[x],
            key="y3",
        )

    with c3:
        z3 = st.selectbox(
            "z축 속성",
            selected_features,
            index=2,
            format_func=lambda x: FEATURE_LABELS[x],
            key="z3",
        )

    fig3 = px.scatter_3d(
        valid,
        x=x3,
        y=y3,
        z=z3,
        color="유형",
        hover_name="movieNm",
        category_orders={"유형": ordered_labels},
        labels={
            **FEATURE_LABELS,
            "유형": "영화 유형",
        },
        title=f"영화 유형 3차원 산점도 ({n_clusters}개 묶음)",
    )

    fig3.update_traces(
        marker={
            "size": 3,
            "opacity": 0.8,
        }
    )

    fig3.update_layout(
        legend_title_text="영화 유형",
        height=680,
    )

    st.plotly_chart(fig3, use_container_width=True)


# --------------------------------------------------
# 4. 묶음 수에 따른 중심 거리 제곱합
# --------------------------------------------------
st.divider()
st.subheader("4. 묶음 수 정하기: 중심 거리 제곱합")

st.write(
    "가로축은 묶음 수, 세로축은 각 영화에서 해당 묶음의 중심까지 "
    "떨어진 거리의 제곱을 모두 더한 값입니다. "
    "묶음 수가 증가하면 이 값은 줄어듭니다."
)

inertia_df = pd.DataFrame({
    "묶음 수": list(range(1, 8)),
    "중심 거리 제곱합": inertia_values,
})

fig_inertia = px.line(
    inertia_df,
    x="묶음 수",
    y="중심 거리 제곱합",
    markers=True,
    title="묶음 수에 따른 중심 거리 제곱합",
    labels={
        "묶음 수": "묶음 수",
        "중심 거리 제곱합": "중심 거리 제곱합",
    },
)

fig_inertia.add_vline(
    x=n_clusters,
    line_dash="dash",
    line_width=2,
    annotation_text=f"현재 선택: {n_clusters}개",
    annotation_position="top",
)

fig_inertia.update_xaxes(
    tickmode="linear",
    dtick=1,
)

fig_inertia.update_layout(
    height=480,
)

st.plotly_chart(fig_inertia, use_container_width=True)


# 묶음 수별 제곱합과 직전 묶음 수 대비 감소량
inertia_table = inertia_df.copy()

inertia_table["직전 값 대비 감소량"] = (
    inertia_table["중심 거리 제곱합"].shift(1)
    - inertia_table["중심 거리 제곱합"]
)

inertia_table.loc[0, "직전 값 대비 감소량"] = np.nan

inertia_table["중심 거리 제곱합"] = (
    inertia_table["중심 거리 제곱합"].round(2)
)

inertia_table["직전 값 대비 감소량"] = (
    inertia_table["직전 값 대비 감소량"].round(2)
)

inertia_table = inertia_table.rename(
    columns={"묶음 수": "묶음 수"}
)

st.dataframe(
    inertia_table,
    hide_index=True,
    use_container_width=True,
)


# --------------------------------------------------
# 5. 유형별 요약표
# --------------------------------------------------
st.divider()
st.subheader("5. 유형별 요약")

summary = (
    valid.groupby("유형")
    .agg(
        편수=("movieNm", "size"),
        스크린수_평균=("first_scrn", "mean"),
        누적관객_평균=("total_audi", "mean"),
        십위권일수_평균=("days_in_top10", "mean"),
        롱런지수_평균=("long_run_index", "mean"),
    )
    .reindex(ordered_labels)
    .reset_index()
)

summary = summary.rename(
    columns={
        "유형": "영화 유형",
        "스크린수_평균": "스크린 수 평균",
        "누적관객_평균": "누적 관객 평균",
        "십위권일수_평균": "10위권 일수 평균",
        "롱런지수_평균": "롱런 지수 평균",
    }
)

summary["스크린 수 평균"] = summary["스크린 수 평균"].round(1)
summary["누적 관객 평균"] = summary["누적 관객 평균"].round(0)
summary["10위권 일수 평균"] = summary["10위권 일수 평균"].round(1)
summary["롱런 지수 평균"] = summary["롱런 지수 평균"].round(2)

st.dataframe(
    summary,
    hide_index=True,
    use_container_width=True,
)


# --------------------------------------------------
# 6. 유형별 누적 관객 TOP 5
# --------------------------------------------------
st.divider()
st.subheader("6. 유형별 누적 관객 TOP 5")

for label in ordered_labels:
    group = (
        valid[valid["유형"] == label]
        .sort_values("total_audi", ascending=False)
        .head(5)
    )

    with st.expander(
        f"{label} 유형 — 누적 관객이 많은 영화",
        expanded=True,
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
                range(1, len(top5) + 1),
            )

            st.dataframe(
                top5,
                hide_index=True,
                use_container_width=True,
            )


# --------------------------------------------------
# 7. 설명
# --------------------------------------------------
st.divider()
st.caption(
    "스크린 수와 누적 관객은 상용로그(log₁₀)를 적용했습니다. "
    "롱런 지수는 누적 관객을 첫 주 관객으로 나눈 값이며 "
    "최대 20으로 제한했습니다. "
    "선택한 속성을 표준화한 뒤 K-평균으로 분류하며, "
    "난수 시드는 42로 고정했습니다. "
    "중심 거리 제곱합 그래프는 현재 선택한 속성을 사용해 계산합니다. "
    "유형 기호는 누적 관객 평균이 높은 묶음부터 ㉮, ㉯, ㉰ 순으로 지정합니다."
)
