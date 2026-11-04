import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="웨이퍼 결함 분석", layout="wide")

SENSORS = {
    "temperature_c": "온도 (℃)",
    "pressure_torr": "압력 (torr)",
    "gas_flow_sccm": "가스 유량 (sccm)",
    "etch_rate_nm_min": "식각률 (nm/min)",
    "voltage_v": "전압 (V)",
    "current_ma": "전류 (mA)",
}
LABEL = "defect_label"
COLORS = {"정상": "#4C78A8", "결함": "#E45756"}
DEFAULT_FILE = os.path.join(os.path.dirname(__file__), "data", "semiconductor_wafer_defect_dataset_small.csv")


@st.cache_data
def load(src):
    df = pd.read_csv(src)
    df["상태"] = np.where(df[LABEL] == 1, "결함", "정상")
    return df


# ---------- 데이터 불러오기 ----------
st.title("🔬 반도체 웨이퍼 센서 · 결함 분석")
up = st.sidebar.file_uploader("CSV 업로드 (없으면 샘플 사용)", type="csv")
if up is not None:
    df = load(up)
elif os.path.exists(DEFAULT_FILE):
    df = load(DEFAULT_FILE)
else:
    st.info("왼쪽에서 CSV 파일을 업로드해 주세요.")
    st.stop()

missing = [c for c in [*SENSORS, LABEL] if c not in df.columns]
if missing:
    st.error(f"필요한 열이 없습니다: {missing}")
    st.stop()

# ---------- 필터 ----------
st.sidebar.header("필터")
if "process_step" in df.columns:
    steps = sorted(df["process_step"].dropna().unique())
    sel = st.sidebar.multiselect("공정 단계", steps, default=steps)
    df = df[df["process_step"].isin(sel)]
if df.empty:
    st.warning("선택한 조건에 맞는 데이터가 없습니다.")
    st.stop()

n, nd = len(df), int(df[LABEL].sum())
c1, c2, c3 = st.columns(3)
c1.metric("웨이퍼 수", f"{n:,}")
c2.metric("결함 수", f"{nd:,}")
c3.metric("결함 비율", f"{nd / n:.2%}")
if nd < 10:
    st.caption("⚠️ 결함 샘플이 매우 적어 그래프의 경향은 참고용으로만 보세요.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["분포 비교", "박스플롯", "산점도", "압력 구간별 결함률", "센서 비교 · 상관", "결측치"]
)

# ---------- 1. 분포 ----------
with tab1:
    s = st.selectbox("센서", list(SENSORS), format_func=SENSORS.get, key="hist")
    bins = st.slider("구간 수", 10, 80, 30)
    fig = px.histogram(df, x=s, color="상태", nbins=bins, barmode="overlay",
                       histnorm="probability density", opacity=0.6,
                       color_discrete_map=COLORS, labels={s: SENSORS[s]})
    for v in df.loc[df[LABEL] == 1, s]:
        fig.add_vline(x=v, line_width=1, line_dash="dot", line_color=COLORS["결함"])
    fig.update_layout(yaxis_title="밀도 (클래스별 정규화)")
    st.plotly_chart(fig)
    st.caption("점선은 결함 웨이퍼의 실제 값입니다. 결함이 적어서 밀도는 클래스별로 정규화했습니다.")

# ---------- 2. 박스플롯 ----------
with tab2:
    cols = st.columns(3)
    for i, s in enumerate(SENSORS):
        f = px.box(df, x="상태", y=s, color="상태", points="all",
                   color_discrete_map=COLORS, labels={s: SENSORS[s]},
                   category_orders={"상태": ["정상", "결함"]})
        f.update_layout(showlegend=False, height=320, margin=dict(t=30, b=10),
                        title=SENSORS[s])
        f.update_traces(jitter=0.4, marker=dict(size=4))
        cols[i % 3].plotly_chart(f)

# ---------- 3. 산점도 ----------
with tab3:
    a, b = st.columns(2)
    x = a.selectbox("X축", list(SENSORS), index=1, format_func=SENSORS.get)
    y = b.selectbox("Y축", list(SENSORS), index=0, format_func=SENSORS.get)
    d = df.sort_values(LABEL)  # 결함 점이 위에 그려지도록
    fig = px.scatter(d, x=x, y=y, color="상태", color_discrete_map=COLORS,
                     opacity=0.7, hover_data=["wafer_id"] + (["process_step"] if "process_step" in d else []),
                     labels={x: SENSORS[x], y: SENSORS[y]})
    fig.update_traces(marker=dict(size=6))
    fig.update_traces(selector=dict(name="결함"), marker=dict(size=12, symbol="diamond"))
    st.plotly_chart(fig)

# ---------- 4. 압력 구간별 결함률 ----------
with tab4:
    s = st.selectbox("센서", list(SENSORS), index=1, format_func=SENSORS.get, key="bin")
    k = st.slider("분위수 구간 수", 3, 10, 5)
    q = pd.qcut(df[s], k, duplicates="drop")
    g = df.groupby(q, observed=True)[LABEL].agg(["count", "sum"]).reset_index()
    g["결함률(%)"] = g["sum"] / g["count"] * 100
    g["구간"] = g[s].astype(str)
    fig = px.bar(g, x="구간", y="결함률(%)", text="sum",
                 hover_data={"count": True, "sum": True},
                 labels={"count": "웨이퍼 수", "sum": "결함 수"})
    fig.update_traces(marker_color=COLORS["결함"], texttemplate="결함 %{text}건")
    fig.update_layout(xaxis_title=f"{SENSORS[s]} 구간 (낮음 → 높음)")
    st.plotly_chart(fig)

# ---------- 5. 센서 비교 / 상관 ----------
with tab5:
    rows = []
    for s in SENSORS:
        auc = roc_auc_score(df[LABEL], df[s]) if 0 < nd < n else np.nan
        z = (df.loc[df[LABEL] == 1, s].mean() - df[s].mean()) / df[s].std()
        rows.append({"센서": SENSORS[s], "단일 AUC": auc,
                     "결함 평균의 z-score": z,
                     "방향": "높을수록 결함" if auc > 0.5 else "낮을수록 결함"})
    t = pd.DataFrame(rows)
    t["분리도 |AUC-0.5|"] = (t["단일 AUC"] - 0.5).abs()
    t = t.sort_values("분리도 |AUC-0.5|", ascending=False)
    l, r = st.columns([1, 1])
    with l:
        st.subheader("센서별 결함 구분력")
        st.dataframe(t.round(3), hide_index=True)
        st.caption("AUC 0.5 = 구분 못 함, 1에 가까우면 높을수록 결함, 0에 가까우면 낮을수록 결함.")
    with r:
        st.subheader("센서 간 상관계수")
        corr = df[list(SENSORS)].corr().rename(index=SENSORS, columns=SENSORS)
        st.plotly_chart(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                                  color_continuous_scale="RdBu_r"))

with st.expander("결함 웨이퍼 원본 데이터"):
    st.dataframe(df[df[LABEL] == 1].drop(columns="상태"), hide_index=True)

# ---------- 6. 결측치 ----------
with tab6:
    data = df.drop(columns="상태")
    miss = data.isna().sum().reset_index()
    miss.columns = ["열", "결측 수"]
    miss["결측률(%)"] = miss["결측 수"] / len(data) * 100

    if miss["결측 수"].sum() == 0:
        st.success("결측치가 없습니다.")

    fig = px.bar(miss, x="열", y="결측률(%)", text="결측 수")
    fig.update_traces(marker_color="#E45756", texttemplate="%{text}개")
    fig.update_layout(yaxis_title="결측률 (%)", yaxis_range=[0, max(5, miss["결측률(%)"].max() * 1.2)])
    st.plotly_chart(fig)
    st.caption("막대가 높을수록 그 열에 빈 값이 많다는 뜻입니다.")

    st.subheader("결측 위치 (빨간색 = 빈 값)")
    fig2 = px.imshow(data.isna().astype(int).T, aspect="auto",
                     color_continuous_scale=[[0, "#dce6f2"], [1, "#E45756"]],
                     labels={"x": "행 번호", "y": "열"})
    fig2.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig2)
    st.caption("가로로 길게 빨간 띠가 있으면 그 열이 많이 비어 있고, 세로로 한꺼번에 빨간 줄이 생기면 특정 행에서 여러 센서가 동시에 빠진 것입니다.")

st.subheader("열별 결측치 개수")
miss = df.drop(columns="상태").isna().sum()
fig = px.bar(x=miss.index, y=miss.values, labels={"x": "열", "y": "결측 수"})
fig.update_traces(marker_color="#E45756")
st.plotly_chart(fig)
