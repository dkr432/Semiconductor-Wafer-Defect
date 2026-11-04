import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="센서 비교", layout="wide")

LABEL = "defect_label"
# (화면용 한글, 그래프용 영문)  ※ 그래프는 한글 폰트 문제를 피하려고 영문 사용
SENSORS = {
    "temperature_c": ("온도 (℃)", "Temperature (C)"),
    "pressure_torr": ("압력 (torr)", "Pressure (torr)"),
    "gas_flow_sccm": ("가스 유량 (sccm)", "Gas flow (sccm)"),
    "etch_rate_nm_min": ("식각률 (nm/min)", "Etch rate (nm/min)"),
    "voltage_v": ("전압 (V)", "Voltage (V)"),
    "current_ma": ("전류 (mA)", "Current (mA)"),
}
KO = {k: v[0] for k, v in SENSORS.items()}
EN = {k: v[1] for k, v in SENSORS.items()}
RED, BLUE, GRAY = "#E45756", "#4C78A8", "#9aa0a6"

path = os.path.join(os.path.dirname(__file__), "data",
                    "semiconductor_wafer_defect_dataset_small.csv")
df = pd.read_csv(path)
y = df[LABEL].values == 1
n, nd = len(df), int(y.sum())

# ------------------------------------------------------------ 사이드바
st.sidebar.header("센서 선택")
sel = st.sidebar.multiselect("비교할 센서", list(SENSORS), default=list(SENSORS),
                             format_func=KO.get)
norm = st.sidebar.radio("값 맞추기 방식",
                        ["z-score (평균 0, 표준편차 1)", "0~1 범위(min-max)"])
st.sidebar.caption("센서마다 단위가 달라서, 서로 비교하려면 값을 같은 척도로 맞춰야 합니다.")

st.title("🔬 센서끼리 비교")
st.markdown(f"웨이퍼 **{n:,}개** · 결함 **{nd}건** ({nd / n:.2%})")
if nd < 30:
    st.caption("⚠️ 결함이 적어서 그래프의 경향은 참고용입니다.")

if not sel:
    st.warning("왼쪽에서 비교할 센서를 1개 이상 선택하세요.")
    st.stop()


def scale(frame):
    if norm.startswith("z"):
        return (frame - df[list(SENSORS)].mean()) / df[list(SENSORS)].std()
    lo, hi = df[list(SENSORS)].min(), df[list(SENSORS)].max()
    return (frame - lo) / (hi - lo)


Z = scale(df[list(SENSORS)])[sel]
Z.columns = [EN[c] for c in sel]
names = list(Z.columns)
ylabel = "z-score" if norm.startswith("z") else "scaled value (0-1)"
center = 0 if norm.startswith("z") else None

# ------------------------------------------------------------ 1. 정상 vs 결함
st.header("1. 선택한 센서의 정상 vs 결함 분포")
fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(names) + 2), 4.5))
pos = np.arange(len(names))
ax.boxplot([Z.loc[~y, c] for c in names], positions=pos, widths=0.5,
           showfliers=False, patch_artist=True,
           boxprops=dict(facecolor="#dce6f2", edgecolor=BLUE),
           medianprops=dict(color=BLUE), whiskerprops=dict(color=BLUE),
           capprops=dict(color=BLUE))
rng = np.random.default_rng(0)
for i, c in enumerate(names):
    ax.scatter(i + rng.uniform(-0.15, 0.15, nd), Z.loc[y, c], color=RED, s=45,
               zorder=3, edgecolor="white", label="Defect wafer" if i == 0 else None)
if center is not None:
    ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(pos, names, rotation=15 if len(names) > 4 else 0)
ax.set_ylabel(ylabel)
ax.set_title("Normal wafers (box) vs defect wafers (red dots)")
ax.legend(loc="best")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption("파란 상자는 정상 웨이퍼, 빨간 점은 결함 웨이퍼 각각의 값입니다.")

# ------------------------------------------------------------ 2. 평균 위치
st.header("2. 결함 웨이퍼의 평균 위치 비교")
m = Z[y].mean() - Z[~y].mean()
m = m.sort_values(key=lambda s: s.abs())
fig, ax = plt.subplots(figsize=(8, 0.6 * len(names) + 1.8))
ax.barh(m.index, m.values, color=[RED if v > 0 else BLUE for v in m.values])
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel(f"Defect mean - normal mean ({ylabel})   + higher / - lower")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption("막대가 길수록 결함 웨이퍼와 정상 웨이퍼의 평균 차이가 큰 센서입니다. "
           "오른쪽(빨강)은 결함에서 더 높고, 왼쪽(파랑)은 더 낮습니다.")

# ------------------------------------------------------------ 3. 두 센서 산점도
st.header("3. 두 센서 직접 비교 (산점도)")
c1, c2 = st.columns(2)
all_cols = list(SENSORS)
x = c1.selectbox("X축 센서", all_cols, index=1, format_func=KO.get)
yy = c2.selectbox("Y축 센서", all_cols, index=0, format_func=KO.get)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(df.loc[~y, x], df.loc[~y, yy], s=8, color=GRAY, alpha=0.5, label="Normal")
ax.scatter(df.loc[y, x], df.loc[y, yy], s=70, color=RED, edgecolor="white",
           zorder=3, label="Defect")
for _, r in df[y].iterrows():
    ax.annotate(str(int(r["wafer_id"])) if "wafer_id" in df else "",
                (r[x], r[yy]), fontsize=7, xytext=(4, 4), textcoords="offset points")
ax.set_xlabel(EN[x])
ax.set_ylabel(EN[yy])
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption("빨간 점 옆의 숫자는 결함 웨이퍼의 wafer_id입니다. 원래 단위 그대로의 값입니다.")

# ------------------------------------------------------------ 4. 분포 겹쳐 보기
st.header("4. 두 센서의 분포 겹쳐 보기")
d1, d2 = st.columns(2)
a = d1.selectbox("센서 A", all_cols, index=1, format_func=KO.get, key="a")
b = d2.selectbox("센서 B", all_cols, index=0, format_func=KO.get, key="b")
ZZ = scale(df[list(SENSORS)])
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, s, col in zip(axes, [a, b], [BLUE, "#F58518"]):
    ax.hist(ZZ.loc[~y, s], bins=30, density=True, color=col, alpha=0.7, label="Normal")
    for i, v in enumerate(ZZ.loc[y, s]):
        ax.axvline(v, color=RED, lw=1.2, label="Defect wafer" if i == 0 else None)
    ax.set_title(EN[s])
    ax.set_xlabel(ylabel)
    ax.legend()
axes[0].set_ylabel("density")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption("두 센서를 같은 척도로 맞춰 나란히 그렸습니다. 빨간 선은 결함 웨이퍼의 값입니다.")

# ------------------------------------------------------------ 5. 상관 / 프로파일
st.header("5. 선택한 센서끼리의 관계")
if len(names) >= 2:
    corr = Z.corr()
    fig, ax = plt.subplots(figsize=(0.9 * len(names) + 3, 0.8 * len(names) + 2.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation between sensors")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("0에 가까우면 두 센서가 거의 무관하고, ±1에 가까울수록 함께(또는 반대로) 움직입니다.")

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(names) + 2), 4))
    for i in np.where(y)[0]:
        ax.plot(names, Z.iloc[i].values, color=RED, alpha=0.35, marker="o", ms=4)
    ax.plot(names, Z[y].mean().values, color=RED, lw=3, marker="o", label="Defect mean")
    ax.plot(names, Z[~y].mean().values, color=BLUE, lw=2, ls="--", label="Normal mean")
    ax.set_ylabel(ylabel)
    ax.set_title("Each defect wafer across the selected sensors")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("가는 빨간 선은 결함 웨이퍼 1건씩입니다. 선들이 모이는 센서는 결함에서 값이 일관됩니다.")
else:
    st.info("2개 이상 선택하면 센서 간 상관관계와 프로파일이 표시됩니다.")
