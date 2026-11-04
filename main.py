import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="센서 비교", layout="wide")

LABEL = "defect_label"
# 그래프 글자는 한글 폰트 문제를 피하려고 영문 사용
SENSORS = {
    "temperature_c": "Temperature",
    "pressure_torr": "Pressure",
    "gas_flow_sccm": "Gas flow",
    "etch_rate_nm_min": "Etch rate",
    "voltage_v": "Voltage",
    "current_ma": "Current",
}
RED, BLUE, GRAY = "#E45756", "#4C78A8", "#9aa0a6"

path = os.path.join(os.path.dirname(__file__), "data",
                    "semiconductor_wafer_defect_dataset_small.csv")
df = pd.read_csv(path)
cols = list(SENSORS)
names = list(SENSORS.values())
y = df[LABEL].values == 1
n, nd = len(df), int(y.sum())

# 센서마다 단위가 달라서, 평균 0 · 표준편차 1로 맞춘 값(z-score)으로 비교한다.
Z = (df[cols] - df[cols].mean()) / df[cols].std()
Z.columns = names

st.title("🔬 센서끼리 비교")
st.markdown(f"웨이퍼 **{n:,}개**, 결함 **{nd}건**. 센서마다 단위가 달라서 "
            "**평균 0, 표준편차 1로 맞춘 값(z-score)**으로 비교합니다. "
            "0은 평균, +2는 평균보다 표준편차 2배 높다는 뜻입니다.")

# ---------------------------------------------------------------- 1
st.header("1. 센서별 정상 vs 결함 분포")
fig, ax = plt.subplots(figsize=(11, 4.5))
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
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(pos, names)
ax.set_ylabel("z-score")
ax.set_title("Normal wafers (box) vs defect wafers (red dots)")
ax.legend(loc="lower right")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption("파란 상자는 정상 웨이퍼의 분포, 빨간 점은 결함 웨이퍼 각각의 값입니다. "
           "빨간 점이 상자에서 한쪽으로 치우쳐 있는 센서가 결함과 관련이 큰 센서입니다.")

# ---------------------------------------------------------------- 2
st.header("2. 결함 웨이퍼의 평균 위치 (센서 순위)")
m = Z[y].mean().sort_values(key=lambda s: s.abs())
band = 1.96 / np.sqrt(nd)
fig, ax = plt.subplots(figsize=(9, 4))
ax.axvspan(-band, band, color=GRAY, alpha=0.25, label="Range expected by chance")
ax.barh(m.index, m.values, color=[RED if v > 0 else BLUE for v in m.values])
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Mean z-score of defect wafers  (+ higher than average, - lower)")
ax.set_title("Which sensors differ most for defect wafers?")
ax.legend(loc="lower right")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption(f"회색 띠는 결함 {nd}건이 우연히 뽑혔을 때 평균이 놓일 만한 범위(약 95%)입니다. "
           "막대가 회색 띠 밖으로 나온 센서만 우연이 아닐 가능성이 있습니다. 결함이 적어 참고용입니다.")

# ---------------------------------------------------------------- 3
st.header("3. 결함 웨이퍼별 센서 프로파일")
fig, ax = plt.subplots(figsize=(11, 4.5))
for i in np.where(y)[0]:
    ax.plot(names, Z.iloc[i].values, color=RED, alpha=0.35, marker="o", ms=4)
ax.plot(names, Z[y].mean().values, color=RED, lw=3, marker="o", label="Defect mean")
ax.plot(names, np.zeros(len(names)), color=BLUE, lw=2, ls="--", label="All wafers mean (0)")
ax.axhline(0, color="black", lw=0.5)
ax.set_ylabel("z-score")
ax.set_title("Each defect wafer's sensor values (thin lines) and their mean (thick)")
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption("가는 선은 결함 웨이퍼 1건씩입니다. 선들이 같은 방향으로 모이는 센서는 결함에서 일관된 값을 보인다는 뜻입니다.")

# ---------------------------------------------------------------- 4
st.header("4. 센서 간 상관관계")
corr = Z.corr()
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
ax.set_yticks(range(len(names)), names)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption("값이 0에 가까우면 두 센서가 서로 거의 무관하다는 뜻입니다(±1에 가까울수록 함께 움직임).")

# ---------------------------------------------------------------- 5
st.header("5. 센서 쌍별 산점도")
k = len(cols)
fig, axes = plt.subplots(k, k, figsize=(12, 12))
for i in range(k):
    for j in range(k):
        ax = axes[i][j]
        if i == j:
            ax.hist(Z.iloc[:, i][~y], bins=25, color=BLUE, alpha=0.7)
            for v in Z.iloc[:, i][y]:
                ax.axvline(v, color=RED, lw=1)
        else:
            ax.scatter(Z.iloc[:, j][~y], Z.iloc[:, i][~y], s=4, color=GRAY, alpha=0.4)
            ax.scatter(Z.iloc[:, j][y], Z.iloc[:, i][y], s=40, color=RED,
                       edgecolor="white", zorder=3)
        ax.tick_params(labelsize=6)
        if i == k - 1:
            ax.set_xlabel(names[j], fontsize=8)
        if j == 0:
            ax.set_ylabel(names[i], fontsize=8)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption("회색 점은 정상, 빨간 점은 결함 웨이퍼입니다. 대각선은 각 센서의 분포(빨간 선은 결함 값)입니다. "
           "빨간 점이 한쪽 구석에 몰리는 조합을 찾아보세요.")
