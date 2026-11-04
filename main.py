import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="웨이퍼 결함 분류 모델", layout="wide")

LABEL = "defect_label"
STEP = "process_step"
# 열 이름: (화면 표시용 한글, 그래프용 영문)  ※ 그래프는 한글 폰트 문제를 피하려고 영문 사용
SENSORS = {
    "temperature_c": ("온도", "Temperature (C)"),
    "pressure_torr": ("압력", "Pressure (torr)"),
    "gas_flow_sccm": ("가스 유량", "Gas flow (sccm)"),
    "etch_rate_nm_min": ("식각률", "Etch rate (nm/min)"),
    "voltage_v": ("전압", "Voltage (V)"),
    "current_ma": ("전류", "Current (mA)"),
}
DEFAULT_FILE = os.path.join(os.path.dirname(__file__), "data",
                            "semiconductor_wafer_defect_dataset_small.csv")
RED, BLUE, GRAY = "#E45756", "#4C78A8", "#888888"


# ------------------------------------------------------------------ 모델/검증
def make_pipe(model_name, C, balanced, num_cols, use_step):
    transformers = [("num", StandardScaler(), list(num_cols))]
    if use_step:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), [STEP]))
    pre = ColumnTransformer(transformers)
    if model_name == "로지스틱 회귀":
        clf = LogisticRegression(C=C, max_iter=2000,
                                 class_weight="balanced" if balanced else None)
    else:
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                     random_state=0, n_jobs=1,
                                     class_weight="balanced_subsample" if balanced else None)
    return Pipeline([("pre", pre), ("clf", clf)])


@st.cache_data(show_spinner="교차검증 중...")
def run_cv(X, y, model_name, C, balanced, num_cols, use_step, folds, repeats):
    """층화 K-겹 교차검증을 repeats번 반복해 '학습에 쓰지 않은' 예측 확률을 모은다."""
    oof = np.zeros((repeats, len(y)))
    for r in range(repeats):
        skf = StratifiedKFold(folds, shuffle=True, random_state=r)
        for tr, te in skf.split(X, y):
            pipe = make_pipe(model_name, C, balanced, num_cols, use_step)
            pipe.fit(X.iloc[tr], y[tr])
            oof[r, te] = pipe.predict_proba(X.iloc[te])[:, 1]
    return oof


@st.cache_data(show_spinner="전체 데이터로 학습 중...")
def fit_full(X, y, model_name, C, balanced, num_cols, use_step):
    pipe = make_pipe(model_name, C, balanced, num_cols, use_step)
    pipe.fit(X, y)
    return pipe


def metrics_at(y, p, t):
    pred = (p >= t).astype(int)
    return {
        "PR-AUC": average_precision_score(y, p),
        "ROC-AUC": roc_auc_score(y, p),
        "재현율(Recall)": recall_score(y, pred, zero_division=0),
        "정밀도(Precision)": precision_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "균형 정확도": balanced_accuracy_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
    }


# ------------------------------------------------------------------ 데이터
@st.cache_data
def load(src):
    return pd.read_csv(src)


st.title("🔬 반도체 웨이퍼 결함 분류 모델")
up = st.sidebar.file_uploader("CSV 업로드 (없으면 샘플 사용)", type="csv")
if up is not None:
    df = load(up)
elif os.path.exists(DEFAULT_FILE):
    df = load(DEFAULT_FILE)
else:
    st.info("왼쪽에서 CSV 파일을 업로드해 주세요.")
    st.stop()

need = [c for c in [*SENSORS, LABEL] if c not in df.columns]
if need:
    st.error(f"필요한 열이 없습니다: {need}")
    st.stop()

y = df[LABEL].astype(int).values
n, n_pos = len(df), int(y.sum())
if n_pos < 2 or n_pos == n:
    st.error("결함(1)이 최소 2건, 정상(0)도 있어야 모델을 평가할 수 있습니다.")
    st.stop()

# ------------------------------------------------------------------ 사이드바
st.sidebar.header("모델 설정")
model_name = st.sidebar.selectbox("모델", ["로지스틱 회귀", "랜덤포레스트"])
feats = st.sidebar.multiselect("입력 센서", list(SENSORS), default=list(SENSORS),
                               format_func=lambda c: SENSORS[c][0])
use_step = st.sidebar.checkbox("공정 단계(process_step) 포함", value=False,
                               disabled=STEP not in df.columns)
balanced = st.sidebar.checkbox("클래스 가중치 보정 (불균형 대응)", value=True)
C = 1.0
if model_name == "로지스틱 회귀":
    C = st.sidebar.select_slider("규제 강도 C (작을수록 강한 규제)",
                                 options=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0], value=1.0)
max_folds = max(2, min(7, n_pos))
folds = st.sidebar.slider("교차검증 겹 수", 2, max_folds, min(5, max_folds))
repeats = st.sidebar.slider("반복 횟수 (결과 안정성 확인)", 1, 10, 3)
thr = st.sidebar.slider("판정 임계값 (이 확률 이상이면 결함)", 0.05, 0.95, 0.5, 0.05)

if not feats:
    st.warning("입력 센서를 1개 이상 선택하세요.")
    st.stop()

use_step = bool(use_step and STEP in df.columns)
cols = list(feats) + ([STEP] if use_step else [])
X = df[cols].copy()
num_cols = tuple(feats)

st.markdown(f"**웨이퍼 {n:,}개** · **결함 {n_pos}건** ({n_pos / n:.2%}) · "
            f"교차검증 {folds}겹 × {repeats}회")
if n_pos < 30:
    st.warning(f"결함이 {n_pos}건뿐이라 결과가 매우 불안정합니다. 경향 확인용으로만 보세요. "
               "정확도(accuracy)는 모두 '정상'이라고 답해도 높게 나오므로 쓰지 않았습니다.")

oof = run_cv(X, y, model_name, C, balanced, num_cols, use_step, folds, repeats)
mean_p = oof.mean(axis=0)

tab1, tab2, tab3, tab4 = st.tabs(["성능 지표", "ROC · PR 곡선", "센서 영향", "예측 확률"])

# ------------------------------------------------------------------ 1. 지표
with tab1:
    rows = pd.DataFrame([metrics_at(y, oof[r], thr) for r in range(repeats)])
    summary = pd.DataFrame({"평균": rows.mean(), "표준편차": rows.std(ddof=0)}).round(3)
    st.subheader(f"교차검증 성능 (임계값 {thr:.2f})")
    st.table(summary)
    st.caption(f"PR-AUC의 기준선(무작위 모델)은 결함 비율 {n_pos / n:.3f}입니다. "
               "표준편차가 크면 데이터 분할에 따라 결과가 크게 바뀐다는 뜻입니다.")

    cm = confusion_matrix(y, (mean_p >= thr).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Pred normal", "Pred defect"])
    ax.set_yticks([0, 1], ["Actual normal", "Actual defect"])
    ax.set_title("Confusion matrix (repeat-averaged prob.)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ------------------------------------------------------------------ 2. 곡선
with tab2:
    c1, c2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for r in range(repeats):
        fpr, tpr, _ = roc_curve(y, oof[r])
        ax.plot(fpr, tpr, color=BLUE, alpha=0.25, lw=1)
    fpr, tpr, _ = roc_curve(y, mean_p)
    ax.plot(fpr, tpr, color=BLUE, lw=2.5, label=f"mean prob. AUC={roc_auc_score(y, mean_p):.3f}")
    ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    c1.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    for r in range(repeats):
        pr, rc, _ = precision_recall_curve(y, oof[r])
        ax.plot(rc, pr, color=RED, alpha=0.25, lw=1)
    pr, rc, _ = precision_recall_curve(y, mean_p)
    ax.plot(rc, pr, color=RED, lw=2.5,
            label=f"mean prob. AP={average_precision_score(y, mean_p):.3f}")
    ax.axhline(n_pos / n, ls="--", color=GRAY, lw=1, label=f"baseline={n_pos / n:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    c2.pyplot(fig)
    plt.close(fig)
    st.caption("연한 선은 반복 1회마다의 곡선, 굵은 선은 반복 평균 확률의 곡선입니다. "
               "연한 선들이 많이 퍼져 있을수록 결과가 불안정합니다.")

# ------------------------------------------------------------------ 3. 센서 영향
with tab3:
    pipe = fit_full(X, y, model_name, C, balanced, num_cols, use_step)
    names = [s.split("__", 1)[1] for s in pipe.named_steps["pre"].get_feature_names_out()]
    clf = pipe.named_steps["clf"]
    label_en = {k: v[1] for k, v in SENSORS.items()}
    show = [label_en.get(nm, nm) for nm in names]

    if model_name == "로지스틱 회귀":
        vals, title = clf.coef_[0], "Standardized coefficient (+ : higher -> more defect)"
    else:
        vals, title = clf.feature_importances_, "Feature importance (direction not shown)"
    order = np.argsort(np.abs(vals))
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(vals) + 1.6))
    colors = [RED if v > 0 else BLUE for v in vals[order]] if model_name == "로지스틱 회귀" else RED
    ax.barh(np.array(show)[order], vals[order], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("센서를 표준화해서 넣었기 때문에 계수의 크기를 센서끼리 비교할 수 있습니다. "
               "결함이 적으면 계수가 부풀려지므로 절대값보다 순위와 부호를 참고하세요.")

    st.subheader("센서 값에 따른 결함 확률 (나머지 센서는 중앙값 고정)")
    base = {c: X[c].median() for c in feats}
    if use_step:
        base[STEP] = X[STEP].mode().iloc[0]
    k = len(feats)
    ncol = 3
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.2 * nrow), squeeze=False)
    for i, s in enumerate(feats):
        ax = axes[i // ncol][i % ncol]
        grid = np.linspace(X[s].quantile(0.01), X[s].quantile(0.99), 60)
        g = pd.DataFrame({c: [base[c]] * len(grid) for c in cols})
        g[s] = grid
        p = pipe.predict_proba(g[cols])[:, 1]
        ax.plot(grid, p, color=RED, lw=2)
        ax.plot(X.loc[y == 1, s], np.full(n_pos, -0.03), "|", color="black", ms=12,
                transform=ax.get_xaxis_transform())
        ax.set_xlabel(label_en[s])
        ax.set_ylabel("P(defect)")
        ax.set_ylim(0, 1.02)
    for j in range(k, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("아래쪽 검은 눈금(|)은 실제 결함 웨이퍼의 센서 값 위치입니다. "
               "곡선이 오른쪽으로 갈수록 올라가면 '값이 높을수록 결함 확률 증가', 내려가면 반대입니다.")

# ------------------------------------------------------------------ 4. 예측 확률
with tab4:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bins = np.linspace(0, 1, 26)
    ax.hist(mean_p[y == 0], bins=bins, color=BLUE, alpha=0.7, label="Normal")
    ax.hist(mean_p[y == 1], bins=bins, color=RED, alpha=0.9, label="Defect")
    ax.axvline(thr, color="black", ls="--", lw=1, label=f"threshold={thr:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted defect probability (out-of-fold, repeat mean)")
    ax.set_ylabel("Wafers (log scale)")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    rank = pd.Series(mean_p).rank(ascending=False, method="min").astype(int)
    out = df.assign(**{"예측 확률": mean_p.round(3), "확률 순위": rank.values})
    top = out[out[LABEL] == 1].sort_values("확률 순위")
    show_cols = [c for c in ["wafer_id", STEP] if c in top.columns] + \
                [LABEL, "예측 확률", "확률 순위"]
    st.subheader(f"실제 결함 웨이퍼의 예측 확률 순위 (전체 {n:,}개 중)")
    st.table(top[show_cols].head(20).reset_index(drop=True))
    st.caption("결함 웨이퍼가 순위 상위에 몰려 있으면 모델이 잘 구분한다는 뜻입니다.")
