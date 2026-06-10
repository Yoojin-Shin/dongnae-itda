"""
============================================================================
streamlit_app.py — 동네잇다 (Dongnae:itda) v2.0 (발표용 UI)
============================================================================
지시서 §3~§5 반영:
  - 사이드바 = 입력 (S1 카드 그리드 / S2 우선순위 / S4 슬라이더 / 데모 프리셋)
  - 메인 = 결과 (Top 5 카드 v2 / 상세 / DNA 탐색)
  - 디자인 토큰(theme.py) + Plotly 시각화 + 발표용 타이포
============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from household_templates import HOUSEHOLD_TEMPLATES
from priority_groups import PRIORITY_GROUPS
from data_loader import load_data, get_feature_list_and_weights
from matching_engine import (
    recommend, similar_dong, generate_explanation, DNA_LABELS,
)
from value_estimator import (
    load_value_predictions, get_value_for_dong, load_model_info,
)
from theme import (
    inject_css, COLORS, CLUSTER_COLORS, CONF_COLORS, VAL_COLORS, GLOSSARY,
    progress_bar, range_bar, conf_badge, val_badge, dna_badge,
)
from map_view import load_geojson, render_seoul_dna_map

# ----------------------------------------------------------------------------
# 페이지 설정 + CSS
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="동네잇다",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# ----------------------------------------------------------------------------
# 데이터 로드 (캐시)
# ----------------------------------------------------------------------------
@st.cache_data
def cached_load_data():
    return load_data()


@st.cache_data
def cached_load_value_predictions():
    return load_value_predictions()


data = cached_load_data()
features, weights = get_feature_list_and_weights(data["weights"])
data["meta"]["DISTRICT_CODE"] = data["meta"]["DISTRICT_CODE"].astype(str)
data["vector"]["DISTRICT_CODE"] = data["vector"]["DISTRICT_CODE"].astype(str)
if "raw_stats" in data and isinstance(data["raw_stats"], pd.DataFrame):
    data["raw_stats"]["DISTRICT_CODE"] = data["raw_stats"]["DISTRICT_CODE"].astype(str)

value_df = cached_load_value_predictions()
model_info = load_model_info() if value_df is not None else {}
geojson = load_geojson()

# 가격 도메인 (range bar 정규화 기준)
if value_df is not None and len(value_df) > 0:
    PRICE_DOMAIN = (
        float(value_df["PRED_LOWER_95"].min()),
        float(value_df["PRED_UPPER_95"].max()),
    )
else:
    PRICE_DOMAIN = (0.0, 10000.0)


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
template_ids_all = list(HOUSEHOLD_TEMPLATES.keys())
priority_ids_all = list(PRIORITY_GROUPS.keys())

st.session_state.setdefault("template_id", template_ids_all[0])
st.session_state.setdefault("priorities", [])
st.session_state.setdefault("slider_adjust", {})
st.session_state.setdefault("budget_filter", None)
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("last_input", None)
st.session_state.setdefault("trigger_run", False)
st.session_state.setdefault("selected_cluster", None)


# ----------------------------------------------------------------------------
# 매칭 실행
# ----------------------------------------------------------------------------
def run_matching(n: int = 5):
    result = recommend(
        template_id=st.session_state.template_id,
        user_priorities=list(st.session_state.priorities),
        slider_adjustments=st.session_state.slider_adjust or None,
        budget_filter=st.session_state.budget_filter,
        dong_vector_df=data["vector"],
        dong_meta_df=data["meta"],
        feature_weights_v3=weights,
        all_features=features,
        n=n,
        enforce_diversity=True,
    )
    st.session_state.last_result = result
    st.session_state.last_input = {
        "template": st.session_state.template_id,
        "priorities": list(st.session_state.priorities),
    }


def apply_preset(template_id: str, priority_ids: list[str]):
    st.session_state.template_id = template_id
    st.session_state.priorities = list(priority_ids)
    st.session_state.slider_adjust = {}
    st.session_state.budget_filter = None
    st.session_state.trigger_run = True


# ============================================================================
# 사이드바: 입력 (S1 카드 그리드 / S2 우선순위 / 프리셋 / 매칭 시작 / S4 / S3)
# ============================================================================
with st.sidebar:
    st.markdown("## 🏙️ 동네잇다")
    st.caption("나에게 맞는 서울 동네를 데이터가 찾아드립니다")

    # ─ 데모 프리셋 ────────────────────────────────────────────────────────
    st.markdown('<div class="di-preset-h">▶ 데모 프리셋 (3클릭 시연)</div>',
                unsafe_allow_html=True)
    pc1, pc2 = st.columns(2)
    with pc1:
        if st.button("🍃 신혼부부 데모", use_container_width=True, key="preset_a"):
            apply_preset("newlywed_no_kids", ["commute_convenience", "asset_value"])
            st.rerun()
    with pc2:
        if st.button("📚 초등 자녀 데모", use_container_width=True, key="preset_b"):
            apply_preset("couple_kids_elementary", ["kids_education"])
            st.rerun()

    st.divider()

    # ─ 단계 1: 가구 형태 (카드 그리드 2×4) ────────────────────────────────
    st.markdown("**단계 1 · 가구 형태**  ", help="가장 비슷한 가구 형태 1개 선택")

    cols_t = st.columns(2)
    for i, tid in enumerate(template_ids_all):
        t = HOUSEHOLD_TEMPLATES[tid]
        with cols_t[i % 2]:
            is_sel = (st.session_state.template_id == tid)
            if st.button(
                f"{t['icon']} {t['label_kr']}",
                key=f"t_{tid}",
                use_container_width=True,
                type=("primary" if is_sel else "secondary"),
                help=f"{t['description']} · {t['age_band']}",
            ):
                st.session_state.template_id = tid
                st.rerun()

    sel_t = HOUSEHOLD_TEMPLATES[st.session_state.template_id]
    st.caption(f"✔ {sel_t['icon']} **{sel_t['label_kr']}** · {sel_t['age_band']}")

    st.divider()

    # ─ 단계 2: 라이프스타일 우선순위 ─────────────────────────────────────
    st.markdown(
        "**단계 2 · 라이프스타일 우선순위**  ",
        help="중요한 순서대로 1~3개 선택 (선택 순서가 곧 순위)",
    )
    st.caption("1순위×1.0 · 2순위×0.6 · 3순위×0.3")

    selected = st.multiselect(
        "우선순위",
        options=priority_ids_all,
        default=st.session_state.priorities,
        format_func=lambda pid: f"{PRIORITY_GROUPS[pid]['icon']} {PRIORITY_GROUPS[pid]['label_kr']}",
        max_selections=3,
        label_visibility="collapsed",
        key="prio_multi",
    )
    st.session_state.priorities = selected

    if selected:
        chips = []
        for i, pid in enumerate(selected, 1):
            chips.append(
                f'<span class="di-badge di-prio-pill">'
                f'<span class="di-prio-rank">{i}순위</span> '
                f'{PRIORITY_GROUPS[pid]["icon"]} {PRIORITY_GROUPS[pid]["label_kr"]}'
                f'</span>'
            )
        st.markdown(" ".join(chips), unsafe_allow_html=True)

    st.divider()

    # ─ 매칭 시작 버튼 ─────────────────────────────────────────────────────
    if st.button("🎯 매칭 시작", type="primary", use_container_width=True, key="run_btn"):
        run_matching()
        st.rerun()

    # ─ 단계 4: 고급 (선택) ────────────────────────────────────────────────
    with st.expander("⚙️ 단계 4 · 고급: 변수 직접 조정 (선택)", expanded=False):
        st.caption("일반 사용자는 건너뛰어도 됩니다.")
        adj = {}
        adj["AGE_UNDER20_PCT"] = st.slider("어린이 비율 선호", -2.0, 2.0, 0.0, 0.5)
        adj["VISITOR_RATIO"] = st.slider("유동인구·상권 활기", -2.0, 2.0, 0.0, 0.5)
        adj["AVG_ASSET_LN"] = st.slider("고자산층 거주", -2.0, 2.0, 0.0, 0.5)
        adj["RESIDENT_RATIO"] = st.slider("거주 안정성(조용함)", -2.0, 2.0, 0.0, 0.5)
        adj["CREDIT_CARD_INTENSITY"] = st.slider("외식·소비 활성도", -2.0, 2.0, 0.0, 0.5)
        st.session_state.slider_adjust = {k: v for k, v in adj.items() if v != 0.0}

    # ─ 단계 3: 예산 필터 (선택) ───────────────────────────────────────────
    if value_df is not None:
        with st.expander("💰 단계 3 · 예산 필터 (선택)", expanded=False):
            tm = model_info.get("test_metrics", {})
            st.caption(
                f"LightGBM (R²={tm.get('R2', 0.928):.3f}, "
                f"MAPE={tm.get('MAPE', 5.57)}%) 기반"
            )
            use_budget = st.checkbox("예산 필터 사용", value=False, key="use_budget")
            if use_budget:
                budget_buy = st.slider("매매 예산 (억원)", 3, 50, (10, 25), key="bf_buy")
                pyeong = st.number_input("평수", 10, 80, 30, key="bf_py")
                ci_strict = st.checkbox(
                    "보수적 (95% CI 하한 사용)", value=True, key="bf_ci",
                    help="체크 시 동의 95% CI 하한이 예산에 들어가야 통과",
                )
                st.session_state.budget_filter = {
                    "min_billion": float(budget_buy[0]),
                    "max_billion": float(budget_buy[1]),
                    "pyeong": int(pyeong),
                    "use_lower_ci": bool(ci_strict),
                }
            else:
                st.session_state.budget_filter = None

    if data.get("is_mock"):
        st.warning(
            "ℹ️ Mock 데이터 사용 중 — "
            f"누락: {', '.join(data.get('missing_files', []))}"
        )


# ============================================================================
# 프리셋 트리거 처리
# ============================================================================
if st.session_state.trigger_run:
    st.session_state.trigger_run = False
    run_matching()


# ============================================================================
# 메인 영역
# ============================================================================
result = st.session_state.last_result

# ── 헤더 ──
st.markdown("# 🏙️ 동네잇다")
st.markdown(
    f'<p style="color:{COLORS["MUTED"]}; font-size:16px; margin-top:-12px;">'
    f'라이프스타일도 맞고, 가격도 합리적인 동네를 데이터가 찾아드립니다 · '
    f'서초·영등포·중구 118개 법정동</p>',
    unsafe_allow_html=True,
)

# ── 결과 없을 때: 대형 지도 + DNA 미리보기 ──
if result is None:
    st.info(
        "👈 사이드바에서 **데모 프리셋** 한 개를 클릭하거나, "
        "**가구 형태 + 우선순위**를 직접 골라 **🎯 매칭 시작**을 눌러주세요."
    )

    if geojson is not None:
        st.markdown("### 🗺️ 서울 118개 법정동 — 동네 DNA 지도")
        st.caption(
            "각 동은 5가지 DNA 클러스터 중 하나에 속합니다. "
            "지도 위로 마우스를 올리면 동 이름·DNA 정보가 표시됩니다."
        )
        render_seoul_dna_map(data["meta"], geojson, title=None, height=560)
    else:
        st.warning("⏸ data/seoul_dong.geojson 누락 — 지도 표시 불가")

    st.markdown("### 📚 동네 DNA 5가지")
    st.caption("데이터가 발견한 서울 동네 유형. 매칭 결과는 이 중 하나의 DNA에 속합니다.")

    cluster_count = data["meta"]["CLUSTER_ID"].value_counts().sort_index()
    cols_dna = st.columns(5)
    for c_id, dna in DNA_LABELS.items():
        n = int(cluster_count.get(c_id, 0))
        with cols_dna[c_id]:
            color = CLUSTER_COLORS.get(c_id, COLORS["PURPLE"])
            st.markdown(
                f'<div class="di-card" style="border-top:4px solid {color};">'
                f'<div style="font-size:24px;">{dna["icon"]}</div>'
                f'<div style="font-weight:800; color:{color}; margin:4px 0;">'
                f'C{c_id} · {dna["name"]}</div>'
                f'<div style="font-size:12px; color:{COLORS["MUTED"]};">'
                f'{dna["summary"]}</div>'
                f'<div style="margin-top:10px; font-size:13px; font-weight:700; color:{COLORS["INK"]};">'
                f'{n}개 동</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.stop()


# ============================================================================
# 결과 표시 — 입력 요약 + 3개 뷰 (Top 5 / 상세 / DNA 탐색)
# ============================================================================

# ── 입력 요약 ──
with st.container(border=True):
    tmpl = HOUSEHOLD_TEMPLATES[st.session_state.last_input["template"]]
    prio_chips = []
    for i, pid in enumerate(st.session_state.last_input["priorities"], 1):
        g = PRIORITY_GROUPS[pid]
        prio_chips.append(
            f'<span class="di-badge di-prio-pill">'
            f'<span class="di-prio-rank">{i}순위</span> {g["icon"]} {g["label_kr"]}'
            f'</span>'
        )
    st.markdown(
        f'<div style="font-size:14px;">'
        f'<span style="color:{COLORS["MUTED"]};">입력 페르소나:</span> '
        f'<b style="font-size:16px; color:{COLORS["INK"]};">'
        f'{tmpl["icon"]} {tmpl["label_kr"]}</b> &nbsp;·&nbsp; '
        f'{" ".join(prio_chips) if prio_chips else "(우선순위 없음)"}'
        f'</div>',
        unsafe_allow_html=True,
    )

# Redundancy 경고
if result["redundancy_warning"]:
    g1, g2, cos = result["redundancy_warning"]
    l1 = PRIORITY_GROUPS.get(g1, {}).get("label_kr", g1)
    l2 = PRIORITY_GROUPS.get(g2, {}).get("label_kr", g2)
    st.warning(
        f"⚠️ 선택한 우선순위가 매우 비슷합니다 (cos={cos:.2f}): {l1} ↔ {l2}. "
        f"한 방향으로 쏠릴 수 있어요."
    )

# ── 사전 계산 (모든 뷰에서 공유) ──
w_cols = [f"{v}_W" for v in features]
dong_matrix_W = data["vector"][w_cols].to_numpy()
cluster_ids_arr = data["meta"]["CLUSTER_ID"].to_numpy()

# 동→메타 인덱스
code_to_idx = {
    str(c): i for i, c in enumerate(data["meta"]["DISTRICT_CODE"].astype(str))
}

# ── 3개 뷰 탭 ──
tab_top, tab_detail, tab_dna = st.tabs([
    "🏆 추천 Top",
    "🔍 상세 분석",
    "🧬 동네 DNA",
])


# ============================================================================
# 공통 헬퍼: 변수 한글 라벨 + 친근 설명 + 미니 기여도 차트
# ============================================================================
# 변수별 친근 설명 — 풀어쓴 "이 동/내 선호" 카피
# 각 entry: emoji, name, mean_plus(이 동이 평균 위일 때 의미),
#           mean_minus(평균 아래일 때 의미),
#           user_plus(페르소나가 양수일 때 사용자가 원하는 것),
#           user_minus(페르소나가 음수일 때 사용자가 원하는 것)
VAR_EXPLAIN: dict[str, dict[str, str]] = {
    "AGE_UNDER20_PCT": {
        "emoji": "👶", "name": "어린이/청소년 비율",
        "mean_plus":  "어린이·청소년이 많아 학군지 분위기",
        "mean_minus": "어린이가 적고 직장·1인 가구 위주",
        "user_plus":  "자녀 키우기 좋은 동네를 선호",
        "user_minus": "조용한 성인 중심 동네 선호",
    },
    "AGE_20S_PCT": {
        "emoji": "🎓", "name": "20대 비율",
        "mean_plus":  "20대 청년이 많은 활기찬 동네",
        "mean_minus": "20대가 적은 차분한 동네",
        "user_plus":  "젊은 분위기를 선호",
        "user_minus": "성숙한 인구층을 선호",
    },
    "AGE_30S_PCT": {
        "emoji": "💼", "name": "30대 비율",
        "mean_plus":  "30대 직장인이 자리 잡은 동네",
        "mean_minus": "30대가 적은 동네",
        "user_plus":  "30대 라이프스타일을 선호",
        "user_minus": "30대 외 인구층을 선호",
    },
    "AGE_40S_PCT": {
        "emoji": "👨‍💼", "name": "40대 비율",
        "mean_plus":  "40대 가족이 안착한 동네",
        "mean_minus": "40대가 적은 동네",
        "user_plus":  "40대 정착 가족 분위기를 선호",
        "user_minus": "더 젊은 인구층을 선호",
    },
    "AGE_50S_PCT": {
        "emoji": "🧑", "name": "50대 비율",
        "mean_plus":  "50대 비중이 높은 차분한 동네",
        "mean_minus": "50대가 적은 젊은 동네",
        "user_plus":  "안정적 시니어 분위기를 선호",
        "user_minus": "더 젊은 동네 선호",
    },
    "RESIDENT_RATIO": {
        "emoji": "🏠", "name": "거주 비중",
        "mean_plus":  "주거지로 안정적이고 조용한 동네",
        "mean_minus": "직장·유동인구 중심 (거주자 적음)",
        "user_plus":  "거주 안정성·조용함을 선호",
        "user_minus": "활기찬 도심 분위기를 선호",
    },
    "WORKER_RATIO": {
        "emoji": "💼", "name": "직장 비중",
        "mean_plus":  "오피스·일터가 많은 도심",
        "mean_minus": "직장이 적은 베드타운",
        "user_plus":  "직주근접을 선호",
        "user_minus": "직장과 거주 분리를 선호",
    },
    "VISITOR_RATIO": {
        "emoji": "🎉", "name": "유동인구·상권 활기",
        "mean_plus":  "유동인구가 많은 활기찬 상권",
        "mean_minus": "유동인구가 적은 정적인 동네",
        "user_plus":  "활기찬 상권 분위기를 선호",
        "user_minus": "조용한 동네를 선호",
    },
    "TOTAL_POP_LN": {
        "emoji": "👥", "name": "인구 규모",
        "mean_plus":  "인구가 많은 큰 동네",
        "mean_minus": "인구가 적은 작은 동네",
        "user_plus":  "큰 동네를 선호",
        "user_minus": "작은 동네를 선호",
    },
    "AVG_ASSET_LN": {
        "emoji": "💎", "name": "평균 자산",
        "mean_plus":  "자산가가 많이 거주하는 동네",
        "mean_minus": "자산 수준이 평이한 동네",
        "user_plus":  "고자산 거주 환경을 선호",
        "user_minus": "고자산 환경 비선호",
    },
    "AVG_INCOME_LN": {
        "emoji": "💵", "name": "평균 소득",
        "mean_plus":  "고소득자가 많이 거주하는 동네",
        "mean_minus": "평균 소득대 동네",
        "user_plus":  "고소득 거주 환경을 선호",
        "user_minus": "고소득 환경 비선호",
    },
    "HIGH_INCOME_PCT": {
        "emoji": "📈", "name": "고소득자 비율",
        "mean_plus":  "고소득자 비중이 두드러진 동네",
        "mean_minus": "고소득자가 적은 동네",
        "user_plus":  "상위 소득자 밀집을 선호",
        "user_minus": "고소득 환경 비선호",
    },
    "HIGH_CREDIT_PCT": {
        "emoji": "🏦", "name": "고신용자 비율",
        "mean_plus":  "신용도 높은 거주자 비중↑",
        "mean_minus": "신용도가 평이한 동네",
        "user_plus":  "안정 거주층 환경을 선호",
        "user_minus": "안정 거주층 비선호",
    },
    "EXECUTIVE_PCT": {
        "emoji": "🎩", "name": "임원/전문직 비율",
        "mean_plus":  "임원·전문직이 많은 프리미엄 동네",
        "mean_minus": "전문직 비중이 평이한 동네",
        "user_plus":  "임원·전문직 거주 환경을 선호",
        "user_minus": "전문직 환경 비선호",
    },
    "MEME_MOM_AVG_CBRT": {
        "emoji": "👩‍👧", "name": "30~40대 여성 풀",
        "mean_plus":  "30~40대 여성 인구 풀이 두꺼움",
        "mean_minus": "30~40대 여성 풀이 평이함",
        "user_plus":  "또래 여성 네트워크를 선호",
        "user_minus": "또래 여성 네트워크 비선호",
    },
    "SEG_ADULT_CHILD_PCT": {
        "emoji": "👨‍👩‍👧", "name": "자녀 동반 가구",
        "mean_plus":  "자녀 키우는 가족이 많은 동네",
        "mean_minus": "자녀 동반 가구가 적은 동네",
        "user_plus":  "가족 친화 동네를 선호",
        "user_minus": "비가족 동네를 선호",
    },
    "CREDIT_CARD_INTENSITY": {
        "emoji": "💳", "name": "카드 소비 강도",
        "mean_plus":  "외식·쇼핑 소비가 활발한 상권",
        "mean_minus": "소비가 차분한 조용한 동네",
        "user_plus":  "외식·소비 활성도를 선호",
        "user_minus": "조용한 소비 동네를 선호",
    },
    "MORTGAGE_LN": {
        "emoji": "🏘️", "name": "주택담보대출 활성도",
        "mean_plus":  "주택 매매·담보대출이 활발한 동네",
        "mean_minus": "주택 거래가 평이한 동네",
        "user_plus":  "활발한 부동산 시장을 선호",
        "user_minus": "조용한 주택 시장을 선호",
    },
    "HAS_RICHGO_APT": {
        "emoji": "📊", "name": "아파트 시세 데이터",
        "mean_plus":  "시세가 투명한 아파트 단지가 풍부",
        "mean_minus": "시세 데이터가 제한적",
        "user_plus":  "시세 투명성을 선호",
        "user_minus": "—",
    },
    "PREMIUM_40PY_RATIO": {
        "emoji": "🏢", "name": "40평+ 프리미엄 비율",
        "mean_plus":  "40평 이상 대형 아파트 비중↑",
        "mean_minus": "중소형 평형 위주",
        "user_plus":  "대형 평형을 선호",
        "user_minus": "중소형 평형을 선호",
    },
}


def _mag_desc(v: float) -> str:
    """|z×√w| 절댓값 → 정성 강도."""
    a = abs(v)
    if a >= 1.5: return "강하게"
    if a >= 0.8: return "뚜렷하게"
    if a >= 0.3: return "약간"
    return "거의 평균 수준으로"


def get_top_friendly_contributions(
    persona_vec: np.ndarray,
    dong_vec: np.ndarray,
    all_features: list[str],
    top_k: int = 3,
) -> list[dict]:
    """상위 K개 기여 변수에 친근 설명을 붙여 반환."""
    p_norm = np.linalg.norm(persona_vec)
    d_norm = np.linalg.norm(dong_vec)
    if p_norm < 1e-10 or d_norm < 1e-10:
        return []
    contributions = (persona_vec * dong_vec) / (p_norm * d_norm)
    order = np.argsort(-contributions)  # 양수 큰 순서
    out = []
    for i in order:
        if len(out) >= top_k:
            break
        c = float(contributions[i])
        if c < 1e-4:
            # 양수 기여가 끝나면 음수 기여(=차이점) 채움
            break
        var = all_features[i]
        meta = VAR_EXPLAIN.get(var, {
            "emoji": "•",
            "name": KOR_LABELS.get(var, var),
            "mean_plus": "평균보다 높음", "mean_minus": "평균보다 낮음",
            "user_plus": "이 변수를 선호", "user_minus": "이 변수를 비선호",
        })
        pw = float(persona_vec[i])
        dw = float(dong_vec[i])
        user_text = meta["user_plus"] if pw >= 0 else meta["user_minus"]
        dong_text = meta["mean_plus"] if dw >= 0 else meta["mean_minus"]
        sign_match = (pw * dw) > 0
        if sign_match and c >= 0.04:
            match_label = "강한 일치"
            match_color = COLORS["GREEN"]
        elif sign_match and c >= 0.015:
            match_label = "일치"
            match_color = COLORS["TEAL"]
        elif sign_match:
            match_label = "약한 일치"
            match_color = COLORS["MUTED"]
        else:
            match_label = "차이"
            match_color = COLORS["CORAL"]

        out.append({
            "var": var,
            "emoji": meta["emoji"],
            "name": meta["name"],
            "user_text": user_text,
            "dong_text": dong_text,
            "user_mag": _mag_desc(pw),
            "dong_mag": _mag_desc(dw),
            "user_dir": "높음" if pw >= 0 else "낮음",
            "dong_dir": "위" if dw >= 0 else "아래",
            "match_label": match_label,
            "match_color": match_color,
            "contribution": c,
        })
    return out


KOR_LABELS = {
    "AGE_UNDER20_PCT":     "어린이/청소년 비율",
    "AGE_20S_PCT":         "20대 비율",
    "AGE_30S_PCT":         "30대 비율",
    "AGE_40S_PCT":         "40대 비율",
    "AGE_50S_PCT":         "50대 비율",
    "AGE_60S_PCT":         "60대 비율",
    "RESIDENT_RATIO":      "거주 비중",
    "WORKER_RATIO":        "직장 비중",
    "VISITOR_RATIO":       "유동인구 비중",
    "TOTAL_POP_LN":        "인구 규모",
    "AVG_ASSET_LN":        "평균 자산",
    "AVG_INCOME_LN":       "평균 소득",
    "HIGH_INCOME_PCT":     "고소득자 비율",
    "HIGH_CREDIT_PCT":     "고신용자 비율",
    "EXECUTIVE_PCT":       "임원/전문직 비율",
    "MEME_MOM_AVG_CBRT":   "30~40대 여성 풀",
    "SEG_ADULT_CHILD_PCT": "자녀 동반 가구 비율",
    "CREDIT_CARD_INTENSITY": "카드 소비 강도",
    "MORTGAGE_LN":         "주택담보대출 활성도",
    "HAS_RICHGO_APT":      "아파트 시세 데이터 존재",
    "PREMIUM_40PY_RATIO":  "40평 이상 프리미엄 비율",
}


def render_mini_contribution(persona_vec: np.ndarray, dong_vec: np.ndarray,
                             all_features: list[str], top_k: int = 5) -> None:
    """카드 내부용 미니 기여도 차트 (height=180)."""
    p_norm = np.linalg.norm(persona_vec)
    d_norm = np.linalg.norm(dong_vec)
    if p_norm < 1e-10 or d_norm < 1e-10:
        st.caption("기여도 계산 불가")
        return
    contributions = (persona_vec * dong_vec) / (p_norm * d_norm)
    order = np.argsort(-np.abs(contributions))[:top_k]
    labels, values, colors = [], [], []
    for i in order:
        if abs(contributions[i]) < 1e-4:
            continue
        var = all_features[i]
        labels.append(KOR_LABELS.get(var, var))
        values.append(float(contributions[i]))
        colors.append(COLORS["TEAL"] if contributions[i] > 0 else COLORS["CORAL"])
    if not values:
        st.caption("기여도 데이터 없음")
        return
    fig = go.Figure(go.Bar(
        x=values[::-1], y=labels[::-1], orientation="h",
        marker=dict(color=colors[::-1], line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>기여도 %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=0, r=10, t=8, b=24),
        height=180,
        xaxis_title="기여도 (양수=일치, 음수=차이)",
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
        font=dict(size=11, family="-apple-system, BlinkMacSystemFont, Pretendard, sans-serif"),
    )
    fig.update_xaxes(zeroline=True, zerolinecolor=COLORS["MUTED"], zerolinewidth=1)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ============================================================================
# 뷰 1: Top N 카드 v2 — 반응형 단일 HTML 카드 + 상세 expander
# ============================================================================
def render_top_card(c: dict) -> None:
    """매칭 점수 · 예상가 · 평가/신뢰도 + 왜 이 동인가 + 상세 확장."""
    code = c["district_code"]
    cluster_id = c["cluster_id"]
    cluster_color = CLUSTER_COLORS.get(cluster_id, COLORS["PURPLE"])
    dna = DNA_LABELS.get(cluster_id, {"icon": "❓", "name": "미분류"})
    value_info = (
        get_value_for_dong(code, value_df) if value_df is not None else None
    )

    # ── 헤더 ──
    head_html = (
        f'<div class="di-rec-head">'
        f'  <div class="di-rec-title-wrap">'
        f'    <span class="di-rec-rank">#{c["rank"]}</span>'
        f'    <span class="di-rec-name">{c["district_kor_name"]}</span>'
        f'    <span class="di-rec-sgg">· {c["sgg"]}</span>'
        f'  </div>'
        f'  <div>{dna_badge(cluster_id, dna["icon"], dna["name"])}</div>'
        f'</div>'
    )

    # ── 매칭 점수 ──
    match_html = (
        f'<div>'
        f'  <div class="di-metric-label" title="{GLOSSARY["match_score"]["tooltip"]}">'
        f'    {GLOSSARY["match_score"]["display"]} ⓘ</div>'
        f'  <div class="di-metric-value">{c["match_pct"]:.1f}'
        f'    <span class="di-metric-unit">%</span></div>'
        f'  {progress_bar(c["match_pct"])}'
        f'</div>'
    )

    # ── 가격 ──
    if value_info is not None:
        point = value_info["point"]
        lower = value_info["lower_95"]
        upper = value_info["upper_95"]
        cohort = value_info.get("cohort", "")
        price_html = (
            f'<div>'
            f'  <div class="di-metric-label" title="{GLOSSARY["price_range"]["tooltip"]}">'
            f'    예상 시세 ⓘ '
            f'    <span style="font-weight:600; color:#9CA3AF; letter-spacing:0; '
            f'    font-size:10px; margin-left:6px;">· {cohort}</span></div>'
            f'  <div class="di-metric-value">{point:,.0f}'
            f'    <span class="di-metric-unit">만원/평</span></div>'
            f'  <div class="di-metric-sub">30평 ≈ '
            f'    <b style="color:{COLORS["INK"]};">{value_info["price_30py_billion"]:.1f}억</b>'
            f'    · 95% 범위 {value_info["lower_30py_billion"]:.1f}~'
            f'    {value_info["upper_30py_billion"]:.1f}억</div>'
            f'  {range_bar(point, lower, upper, PRICE_DOMAIN[0], PRICE_DOMAIN[1])}'
            f'</div>'
        )
    else:
        price_html = (
            f'<div class="di-metric-label" style="color:{COLORS["MUTED"]};">'
            f'⏸ 가격 데이터 없음</div>'
        )

    # ── 배지 ──
    if value_info is not None:
        valuation_q = value_df.loc[value_df["DISTRICT_CODE"] == code, "VALUATION"]
        valuation = valuation_q.iloc[0] if len(valuation_q) > 0 else "FAIR"
        is_imputed = (value_info.get("cohort") == "IMPUTED_92")
        badges_html = (
            f'<div class="di-rec-col-badges">'
            f'{conf_badge(value_info["confidence"])}'
            f'{val_badge(valuation, is_imputed=is_imputed)}'
            f'</div>'
        )
    else:
        badges_html = '<div class="di-rec-col-badges"></div>'

    body_html = f'<div class="di-rec-body">{match_html}{price_html}{badges_html}</div>'

    # ── 왜 이 동인가 (친근 설명, 인라인 항상 보임) ──
    idx_for_why = code_to_idx.get(code)
    why_items_html = ""
    if idx_for_why is not None:
        persona_vec = result["persona_vec"]
        dong_vec = dong_matrix_W[idx_for_why]
        friendly = get_top_friendly_contributions(
            persona_vec, dong_vec, features, top_k=3,
        )
        for f in friendly:
            user_seg = (
                f'당신은 <b>{f["user_text"]}</b>'
                f'<span class="di-tag {("di-tag-down" if f["user_dir"]=="낮음" else "")}">'
                f'{f["user_mag"]} 원함</span>'
            )
            dong_seg = (
                f'이 동은 <b>{f["dong_text"]}</b>'
                f'<span class="di-tag {("di-tag-down" if f["dong_dir"]=="아래" else "")}">'
                f'평균 {f["dong_dir"]} · {f["dong_mag"]}</span>'
            )
            why_items_html += (
                f'<div class="di-why-item">'
                f'  <div class="di-why-row">'
                f'    <span class="di-why-emoji">{f["emoji"]}</span>'
                f'    <span class="di-why-name">{f["name"]}</span>'
                f'    <span class="di-why-match" style="background:{f["match_color"]};">'
                f'      {f["match_label"]}</span>'
                f'  </div>'
                f'  <div class="di-why-detail">{user_seg} · {dong_seg}</div>'
                f'</div>'
            )

    if why_items_html:
        why_html = (
            f'<div class="di-why">'
            f'<div class="di-why-h">💡 왜 이 동인가 — 가장 잘 맞은 항목 3가지</div>'
            f'{why_items_html}'
            f'</div>'
        )
    else:
        why_html = ""

    # ── LOW/EXTRAP 경고 ──
    warn_html = ""
    if value_info is not None and value_info["confidence"] in ("LOW", "EXTRAPOLATION"):
        meta = CONF_COLORS[value_info["confidence"]]
        warn_html = (
            f'<div class="di-warn" style="background:{meta["bg"]}15; '
            f'border-left:4px solid {meta["bg"]}; color:{COLORS["INK"]};">'
            f'{meta["icon"]} <b>{meta["label"]}</b> · {value_info.get("usage_guide", "")}'
            f'</div>'
        )

    # ── 통합 카드 ──
    card_html = (
        f'<div class="di-rec-card" style="border-left-color:{cluster_color};">'
        f'{head_html}{body_html}{why_html}{warn_html}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    # ── 상세 expander ──
    detail_label = (
        f"📖 #{c['rank']} {c['district_kor_name']} 자세히 보기 "
        f"— 변수 기여도 · 비슷한 동"
    )
    with st.expander(detail_label, expanded=False):
        idx = code_to_idx.get(code)
        if idx is None:
            st.caption("상세 데이터를 찾을 수 없습니다.")
            return

        persona_vec = result["persona_vec"]
        dong_vec = dong_matrix_W[idx]

        col_d1, col_d2 = st.columns([3, 2])
        with col_d1:
            st.markdown("**📐 매칭 기여 Top 5 변수**")
            render_mini_contribution(persona_vec, dong_vec, features, top_k=5)

        with col_d2:
            st.markdown("**🧬 같은 DNA · 비슷한 동 3곳**")
            sim_idx, sim_sims = similar_dong(
                idx, dong_matrix_W, cluster_ids_arr,
                same_cluster_only=True, n=3,
            )
            if sim_idx:
                for i, s in zip(sim_idx, sim_sims):
                    mr = data["meta"].iloc[i]
                    st.markdown(
                        f"- **{mr['DISTRICT_KOR_NAME']}** "
                        f"<span style='color:{COLORS['MUTED']}; font-size:12px;'>"
                        f"({mr['SGG']})</span> · 유사도 "
                        f"<b style='color:{COLORS['TEAL']};'>{s*100:.1f}%</b>",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("같은 DNA 클러스터에 다른 동이 없습니다.")

            if value_info is not None and value_info.get("cohort") == "IMPUTED_92":
                st.markdown("")
                st.caption(
                    "⚠ IMPUTED 동 — 평가 배지는 시군구 평균 대비 판단입니다 "
                    "(실측 시장가 대비 아님)."
                )


with tab_top:
    # 지도 — Top N 핀 표시
    if geojson is not None:
        pins = []
        for c in result["candidates"]:
            vi = (
                get_value_for_dong(c["district_code"], value_df)
                if value_df is not None else None
            )
            pins.append({**c, "value_info": vi})

        st.markdown("#### 🗺️ 추천 동 지도")
        st.caption(
            f"DNA 컬러로 채색된 서울 118개 동 위에 추천 {len(pins)}개를 "
            f"순위 핀으로 표시했습니다. 핀에 마우스를 올리면 매칭 점수·가격이 보입니다."
        )
        render_seoul_dna_map(
            data["meta"], geojson, pins=pins, height=480,
        )
        st.markdown("")

    st.markdown(f"### 🏆 Top {len(result['candidates'])} 추천 동네")

    for c in result["candidates"]:
        render_top_card(c)

    # 결과 통계 요약
    with st.container(border=True):
        st.markdown("**📊 결과 분포**")
        sgg_dist: dict[str, int] = {}
        cluster_dist: dict[int, int] = {}
        for c in result["candidates"]:
            sgg_dist[c["sgg"]] = sgg_dist.get(c["sgg"], 0) + 1
            cluster_dist[c["cluster_id"]] = cluster_dist.get(c["cluster_id"], 0) + 1

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**시군구 분포**")
            for sgg, n in sorted(sgg_dist.items(), key=lambda x: -x[1]):
                st.markdown(f"- {sgg}: {n}개")
        with col2:
            st.markdown("**DNA 분포**")
            for cid, n in sorted(cluster_dist.items()):
                dna = DNA_LABELS[cid]
                color = CLUSTER_COLORS[cid]
                st.markdown(
                    f'<div><span style="color:{color}; font-weight:700;">'
                    f'{dna["icon"]} C{cid} {dna["name"]}</span> · {n}개</div>',
                    unsafe_allow_html=True,
                )


# ============================================================================
# 뷰 2: 상세 분석 (기여도 차트 + 동 프로파일)
# ============================================================================
def render_contribution_chart(persona_vec: np.ndarray, dong_vec: np.ndarray,
                              all_features: list[str], dong_name: str) -> None:
    """기여도 막대 차트 (Plotly horizontal bar)."""
    p_norm = np.linalg.norm(persona_vec)
    d_norm = np.linalg.norm(dong_vec)
    if p_norm < 1e-10 or d_norm < 1e-10:
        st.caption("기여도 계산 불가")
        return

    contributions = (persona_vec * dong_vec) / (p_norm * d_norm)
    order = np.argsort(-np.abs(contributions))[:8]
    labels, values, colors = [], [], []
    for i in order:
        if abs(contributions[i]) < 1e-4:
            continue
        var = all_features[i]
        kor = KOR_LABELS.get(var, var)
        labels.append(f"{kor} <span style='color:#9CA3AF;font-size:10px'>({var})</span>")
        values.append(float(contributions[i]))
        colors.append(COLORS["TEAL"] if contributions[i] > 0 else COLORS["CORAL"])

    fig = go.Figure(go.Bar(
        x=values[::-1],
        y=labels[::-1],
        orientation="h",
        marker=dict(color=colors[::-1]),
        hovertemplate="<b>%{y}</b><br>기여도 %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{dong_name} — 매칭 기여 Top 변수", font=dict(size=15)),
        margin=dict(l=0, r=10, t=40, b=10),
        height=320,
        xaxis_title="기여도 (양수=일치, 음수=차이)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="-apple-system, BlinkMacSystemFont, Pretendard, sans-serif"),
    )
    fig.update_xaxes(zeroline=True, zerolinecolor=COLORS["MUTED"], zerolinewidth=1)
    st.plotly_chart(fig, use_container_width=True)


def render_dong_profile(dong_idx: int, dong_name: str) -> None:
    """동 프로파일 (핵심 z-score 막대 + 같은 클러스터 평균 비교)."""
    profile_vars = [
        "RESIDENT_RATIO", "AGE_UNDER20_PCT", "AGE_30S_PCT", "AVG_INCOME_LN",
        "AVG_ASSET_LN", "VISITOR_RATIO", "WORKER_RATIO",
    ]
    rows = []
    cluster_id = int(cluster_ids_arr[dong_idx])
    cluster_mask = (cluster_ids_arr == cluster_id)

    for var in profile_vars:
        wcol = f"{var}_W"
        if wcol not in data["vector"].columns:
            continue
        dong_val = float(data["vector"][wcol].iloc[dong_idx])
        cluster_mean = float(data["vector"][wcol].iloc[cluster_mask].mean())
        rows.append({
            "var": KOR_LABELS.get(var, var),
            "dong": dong_val,
            "cluster_mean": cluster_mean,
        })

    if not rows:
        st.caption("프로파일 데이터 없음")
        return

    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["var"], x=df["dong"], orientation="h",
        name="이 동네", marker=dict(color=COLORS["TEAL"]),
        hovertemplate="<b>%{y}</b><br>이 동: %{x:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        y=df["var"], x=df["cluster_mean"], mode="markers",
        name=f"C{cluster_id} 평균",
        marker=dict(color=COLORS["AMBER"], size=10, symbol="diamond"),
        hovertemplate="<b>%{y}</b><br>클러스터 평균: %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{dong_name} 프로파일 (z×√w 단위)", font=dict(size=15)),
        margin=dict(l=0, r=10, t=40, b=10),
        height=300,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.2),
        font=dict(family="-apple-system, BlinkMacSystemFont, Pretendard, sans-serif"),
    )
    fig.update_xaxes(zeroline=True, zerolinecolor=COLORS["MUTED"])
    st.plotly_chart(fig, use_container_width=True)


def render_price_card(value_info: dict) -> None:
    """예측 추이 자리 — 시계열 데이터 없으므로 점추정+CI 카드로 대체."""
    if value_info is None:
        st.caption("⏸ 가격 데이터 없음")
        return

    cohort = value_info.get("cohort", "")
    is_imputed = (cohort == "IMPUTED_92")
    label_cohort = (
        "🟢 LEARN_26 (실측 학습 동)" if cohort == "LEARN_26"
        else "🟡 IMPUTED_92 (모델 추정 동)"
    )

    fig = go.Figure()
    p, l, u = value_info["point"], value_info["lower_95"], value_info["upper_95"]

    # CI 박스
    fig.add_shape(
        type="rect", x0=l, x1=u, y0=0.3, y1=0.7,
        fillcolor=COLORS["AMBER"], opacity=0.25, line=dict(width=0),
    )
    # 점추정 라인
    fig.add_shape(
        type="line", x0=p, x1=p, y0=0.2, y1=0.8,
        line=dict(color=COLORS["INK"], width=4),
    )
    # 마커
    fig.add_trace(go.Scatter(
        x=[l, p, u], y=[0.5, 0.5, 0.5], mode="markers+text",
        marker=dict(color=[COLORS["AMBER"], COLORS["INK"], COLORS["AMBER"]], size=[10, 14, 10]),
        text=[f"하한 {l:,.0f}", f"점추정 {p:,.0f}", f"상한 {u:,.0f}"],
        textposition=["bottom center", "top center", "bottom center"],
        showlegend=False, hoverinfo="skip",
    ))

    pad = (u - l) * 0.5 if u > l else max(p * 0.2, 500)
    fig.update_xaxes(range=[max(0, l - pad), u + pad], title="만원/평")
    fig.update_yaxes(range=[0, 1], showticklabels=False, showgrid=False)
    fig.update_layout(
        title=dict(text=f"예측 추이 — {label_cohort}", font=dict(size=15)),
        margin=dict(l=0, r=10, t=40, b=10),
        height=220,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="-apple-system, BlinkMacSystemFont, Pretendard, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)

    if is_imputed:
        st.info(
            "ℹ️ **IMPUTED 동** — 실측 분기 시계열이 없어 모델 추정값만 표시됩니다. "
            "평가는 시군구 평균 대비 판단입니다."
        )


with tab_detail:
    candidate_names = [
        f"#{c['rank']} {c['district_kor_name']} ({c['sgg']})"
        for c in result["candidates"]
    ]
    sel_idx = st.selectbox(
        "분석할 동네 선택",
        options=list(range(len(result["candidates"]))),
        format_func=lambda i: candidate_names[i],
        key="detail_sel",
    )
    target = result["candidates"][sel_idx]
    code = target["district_code"]
    dong_meta_idx = code_to_idx.get(code)

    # 기여도 차트
    if dong_meta_idx is not None:
        persona_vec = result["persona_vec"]
        dong_vec = dong_matrix_W[dong_meta_idx]
        st.markdown("#### 💡 왜 이 동인가 — 기여 변수 Top")
        render_contribution_chart(
            persona_vec, dong_vec, features, target["district_kor_name"],
        )

        # 동 프로파일 + 가격
        col_p, col_v = st.columns([1, 1])
        with col_p:
            st.markdown("#### 📐 동 프로파일")
            render_dong_profile(dong_meta_idx, target["district_kor_name"])
        with col_v:
            st.markdown("#### 💰 예측 추이")
            value_info = (
                get_value_for_dong(code, value_df) if value_df is not None else None
            )
            render_price_card(value_info)


# ============================================================================
# 뷰 3: DNA 탐색
# ============================================================================
with tab_dna:
    st.markdown("### 🧬 5가지 동네 DNA")
    st.caption("클러스터 카드를 클릭하면 해당 DNA의 동 목록과 유사 동 추천이 표시됩니다.")

    cluster_count = data["meta"]["CLUSTER_ID"].value_counts().sort_index()
    cols_dna = st.columns(5)
    for c_id, dna in DNA_LABELS.items():
        n = int(cluster_count.get(c_id, 0))
        color = CLUSTER_COLORS.get(c_id, COLORS["PURPLE"])
        is_sel = (st.session_state.selected_cluster == c_id)
        with cols_dna[c_id]:
            if st.button(
                f"{dna['icon']} C{c_id}\n{dna['name']}\n· {n}개 동",
                key=f"cluster_btn_{c_id}",
                use_container_width=True,
                type=("primary" if is_sel else "secondary"),
            ):
                st.session_state.selected_cluster = c_id
                st.rerun()
            st.caption(dna["summary"])

    sel_c = st.session_state.selected_cluster
    if sel_c is None:
        st.info("👆 위의 DNA 카드 하나를 클릭해보세요.")
    else:
        dna = DNA_LABELS[sel_c]
        color = CLUSTER_COLORS[sel_c]
        st.markdown(
            f'<div class="di-card" style="border-left:6px solid {color};">'
            f'<div style="font-size:20px; font-weight:800; color:{color};">'
            f'{dna["icon"]} C{sel_c} · {dna["name"]}</div>'
            f'<div style="color:{COLORS["MUTED"]}; margin-top:4px;">{dna["summary"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # 소속 동 목록
        c_dongs = data["meta"][data["meta"]["CLUSTER_ID"] == sel_c]
        st.markdown(f"#### 📍 소속 동 ({len(c_dongs)}개)")
        for sgg in sorted(c_dongs["SGG"].unique()):
            sgg_dongs = c_dongs[c_dongs["SGG"] == sgg]
            names = sgg_dongs["DISTRICT_KOR_NAME"].tolist()
            st.markdown(f"- **{sgg}** ({len(names)}): {', '.join(names)}")

        # 추천 결과 중 같은 DNA가 있으면 유사 동 표시
        st.markdown("#### 🔗 추천 결과 중 이 DNA의 동")
        same_dna_in_result = [
            c for c in result["candidates"] if c["cluster_id"] == sel_c
        ]
        if same_dna_in_result:
            for c in same_dna_in_result:
                st.markdown(
                    f"- **#{c['rank']} {c['district_kor_name']}** ({c['sgg']}) · "
                    f"매칭 {c['match_pct']:.1f}%"
                )

                # 같은 클러스터 내 유사 동
                idx = code_to_idx.get(c["district_code"])
                if idx is not None:
                    sim_idx, sim_sims = similar_dong(
                        idx, dong_matrix_W, cluster_ids_arr,
                        same_cluster_only=True, n=3,
                    )
                    if sim_idx:
                        items = []
                        for i, s in zip(sim_idx, sim_sims):
                            mr = data["meta"].iloc[i]
                            items.append(
                                f"{mr['DISTRICT_KOR_NAME']}({mr['SGG']}) "
                                f"{s*100:.1f}%"
                            )
                        st.caption(f"  · 같은 DNA 유사: {' / '.join(items)}")
        else:
            st.caption("이번 추천 결과에는 이 DNA의 동이 없습니다.")


# ============================================================================
# 푸터
# ============================================================================
st.divider()
tm = model_info.get("test_metrics", {}) if model_info else {}
st.caption(
    f"동네잇다 v2.0 · 데이터: SPH(Grandata) × Richgo · "
    f"118개 법정동 (서초·영등포·중구) · "
    f"Engine 1: cosine 55D · Engine 2: LightGBM "
    f"(R²={tm.get('R2', 0.928):.3f}, MAPE={tm.get('MAPE', 5.57)}%)"
)
