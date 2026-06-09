"""
============================================================================
streamlit_app.py — 동네잇다 (Dongnae:itda) Life Matcher v1.1
============================================================================
v1.1 변경사항 (Day 8 hotfix):
  - 탭 제거 → 단일 페이지 흐름 (Streamlit 탭 자동전환 미지원 이슈 해결)
  - DISTRICT_CODE 타입 일치 (int vs str 비교 실패 → IndexError 해결)
  - 세션 상태 안정화 (버튼 클릭 후 결과 유지)

handoff §"User Input 4단계 시스템" 구현.
실행:
    streamlit run streamlit_app.py
============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np

from household_templates import HOUSEHOLD_TEMPLATES, list_templates
from priority_groups import PRIORITY_GROUPS, list_groups
from data_loader import load_data, get_feature_list_and_weights
from matching_engine import recommend, similar_dong, DNA_LABELS
from value_estimator import (
    load_value_predictions,
    get_value_for_dong,
    get_confidence_metadata,
    load_model_info,
    get_value_summary,
)

# ============================================================================
# 페이지 설정
# ============================================================================
st.set_page_config(
    page_title="동네잇다 — 나에게 맞는 동네 찾기",
    page_icon="🏘️",
    layout="wide",
)


# ============================================================================
# 데이터 캐싱
# ============================================================================
@st.cache_data
def cached_load_data():
    return load_data()


# ============================================================================
# 데이터 로드
# ============================================================================
data = cached_load_data()
features, weights = get_feature_list_and_weights(data["weights"])

# v1.1 fix — DISTRICT_CODE 타입 강제 통일 (int64 → str)
# matching_engine이 str로 반환하므로 meta도 str로 맞춤
data["meta"]["DISTRICT_CODE"] = data["meta"]["DISTRICT_CODE"].astype(str)
data["vector"]["DISTRICT_CODE"] = data["vector"]["DISTRICT_CODE"].astype(str)


# ============================================================================
# 헤더
# ============================================================================
st.title("🏘️ 동네잇다")
st.caption("당신과 잘 맞는 서울 동네를 찾아드립니다 (서초·영등포·중구 118개 법정동)")

if data["is_mock"]:
    with st.expander("ℹ️ 현재 mock 데이터 사용 중", expanded=False):
        st.warning(
            f"**Day 8 GitHub 익스포트 전이라 mock 데이터로 작동 중입니다.** "
            f"누락 파일: {', '.join(data.get('missing_files', []))} — "
            f"클러스터 분포(33/11/25/17/32)와 시그널은 Day 6 결과 기반으로 재현됨."
        )


# ============================================================================
# 사이드바: 콜드 스타트 (handoff §"K-means 활용처 B")
# ============================================================================
with st.sidebar:
    st.header("🧬 5가지 동네 DNA")
    st.caption("어떤 동네 분위기를 원하는지 먼저 둘러보세요")

    cluster_count = data["meta"]["CLUSTER_ID"].value_counts().sort_index()
    for c_id, dna in DNA_LABELS.items():
        n = cluster_count.get(c_id, 0)
        with st.container(border=True):
            st.markdown(f"**{dna['icon']} {dna['name']}** · {n}개 동")
            st.caption(dna["summary"])

            sample = data["meta"][data["meta"]["CLUSTER_ID"] == c_id]
            if len(sample) > 0:
                names = sample["DISTRICT_KOR_NAME"].head(3).tolist()
                st.caption(f"예: {', '.join(names)}")


# ============================================================================
# 단계 1: 가구 형태
# ============================================================================
st.subheader("단계 1 · 가구 형태")
st.caption("가장 비슷한 가구 형태를 골라주세요 (필수, 30초)")

template_choices = list_templates()
template_ids = list(template_choices.keys())
template_labels = [
    f"{HOUSEHOLD_TEMPLATES[tid]['icon']} {HOUSEHOLD_TEMPLATES[tid]['label_kr']}"
    for tid in template_ids
]

selected_template_label = st.radio(
    "가구 형태",
    options=template_labels,
    horizontal=False,
    label_visibility="collapsed",
)
selected_template_id = template_ids[template_labels.index(selected_template_label)]

template_meta = HOUSEHOLD_TEMPLATES[selected_template_id]
st.caption(f"💡 {template_meta['description']} · 대표 연령: {template_meta['age_band']}")

st.divider()


# ============================================================================
# 단계 2: 라이프스타일 우선순위
# ============================================================================
st.subheader("단계 2 · 라이프스타일 우선순위")
st.caption("중요한 순서대로 1~3개를 골라주세요 (1순위 1.5배, 2순위 1.0배, 3순위 0.5배)")

priority_choices = list_groups()
priority_ids = list(priority_choices.keys())
priority_labels = {
    gid: f"{PRIORITY_GROUPS[gid]['icon']} {PRIORITY_GROUPS[gid]['label_kr']}"
    for gid in priority_ids
}

col1, col2, col3 = st.columns(3)

with col1:
    p1 = st.selectbox(
        "1순위 (필수)",
        options=priority_ids,
        format_func=lambda gid: priority_labels[gid],
        key="p1",
    )

p2_options = [None] + [g for g in priority_ids if g != p1]
with col2:
    p2 = st.selectbox(
        "2순위 (선택)",
        options=p2_options,
        format_func=lambda gid: "(선택 안 함)" if gid is None else priority_labels[gid],
        key="p2",
    )

p3_options = [None] + [g for g in priority_ids if g != p1 and g != p2]
with col3:
    p3 = st.selectbox(
        "3순위 (선택)",
        options=p3_options,
        format_func=lambda gid: "(선택 안 함)" if gid is None else priority_labels[gid],
        key="p3",
    )

user_priorities = [p for p in [p1, p2, p3] if p is not None]

if user_priorities:
    descs = [
        f"**{i+1}순위** · {priority_labels[g]}: {PRIORITY_GROUPS[g]['description']}"
        for i, g in enumerate(user_priorities)
    ]
    st.info(" / ".join(descs))

st.divider()


# ============================================================================
# 단계 3: 예산 필터 (Phase 7 Value Estimator 활성화 v2.0)
# ============================================================================

# 적정가 데이터 로드 (캐시)
@st.cache_data
def cached_load_value_predictions():
    return load_value_predictions()

value_df = cached_load_value_predictions()

st.subheader("단계 3 · 예산 필터 (선택)")

if value_df is None:
    st.warning("⏸ data/dong_value_predictions.csv 누락 — 예산 필터 비활성화")
    budget_filter = None
else:
    model_info = load_model_info()
    test_metrics = model_info.get("test_metrics", {})
    
    with st.expander("💡 예산으로 동네 필터링 (선택)", expanded=False):
        st.caption(
            f"LightGBM 모델 (R²={test_metrics.get('R2', 0.928):.3f}, "
            f"MAPE={test_metrics.get('MAPE', 5.57)}%) 기반 "
            f"118개 동 적정가 적용"
        )
        
        use_budget = st.checkbox("예산 필터 사용", value=False, key="use_budget_v2")
        
        col_a, col_b, col_c = st.columns([2, 2, 1])
        with col_a:
            budget_buy = st.slider(
                "매매 예산 (억원)",
                3, 50, (10, 25),
                disabled=not use_budget,
                key="budget_buy_v2",
                help="30평 기준 매매 예산 범위",
            )
        with col_b:
            ci_strict = st.checkbox(
                "보수적 필터 (95% CI 하한 사용)",
                value=True,
                disabled=not use_budget,
                key="ci_strict_v2",
                help="체크 시: 동의 95% CI 하한이 예산 내에 들어가야 통과 (보수적)",
            )
        with col_c:
            pyeong = st.number_input(
                "평수",
                10, 80, 30,
                disabled=not use_budget,
                key="pyeong_v2",
            )
        
        if use_budget:
            st.info(
                f"💰 **{budget_buy[0]}~{budget_buy[1]}억원** 범위, "
                f"**{pyeong}평** 기준, "
                f"{'95% CI 하한' if ci_strict else '점추정'} 사용"
            )
    
    # budget_filter dict 구성
    if use_budget:
        budget_filter = {
            "min_billion": float(budget_buy[0]),
            "max_billion": float(budget_buy[1]),
            "pyeong": int(pyeong),
            "use_lower_ci": bool(ci_strict),
        }
    else:
        budget_filter = None

st.divider()

# ============================================================================
# 단계 4: 세부 슬라이더
# ============================================================================
with st.expander("단계 4 · 세부 선호 (선택, 5개)"):
    st.caption("기본 페르소나에서 미세 조정합니다. 0이면 영향 없음.")
    slider_adjust = {}
    slider_adjust["AGE_UNDER20_PCT"] = st.slider(
        "어린이/청소년 비율 선호", -2.0, +2.0, 0.0, 0.5,
        help="+ 값일수록 어린이 많은 동네",
    )
    slider_adjust["VISITOR_RATIO"] = st.slider(
        "유동인구·상권 활기 선호", -2.0, +2.0, 0.0, 0.5,
    )
    slider_adjust["AVG_ASSET_LN"] = st.slider(
        "고자산층 거주 동네 선호", -2.0, +2.0, 0.0, 0.5,
    )
    slider_adjust["RESIDENT_RATIO"] = st.slider(
        "거주 안정성 (조용함) 선호", -2.0, +2.0, 0.0, 0.5,
    )
    slider_adjust["CREDIT_CARD_INTENSITY"] = st.slider(
        "외식·소비 활성도 선호", -2.0, +2.0, 0.0, 0.5,
    )
    slider_adjust = {k: v for k, v in slider_adjust.items() if v != 0.0}

st.divider()


# ============================================================================
# 추천 받기 버튼
# ============================================================================
col_left, col_right = st.columns([3, 1])
with col_left:
    diversity = st.checkbox(
        "다양성 보장 (같은 DNA 클러스터 최대 2개)", value=True,
        help="handoff §'K-means 활용처 C' — 추천이 한 클러스터에 쏠리지 않도록 필터링",
    )
    n_results = st.slider("추천 개수", 3, 10, 5, key="n_results_slider")

with col_right:
    st.write("")
    st.write("")
    run_button = st.button("🎯 추천 받기", type="primary", use_container_width=True)


# ============================================================================
# 추천 실행 (버튼 클릭 시 또는 이전 결과 유지)
# ============================================================================
if run_button:
    with st.spinner("매칭 중..."):
        result = recommend(
            template_id=selected_template_id,
            user_priorities=user_priorities,
            slider_adjustments=slider_adjust if slider_adjust else None,
            budget_filter=None,
            dong_vector_df=data["vector"],
            dong_meta_df=data["meta"],
            feature_weights_v3=weights,
            all_features=features,
            n=n_results,
            enforce_diversity=diversity,
        )
        st.session_state.last_result = result
        st.session_state.last_input = {
            "template": selected_template_id,
            "priorities": user_priorities,
        }
        # v1.1 fix — 사용자에게 결과 위치 알림 (자동 스크롤은 Streamlit 한계로 불가)
        st.success("✅ 추천 결과가 아래에 생성되었습니다. 화면을 아래로 스크롤하세요.")


# ============================================================================
# 추천 결과 표시 (세션 상태에 결과 있으면 표시)
# ============================================================================
if "last_result" in st.session_state:
    st.divider()
    st.header("🎯 추천 결과")

    result = st.session_state.last_result

    # ─────────────────────────────────────────────────────────────────
    # Redundancy 경고 (§1 A안)
    # ─────────────────────────────────────────────────────────────────
    if result["redundancy_warning"]:
        g1, g2, cos = result["redundancy_warning"]
        l1 = priority_labels[g1] if g1 in priority_labels else g1
        l2 = priority_labels[g2] if g2 in priority_labels else g2
        st.warning(
            f"⚠️ **선택한 우선순위가 매우 비슷합니다** (cos={cos:.2f}). "
            f"\n- {l1}\n- {l2}\n\n"
            f"두 항목을 모두 선택하면 페르소나가 한 방향으로 쏠릴 수 있습니다. "
            f"추천 결과는 정상 표시되지만, 다른 그룹과의 조합도 시도해보세요."
        )

    # ─────────────────────────────────────────────────────────────────
    # 입력 요약
    # ─────────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("**📋 입력 요약**")
        tmpl = HOUSEHOLD_TEMPLATES[st.session_state.last_input["template"]]
        st.markdown(f"가구 형태: {tmpl['icon']} {tmpl['label_kr']}")
        if st.session_state.last_input["priorities"]:
            ranks = []
            for i, gid in enumerate(st.session_state.last_input["priorities"], 1):
                g = PRIORITY_GROUPS[gid]
                ranks.append(f"{i}. {g['icon']} {g['label_kr']}")
            st.markdown(" → ".join(ranks))

    # ─────────────────────────────────────────────────────────────────
    # 추천 카드
    # ─────────────────────────────────────────────────────────────────
    st.subheader(f"🏆 Top {len(result['candidates'])} 추천 동네")

    # 사전 계산 (모든 카드에서 공유)
    w_cols = [f"{v}_W" for v in features]
    dong_matrix = data["vector"][w_cols].to_numpy()
    cluster_ids = data["meta"]["CLUSTER_ID"].to_numpy()

    for c in result["candidates"]:
        with st.container(border=True):
            col_main, col_score = st.columns([4, 1])

            with col_main:
                st.markdown(
                    f"### #{c['rank']} {c['district_kor_name']} "
                    f"<span style='color: #888; font-size: 0.8em'>· {c['sgg']}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"**DNA**: {c['dna_label']} — _{c['dna_summary']}_"
                )

                # 왜 추천?
                if c["explanations_kor"]:
                    with st.expander("💡 왜 이 동네를 추천했나요?"):
                        for line in c["explanations_kor"]:
                            st.markdown(f"- {line}")

            with col_score:
                st.metric(
                    label="매칭도",
                    value=f"{c['match_pct']:.1f}%",
                )
            
            # ✅ Phase 7: 적정가 정보 표시
            if value_df is not None:
                value_info = get_value_for_dong(c["district_code"], value_df)
                if value_info is not None:
                    conf_meta = get_confidence_metadata(value_info["confidence"])
                    
                    st.divider()
                    
                    col_v1, col_v2, col_v3 = st.columns([2, 2, 1])
                    with col_v1:
                        st.metric(
                            "💰 적정 시세",
                            f"{value_info['point']:,.0f}만원/평",
                            help=f"{value_info.get('cohort', '')} 코호트",
                        )
                        st.caption(
                            f"30평 ≈ **{value_info['price_30py_billion']:.1f}억원**"
                        )
                    with col_v2:
                        st.metric(
                            "95% 신뢰구간",
                            f"{value_info['lower_30py_billion']:.1f}~"
                            f"{value_info['upper_30py_billion']:.1f}억",
                        )
                        st.caption(f"폭 {value_info['ci_width_pct']:.1f}%")
                    with col_v3:
                        st.markdown(
                            f"<div style='text-align:center; padding:0.5rem; "
                            f"border-radius:0.5rem; background:{conf_meta['color']}20'>"
                            f"<div style='font-size:1.5rem'>{conf_meta['icon']}</div>"
                            f"<div style='font-size:0.8rem; color:{conf_meta['color']}; "
                            f"font-weight:bold'>{conf_meta['label']}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    
                    # 신뢰도 낮은 경우 경고
                    if value_info["confidence"] in ["LOW", "EXTRAPOLATION"]:
                        st.warning(
                            f"{conf_meta['icon']} **{conf_meta['message']}** "
                            f"({value_info.get('usage_guide', '')})"
                        )
            
            # 비슷한 동 보기 버튼 (toggle 방식)
                button_key = f"sim_btn_{c['district_code']}"
                state_key = f"show_similar_{c['district_code']}"

                if state_key not in st.session_state:
                    st.session_state[state_key] = False

                btn_label = "비슷한 동 숨기기" if st.session_state[state_key] else "비슷한 동 보기"
                if st.button(btn_label, key=button_key, use_container_width=True):
                    st.session_state[state_key] = not st.session_state[state_key]
                    st.rerun()

            # 비슷한 동 표시 (v1.1 fix — 타입 일치 보장)
            if st.session_state.get(state_key, False):
                # DISTRICT_CODE 양쪽 모두 str로 비교
                target_str = str(c["district_code"])
                meta_codes = data["meta"]["DISTRICT_CODE"].astype(str)
                matches = data["meta"][meta_codes == target_str]

                if len(matches) == 0:
                    st.warning(
                        f"⚠️ {c['district_kor_name']}에 대한 매칭을 찾을 수 없습니다. "
                        f"district_code={c['district_code']}"
                    )
                else:
                    target_idx = matches.index[0]

                    sim_idx, sim_sims = similar_dong(
                        target_idx, dong_matrix, cluster_ids,
                        same_cluster_only=True, n=3,
                    )

                    if sim_idx:
                        st.markdown("**비슷한 동네 (같은 DNA 클러스터)**")
                        for i, s in zip(sim_idx, sim_sims):
                            meta_row = data["meta"].iloc[i]
                            st.caption(
                                f"• {meta_row['DISTRICT_KOR_NAME']} ({meta_row['SGG']}) "
                                f"· 유사도 {s*100:.1f}%"
                            )
                    else:
                        st.caption("같은 DNA 클러스터에 다른 동이 없습니다.")

    # ─────────────────────────────────────────────────────────────────
    # 결과 통계 요약
    # ─────────────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("**📊 추천 결과 통계**")
        sgg_dist = {}
        cluster_dist = {}
        for c in result["candidates"]:
            sgg_dist[c["sgg"]] = sgg_dist.get(c["sgg"], 0) + 1
            cluster_dist[c["cluster_id"]] = cluster_dist.get(c["cluster_id"], 0) + 1

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**시군구 분포**")
            for sgg, n in sorted(sgg_dist.items(), key=lambda x: -x[1]):
                st.markdown(f"- {sgg}: {n}개")
        with col2:
            st.markdown("**DNA 클러스터 분포**")
            for c_id, n in sorted(cluster_dist.items()):
                dna = DNA_LABELS[c_id]
                st.markdown(f"- {dna['icon']} {dna['name']}: {n}개")
else:
    # 결과 없을 때 안내
    st.info("👆 위에서 입력 후 **🎯 추천 받기** 버튼을 눌러주세요.")


# ============================================================================
# 푸터
# ============================================================================
st.divider()
st.caption(
    "동네잇다 (Dongnae:itda) MVP v1.1 · Day 8 hotfix · "
    "데이터: SPH (Grandata) × Richgo · "
    "범위: 서초·영등포·중구 118개 법정동"
)
