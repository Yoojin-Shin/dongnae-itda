"""
============================================================================
streamlit_app.py — 동네잇다 (Dongnae:itda) Life Matcher v1.0
============================================================================
Day 7 진입 직전 (2026-04-26)

handoff §"User Input 4단계 시스템"을 그대로 구현.
mock 데이터 fallback으로 Day 8(GitHub 익스포트) 전에도 데모 가능.

실행:
    pip install streamlit pandas numpy
    streamlit run streamlit_app.py

Streamlit Cloud 배포 시 requirements.txt:
    streamlit
    pandas
    numpy
============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np

# 작업 1·2·3 모듈
from household_templates import HOUSEHOLD_TEMPLATES, list_templates
from priority_groups import PRIORITY_GROUPS, list_groups
from data_loader import load_data, get_feature_list_and_weights
from matching_engine import recommend, similar_dong, DNA_LABELS


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


@st.cache_data
def cached_features_and_weights(weights_df_csv):
    df = pd.read_csv(pd.io.common.StringIO(weights_df_csv))
    return get_feature_list_and_weights(df)


# ============================================================================
# 데이터 로드
# ============================================================================
data = cached_load_data()
features, weights = get_feature_list_and_weights(data["weights"])


# ============================================================================
# 헤더
# ============================================================================
st.title("🏘️ 동네잇다")
st.caption("당신과 잘 맞는 서울 동네를 찾아드립니다 (서초·영등포·중구 118개 법정동)")

if data["is_mock"]:
    with st.expander("ℹ️ 현재 mock 데이터 사용 중", expanded=False):
        st.warning(
            f"**Day 8 GitHub 익스포트 전이라 실제 데이터가 아닌 mock 데이터로 작동 중입니다.** "
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
            
            # 대표 동 표시
            sample = data["meta"][data["meta"]["CLUSTER_ID"] == c_id]
            if len(sample) > 0:
                names = sample["DISTRICT_KOR_NAME"].head(3).tolist()
                st.caption(f"예: {', '.join(names)}")


# ============================================================================
# 본문: 4단계 입력
# ============================================================================
tab_input, tab_result = st.tabs(["📝 페르소나 입력", "🎯 추천 결과"])


with tab_input:
    
    # ─────────────────────────────────────────────────────────────────
    # 단계 1: 가구 형태 (handoff §"단계 1")
    # ─────────────────────────────────────────────────────────────────
    st.subheader("단계 1 · 가구 형태")
    st.caption("가장 비슷한 가구 형태를 골라주세요 (필수, 30초)")
    
    template_choices = list_templates()
    template_ids = list(template_choices.keys())
    template_labels = [
        f"{HOUSEHOLD_TEMPLATES[tid]['icon']} {HOUSEHOLD_TEMPLATES[tid]['label_kr']}"
        for tid in template_ids
    ]
    
    # 2열로 라디오 표시
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
    
    # ─────────────────────────────────────────────────────────────────
    # 단계 2: 우선순위 (handoff §"단계 2")
    # ─────────────────────────────────────────────────────────────────
    st.subheader("단계 2 · 라이프스타일 우선순위")
    st.caption("중요한 순서대로 1~3개를 골라주세요 (1순위 1.5배, 2순위 1.0배, 3순위 0.5배)")
    
    priority_choices = list_groups()
    priority_ids = list(priority_choices.keys())
    priority_labels = {
        gid: f"{PRIORITY_GROUPS[gid]['icon']} {PRIORITY_GROUPS[gid]['label_kr']}"
        for gid in priority_ids
    }
    
    # 1·2·3 순위 selectbox
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
    
    # 선택된 우선순위 설명 표시
    if user_priorities:
        descs = [
            f"**{i+1}순위** · {priority_labels[g]}: {PRIORITY_GROUPS[g]['description']}"
            for i, g in enumerate(user_priorities)
        ]
        st.info(" / ".join(descs))
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────────
    # 단계 3: 예산 (handoff §"단계 3", Day 7 후 활성화)
    # ─────────────────────────────────────────────────────────────────
    with st.expander("단계 3 · 예산 필터 (선택, Day 7 Value Estimator 후 활성화)"):
        st.caption("⏸ Day 7 LightGBM 학습 완료 후 사용 가능. 현재는 placeholder.")
        col_a, col_b = st.columns(2)
        with col_a:
            budget_buy = st.slider("매매 예산 (억원)", 5, 50, (10, 25), disabled=True)
        with col_b:
            budget_rent = st.slider("전세 예산 (억원)", 1, 20, (3, 10), disabled=True)
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────────
    # 단계 4: 세부 슬라이더 (handoff §"단계 4")
    # ─────────────────────────────────────────────────────────────────
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
        # 0 값 슬라이더는 vector 영향 없음
        slider_adjust = {k: v for k, v in slider_adjust.items() if v != 0.0}
    
    st.divider()
    
    # ─────────────────────────────────────────────────────────────────
    # 추천 받기 버튼
    # ─────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 1])
    with col_left:
        diversity = st.checkbox(
            "다양성 보장 (같은 DNA 클러스터 최대 2개)", value=True,
            help="handoff §'K-means 활용처 C' — 추천이 한 클러스터에 쏠리지 않도록 필터링",
        )
        n_results = st.slider("추천 개수", 3, 10, 5)
    
    with col_right:
        st.write("")
        st.write("")
        run_button = st.button("🎯 추천 받기", type="primary", use_container_width=True)


# ============================================================================
# 결과 탭
# ============================================================================
with tab_result:
    if not run_button and "last_result" not in st.session_state:
        st.info("👈 페르소나 입력 탭에서 입력 후 '추천 받기'를 눌러주세요.")
    else:
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
                    
                    # "비슷한 동" 버튼 (handoff §"K-means 활용처 D")
                    if st.button(
                        f"비슷한 동 보기",
                        key=f"sim_{c['district_code']}",
                        use_container_width=True,
                    ):
                        st.session_state[f"show_similar_{c['district_code']}"] = True
                
                # 비슷한 동 표시
                if st.session_state.get(f"show_similar_{c['district_code']}", False):
                    target_idx = data["meta"][
                        data["meta"]["DISTRICT_CODE"] == c["district_code"]
                    ].index[0]
                    
                    w_cols = [f"{v}_W" for v in features]
                    dong_matrix = data["vector"][w_cols].to_numpy()
                    cluster_ids = data["meta"]["CLUSTER_ID"].to_numpy()
                    
                    sim_idx, sim_sims = similar_dong(
                        target_idx, dong_matrix, cluster_ids,
                        same_cluster_only=True, n=3,
                    )
                    
                    st.markdown("**비슷한 동네 (같은 DNA 클러스터)**")
                    for i, s in zip(sim_idx, sim_sims):
                        meta = data["meta"].iloc[i]
                        st.caption(
                            f"• {meta['DISTRICT_KOR_NAME']} ({meta['SGG']}) "
                            f"· 유사도 {s*100:.1f}%"
                        )
        
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


# ============================================================================
# 푸터
# ============================================================================
st.divider()
st.caption(
    "동네잇다 (Dongnae:itda) MVP v1.0 · Day 7 진입 · "
    "데이터: SPH (Grandata) × Richgo · "
    "범위: 서초·영등포·중구 118개 법정동"
)
