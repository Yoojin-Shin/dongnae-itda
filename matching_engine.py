"""
============================================================================
matching_engine.py — 동네잇다 Life Matcher 코어 v1.0
============================================================================
Day 7 진입 직전 (2026-04-26)

순수 함수 모듈 — Streamlit/Snowflake 양쪽에서 import 가능.
의존성: numpy, pandas (sklearn ❌, scipy 선택적)

핵심 함수:
  build_persona_vector()    — 단계 1+2+4 → 페르소나 벡터
  cosine_top_n()            — 코사인 유사도 + Top N
  diversity_filter()        — 같은 DNA 클러스터 3개+ 회피 (handoff §C)
  generate_explanation()    — "왜 추천?" 설명 자동 생성
  similar_dong()            — 같은 클러스터 내 추천 (handoff §D)

페르소나 합성 공식 (handoff §"User Input 4단계 시스템"):
    final = 0.4 × S1(household) + 0.4 × S2(priority) + 0.2 × S4(slider)
    (S3 예산은 후처리 필터)
============================================================================
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import math

import numpy as np
import pandas as pd

# 모듈 import — 작업 1·2 산출물
from household_templates import HOUSEHOLD_TEMPLATES, get_template
from priority_groups import PRIORITY_GROUPS, PRIORITY_WEIGHTS, get_group


# ============================================================================
# DNA 클러스터 라벨 (Day 6 결과)
# ============================================================================
DNA_LABELS: Dict[int, Dict[str, str]] = {
    0: {"icon": "🏘️", "name": "도심 평균형",     "summary": "유동인구 활발, 평균적 자산"},
    1: {"icon": "💼", "name": "고소득 도심 직장", "summary": "임원·전문직 밀집, 고소득"},
    2: {"icon": "👻", "name": "도심 비거주",      "summary": "직장인구 ≫ 거주인구"},
    3: {"icon": "⭐", "name": "프리미엄 가족",    "summary": "어린이 비율↑, 자산↑, 학군지"},
    4: {"icon": "🏠", "name": "주거형 일반",      "summary": "거주 우선, 조용한 동네"},
}


# ============================================================================
# 1. 페르소나 벡터 빌더
# ============================================================================
def build_persona_vector(
    template_id: str,
    user_priorities: List[str],
    slider_adjustments: Optional[Dict[str, float]],
    feature_weights_v3: Dict[str, float],
    all_features: List[str],
    weight_s1: float = 0.4,
    weight_s2: float = 0.4,
    weight_s4: float = 0.2,
) -> np.ndarray:
    """
    단계 1+2+4 합산하여 최종 페르소나 벡터 생성.
    
    공식: final = 0.4·S1 + 0.4·S2 + 0.2·S4 (handoff §"단계 4 시스템")
    
    Args:
        template_id: HOUSEHOLD_TEMPLATES 키 (예: "couple_kids_elementary")
        user_priorities: 1~3개 PRIORITY_GROUPS 키 (rank 순서)
        slider_adjustments: 단계 4 슬라이더 {var_name: z_pref} (선택적, None 가능)
        feature_weights_v3: V3 LIFE_MATCHER_USE=TRUE 변수 가중치 dict
        all_features: 전체 활성 변수 리스트 (DONG_VECTOR_V2 컬럼 순서)
        weight_s1, weight_s2, weight_s4: 합성 비중 (기본 40:40:20)
    
    Returns:
        np.ndarray (shape: [len(all_features),]) — 정규화 전 raw 벡터
    """
    # === S1: 가구 템플릿 → 벡터 ===
    s1 = np.zeros(len(all_features))
    template = get_template(template_id)
    for i, var in enumerate(all_features):
        if var in template["weights"] and var in feature_weights_v3:
            z_pref = template["weights"][var]
            w = feature_weights_v3[var]
            s1[i] = z_pref * math.sqrt(w)
    
    # === S2: 우선순위 그룹 → 벡터 (rank 가중) ===
    s2 = np.zeros(len(all_features))
    for rank, gid in enumerate(user_priorities, start=1):
        if rank > 3:
            break
        if gid not in PRIORITY_GROUPS:
            continue
        rank_mult = PRIORITY_WEIGHTS.get(rank, 0.0)
        group_weights = PRIORITY_GROUPS[gid]["weights"]
        for i, var in enumerate(all_features):
            if var in group_weights and var in feature_weights_v3:
                z_pref = group_weights[var]
                w = feature_weights_v3[var]
                s2[i] += z_pref * rank_mult * math.sqrt(w)
    
    # === S4: 슬라이더 (선택적) ===
    s4 = np.zeros(len(all_features))
    if slider_adjustments:
        for i, var in enumerate(all_features):
            if var in slider_adjustments and var in feature_weights_v3:
                z_pref = slider_adjustments[var]
                w = feature_weights_v3[var]
                s4[i] = z_pref * math.sqrt(w)
    
    # === 합성 ===
    final = weight_s1 * s1 + weight_s2 * s2 + weight_s4 * s4
    return final


def detect_redundant_priorities(
    user_priorities: List[str],
    threshold: float = 0.8,
) -> Optional[Tuple[str, str, float]]:
    """
    [§1 A안 반영] 사용자가 너무 비슷한 우선순위 그룹 2개 선택했는지 점검.
    
    예: lone_lifestyle + vibrant_atmosphere (cos=+0.86)
    
    Returns:
        (gid1, gid2, cos) if redundant detected, else None
    """
    from priority_groups import compute_group_orthogonality_matrix
    matrix = compute_group_orthogonality_matrix()
    
    for i, g1 in enumerate(user_priorities):
        for g2 in user_priorities[i+1:]:
            if g1 in matrix and g2 in matrix[g1]:
                cos = matrix[g1][g2]
                if cos >= threshold:
                    return (g1, g2, cos)
    return None


# ============================================================================
# 2. 코사인 유사도 + Top N
# ============================================================================
def cosine_top_n(
    persona_vec: np.ndarray,
    dong_matrix: np.ndarray,
    n: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    페르소나 벡터와 동네 벡터들 간 코사인 유사도 계산 후 Top N 반환.
    
    Args:
        persona_vec: shape [D]
        dong_matrix: shape [N, D] (DONG_VECTOR_V2의 _W 컬럼들)
        n: 상위 몇 개
    
    Returns:
        top_indices: shape [n] — dong_matrix 행 인덱스
        top_sims:    shape [n] — 코사인 유사도 값 (-1 ~ +1)
    """
    # 정규화
    p_norm = np.linalg.norm(persona_vec)
    if p_norm < 1e-10:
        # 페르소나가 0벡터인 경우 (입력 부족) — 임의 순서
        return np.arange(min(n, dong_matrix.shape[0])), np.zeros(min(n, dong_matrix.shape[0]))
    
    p_unit = persona_vec / p_norm
    d_norms = np.linalg.norm(dong_matrix, axis=1)
    d_norms = np.where(d_norms < 1e-10, 1e-10, d_norms)  # divide-by-zero 방지
    d_unit = dong_matrix / d_norms[:, None]
    
    sims = d_unit @ p_unit
    top_indices = np.argsort(-sims)[:n]
    top_sims = sims[top_indices]
    return top_indices, top_sims


# ============================================================================
# 3. 다양성 보장 필터 (handoff §"K-means 활용처 C")
# ============================================================================
def diversity_filter(
    sorted_indices: np.ndarray,
    sorted_sims: np.ndarray,
    cluster_ids: np.ndarray,
    target_n: int = 5,
    max_per_cluster: int = 2,
) -> Tuple[List[int], List[float]]:
    """
    Top N 추천 중 같은 DNA 클러스터가 max_per_cluster개를 초과하지 않도록 필터링.
    
    handoff §"K-means 활용처 C" 알고리즘 그대로:
        Top 5 중 동일 DNA가 3개 이상이면 다른 DNA에서 추천 추가
    
    Args:
        sorted_indices: cosine_top_n 결과 (이미 정렬됨)
        sorted_sims:    cosine_top_n 결과
        cluster_ids:    shape [N_total] — 각 동의 cluster_id (0~4)
        target_n:       최종 결과 개수
        max_per_cluster: 클러스터당 최대 허용 (handoff: 3 미만 = 2)
    
    Returns:
        (selected_indices, selected_sims): 다양성 적용 후 [target_n]개
    """
    selected_idx, selected_sim = [], []
    cluster_count: Dict[int, int] = {}
    
    for idx, sim in zip(sorted_indices, sorted_sims):
        c = int(cluster_ids[idx])
        if cluster_count.get(c, 0) >= max_per_cluster and len(selected_idx) >= max_per_cluster:
            continue
        selected_idx.append(int(idx))
        selected_sim.append(float(sim))
        cluster_count[c] = cluster_count.get(c, 0) + 1
        if len(selected_idx) >= target_n:
            break
    
    return selected_idx, selected_sim


# ============================================================================
# 4. "비슷한 동" 추천 (handoff §"K-means 활용처 D")
# ============================================================================
def similar_dong(
    target_idx: int,
    dong_matrix: np.ndarray,
    cluster_ids: np.ndarray,
    same_cluster_only: bool = True,
    n: int = 5,
) -> Tuple[List[int], List[float]]:
    """
    target_idx 동네와 가장 비슷한 동네들 추천.
    
    Args:
        target_idx: 기준 동네 인덱스
        same_cluster_only: True면 같은 DNA 클러스터 내에서만
        n: 반환 개수 (target 자신 제외)
    """
    target_vec = dong_matrix[target_idx]
    target_cluster = int(cluster_ids[target_idx])
    
    # 비교 대상 마스크
    if same_cluster_only:
        mask = (cluster_ids == target_cluster)
        mask[target_idx] = False  # 자기 자신 제외
    else:
        mask = np.ones(len(dong_matrix), dtype=bool)
        mask[target_idx] = False
    
    if not mask.any():
        return [], []
    
    candidate_indices = np.where(mask)[0]
    candidate_matrix = dong_matrix[mask]
    
    top_local, top_sims = cosine_top_n(target_vec, candidate_matrix, n=n)
    top_global = candidate_indices[top_local]
    return list(top_global.astype(int)), list(top_sims.astype(float))


# ============================================================================
# 5. "왜 추천?" 설명 자동 생성
# ============================================================================
def generate_explanation(
    persona_vec: np.ndarray,
    dong_vec: np.ndarray,
    all_features: List[str],
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """
    페르소나와 동네 벡터의 contribution 분해.
    
    공식: 코사인 유사도 = Σ (p_i × d_i) / (|p| |d|)
          → 변수 i의 기여도 = p_i × d_i (정규화 후)
    
    Returns:
        [(변수명, 기여도)] — 절댓값 큰 순서, 양수 기여 우선
    """
    p_norm = np.linalg.norm(persona_vec)
    d_norm = np.linalg.norm(dong_vec)
    if p_norm < 1e-10 or d_norm < 1e-10:
        return []
    
    contributions = (persona_vec * dong_vec) / (p_norm * d_norm)
    
    # 양수 기여 우선, 그 다음 절댓값 큰 순
    pos_indices = np.argsort(-contributions)  # 큰 양수 → 큰 음수
    
    result = []
    for i in pos_indices[:top_k]:
        if abs(contributions[i]) > 0.001:  # 의미 있는 기여만
            result.append((all_features[i], float(contributions[i])))
    return result


def explanation_to_korean(
    explanations: List[Tuple[str, float]],
    dong_raw_stats: Optional[pd.Series] = None,
) -> List[str]:
    """
    변수명 → 사람이 읽을 수 있는 한국어 문장.
    
    예: ("AGE_UNDER20_PCT", +0.12) → "어린이 비율이 평균보다 높음 (+1.5σ)"
    """
    # 변수명 → 한국어 매핑
    KOR_LABEL = {
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
        "HAS_ECOMMERCE":       "온라인 상거래 활성",
        "IS_GHOST_ECONOMY":    "유령 경제 동",
        "AGE_DATA_RELIABLE":   "연령 데이터 신뢰도",
        "WORKER_DOMINANT":     "직장 우세 동",
        "LOW_RESIDENT_FLAG":   "저거주 동 플래그",
        "SPH_LOW_QUALITY":     "데이터 품질 낮음",
    }
    
    sentences = []
    for var, contrib in explanations:
        kor = KOR_LABEL.get(var, var)
        sign = "잘 맞음" if contrib > 0 else "다소 차이"
        sentences.append(f"{kor} ({sign}, 기여도 {contrib:+.2f})")
    return sentences


# ============================================================================
# 6. 통합 추천 함수 (Streamlit이 호출하는 엔드포인트)
# ============================================================================
def recommend(
    template_id: str,
    user_priorities: List[str],
    slider_adjustments: Optional[Dict[str, float]],
    budget_filter: Optional[Dict[str, float]],
    dong_vector_df: pd.DataFrame,
    dong_meta_df: pd.DataFrame,
    feature_weights_v3: Dict[str, float],
    all_features: List[str],
    n: int = 5,
    enforce_diversity: bool = True,
) -> Dict[str, Any]:
    """
    Streamlit이 호출하는 통합 추천 엔드포인트.
    
    Args:
        template_id, user_priorities, slider_adjustments: 입력 단계 1·2·4
        budget_filter: 단계 3 (예산 범위) — Day 7 후 활성화. 현재는 None or dict
        dong_vector_df: DONG_VECTOR_V2 (식별자 + _W 컬럼)
        dong_meta_df:   DONG_DNA_CLUSTER_V2 + 라벨 + 시군구
        feature_weights_v3: V3 dict
        all_features: _W 접미사 제외한 변수명 리스트
    
    Returns:
        {
            "persona_vec": np.ndarray,
            "redundancy_warning": Optional[Tuple],  # §1 A안
            "candidates": [
                {
                    "rank": 1,
                    "district_code": "...",
                    "district_kor_name": "신원동",
                    "sgg": "서초구",
                    "similarity": 0.87,
                    "cluster_id": 3,
                    "dna_label": "⭐ 프리미엄 가족",
                    "explanations_kor": ["어린이 비율 잘 맞음 (+0.12)", ...],
                },
                ...
            ],
        }
    """
    # 1) 페르소나 벡터
    persona = build_persona_vector(
        template_id=template_id,
        user_priorities=user_priorities,
        slider_adjustments=slider_adjustments,
        feature_weights_v3=feature_weights_v3,
        all_features=all_features,
    )
    
    # 2) 우선순위 redundancy 점검
    warning = detect_redundant_priorities(user_priorities)
    
    # 3) 동 벡터 매트릭스 추출 (_W 접미사 컬럼들)
    w_cols = [f"{v}_W" for v in all_features]
    available_cols = [c for c in w_cols if c in dong_vector_df.columns]
    if len(available_cols) < len(w_cols):
        # 일부 _W 컬럼이 없으면 0으로 채움
        missing = [c for c in w_cols if c not in dong_vector_df.columns]
        for c in missing:
            dong_vector_df[c] = 0.0
    
    dong_matrix = dong_vector_df[w_cols].to_numpy()
    
# 4) 예산 필터 (단계 3) — Day 7 이후 활성화
    valid_mask = np.ones(len(dong_vector_df), dtype=bool)
    if budget_filter:
        # ✅ Phase 7 Value Estimator 활성화
        try:
            from value_estimator import (
                load_value_predictions,
                apply_budget_filter as ve_apply_budget_filter,
            )
            value_df = load_value_predictions()
            if value_df is not None and len(value_df) > 0:
                all_codes = dong_vector_df["DISTRICT_CODE"].astype(str).tolist()
                passed_codes = ve_apply_budget_filter(
                    all_codes,
                    value_df,
                    budget_min_billion=budget_filter.get("min_billion", 0),
                    budget_max_billion=budget_filter.get("max_billion", 100),
                    pyeong=budget_filter.get("pyeong", 30),
                    use_lower_ci=budget_filter.get("use_lower_ci", True),
                )
                passed_set = set(passed_codes)
                valid_mask = np.array([
                    str(c) in passed_set
                    for c in dong_vector_df["DISTRICT_CODE"]
                ])
        except Exception as e:
            # Value Estimator 모듈 없거나 데이터 누락 시 필터링 스킵
            print(f"[matching_engine] Budget filter skipped: {e}")
            valid_mask = np.ones(len(dong_vector_df), dtype=bool)

    # 5) 코사인 Top N (다양성 적용 시 여유분 확보)
    
    # 5) 코사인 Top N (다양성 적용 시 여유분 확보)
    pool_n = n * 4 if enforce_diversity else n
    pool_n = min(pool_n, valid_mask.sum())
    
    top_idx, top_sims = cosine_top_n(persona, dong_matrix, n=pool_n)
    
    # 6) 다양성 필터
    if enforce_diversity and "CLUSTER_ID" in dong_meta_df.columns:
        cluster_ids = dong_meta_df["CLUSTER_ID"].to_numpy()
        final_idx, final_sims = diversity_filter(
            top_idx, top_sims, cluster_ids, target_n=n, max_per_cluster=2
        )
    else:
        final_idx = list(top_idx[:n].astype(int))
        final_sims = list(top_sims[:n].astype(float))
    
    # 7) 카드 데이터 구성
    candidates = []
    for rank, (i, sim) in enumerate(zip(final_idx, final_sims), 1):
        meta_row = dong_meta_df.iloc[i]
        c_id = int(meta_row.get("CLUSTER_ID", 0))
        dna = DNA_LABELS.get(c_id, {"icon": "❓", "name": "미분류", "summary": ""})
        
        # 설명 생성
        explanations = generate_explanation(persona, dong_matrix[i], all_features, top_k=3)
        explanations_kor = explanation_to_korean(explanations)
        
        candidates.append({
            "rank": rank,
            "district_code": str(meta_row.get("DISTRICT_CODE", "")),
            "district_kor_name": str(meta_row.get("DISTRICT_KOR_NAME", "")),
            "sgg": str(meta_row.get("SGG", "")),
            "similarity": round(float(sim), 4),
            "match_pct": round(float(sim) * 100, 1),
            "cluster_id": c_id,
            "dna_label": f"{dna['icon']} {dna['name']}",
            "dna_summary": dna["summary"],
            "explanations_kor": explanations_kor,
        })
    
    return {
        "persona_vec": persona,
        "redundancy_warning": warning,
        "candidates": candidates,
    }


# ============================================================================
# 셀프 테스트
# ============================================================================
if __name__ == "__main__":
    print("=" * 78)
    print(" matching_engine.py — 셀프 점검")
    print("=" * 78)
    
    # mock 데이터로 단위 함수만 점검
    rng = np.random.default_rng(42)
    n_dong, n_feat = 118, 30
    mock_features = [f"VAR_{i}" for i in range(n_feat)]
    mock_v3 = {v: rng.choice([0.5, 1.0, 1.5]) for v in mock_features}
    
    # 가짜 페르소나 (HOUSEHOLD/PRIORITY import 필요해서 skip)
    persona = rng.standard_normal(n_feat)
    dongs = rng.standard_normal((n_dong, n_feat))
    cluster_ids = rng.integers(0, 5, size=n_dong)
    
    print("\n[1] cosine_top_n")
    idx, sims = cosine_top_n(persona, dongs, n=10)
    print(f"  Top 10 sims: {sims.round(3)}")
    assert len(idx) == 10
    print("  ✅ 통과")
    
    print("\n[2] diversity_filter")
    sel_idx, sel_sims = diversity_filter(idx, sims, cluster_ids, target_n=5, max_per_cluster=2)
    cluster_dist = {}
    for i in sel_idx:
        c = int(cluster_ids[i])
        cluster_dist[c] = cluster_dist.get(c, 0) + 1
    print(f"  선택 5개 cluster 분포: {cluster_dist}")
    assert max(cluster_dist.values()) <= 2
    print("  ✅ 통과 (클러스터당 최대 2개)")
    
    print("\n[3] similar_dong")
    sim_idx, sim_sims = similar_dong(0, dongs, cluster_ids, same_cluster_only=True, n=3)
    print(f"  동일 클러스터 내 유사 동: {len(sim_idx)}개, sims={[round(s,3) for s in sim_sims]}")
    print("  ✅ 통과")
    
    print("\n[4] generate_explanation")
    expl = generate_explanation(persona, dongs[0], mock_features, top_k=3)
    print(f"  Top 3 contributions: {[(v, round(c,3)) for v,c in expl]}")
    assert len(expl) <= 3
    print("  ✅ 통과")
    
    print("\n[5] detect_redundant_priorities")
    warn = detect_redundant_priorities(["lone_lifestyle", "vibrant_atmosphere"])
    if warn:
        print(f"  ⚠️ 감지: {warn[0]} ↔ {warn[1]} cos={warn[2]:.2f}")
        print("  ✅ §1 A안 정상 작동")
    
    warn2 = detect_redundant_priorities(["kids_education", "asset_value"])
    print(f"  정상 조합: {warn2} (None이어야 정상)")
    assert warn2 is None
    print("  ✅ 통과")
    
    print("\n✅ 매칭 엔진 코어 함수 모두 정상 작동")
