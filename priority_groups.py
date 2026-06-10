"""
============================================================================
PRIORITY_GROUPS — 동네잇다 단계 2 라이프스타일 우선순위 매핑 v1.0
============================================================================
Day 7 진입 직전 (2026-04-26 기준)

설계 원칙:
  1. 6개 라이프스타일 그룹 (handoff §"단계 2 라이프스타일 우선순위")
  2. ⚠️ Σ|z| ∈ [5, 12] (단계 1 HOUSEHOLD_TEMPLATES와 동일 magnitude 범위)
     → 최종 공식 0.4·S1 + 0.4·S2 의 40:40 비율 보장
  3. 변수당 6~10개 sparse 가중치
  4. 그룹 간 의도적 직교성 (cosine matrix로 사후 검증)
  5. V3 LIFE_MATCHER_USE=TRUE 변수만 사용 (FERTILITY_CBRT EXCLUDE 준수)
  6. 매핑 정당성: 각 변수의 source = (Day 5 ANOVA | Day 6 클러스터 시그널)

가중치 적용 공식 (handoff §"단계 2"):
    PRIORITY_WEIGHTS = {1: 1.0, 2: 0.6, 3: 0.3}
    for rank, label in enumerate(user_priorities, 1):
        multiplier = PRIORITY_WEIGHTS[rank]
        for var, val in PRIORITY_GROUPS[label]["weights"].items():
            persona_vec[var] += val * multiplier * sqrt(V3_weight[var])

호환:
  - Snowflake Notebook (Excel ❌, plain dict ✅)
  - Streamlit Cloud (CSV 4개 + 이 파일 import)
============================================================================
"""

from typing import Dict, Any
import math

# ----------------------------------------------------------------------------
# 메인 매핑 딕셔너리 (6개 그룹)
# ----------------------------------------------------------------------------
# 각 항목 구조:
#   id: 식별자 (영문 snake_case)
#   label_kr: UI 표시 라벨 (드래그 정렬 카드)
#   icon: 이모지 아이콘
#   description: 한 줄 설명 (UI 보조 텍스트)
#   weights: {변수명: Z-score 단위 가중치}
#   sources: {변수명: 매핑 근거 출처} ← 정당성 보고서용
#   anti_correlation_with: 의도적 음의 상관 그룹 (검증용)
# ----------------------------------------------------------------------------

PRIORITY_GROUPS: Dict[str, Dict[str, Any]] = {

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1) 자녀 교육 환경
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "kids_education": {
        "label_kr": "자녀 교육 환경",
        "icon": "🎒",
        "description": "어린이·학생 비율, 안전한 거주 동네",
        "weights": {
            "AGE_UNDER20_PCT":     +2.0,   # HIGH: Day 6 C3 시그널 +1.86σ
            "SEG_ADULT_CHILD_PCT": +1.5,   # 부모 동반 풀 (어린이 동반 성인)
            "MEME_MOM_AVG_CBRT":   +1.5,   # HIGH: 3040 여성 = 또래 엄마 풀
            "RESIDENT_RATIO":      +1.5,   # HIGH: 거주 우선 (학군지)
            "WORKER_RATIO":        -1.0,   # HIGH: 직장가 회피 (음수)
            "VISITOR_RATIO":       -0.5,   # HIGH: 유동인구 회피
            "AVG_ASSET_LN":        +0.7,   # HIGH: 학군지 자산 신호 (약하게)
            "AGE_40S_PCT":         +0.5,   # 학부모 연령
        },
        "sources": {
            "AGE_UNDER20_PCT":     "Day 6 K=5: C3 클러스터 +1.86σ (당산동4가·신원동·방배동)",
            "SEG_ADULT_CHILD_PCT": "Day 5 ANOVA: SGG p<0.001 + 페르소나 2번 ✅ 6/7",
            "MEME_MOM_AVG_CBRT":   "Day 6: V3 HIGH 변수 - 3040 여성 모집단 (또래 엄마 풀)",
            "RESIDENT_RATIO":      "Day 6 viz3 boxplot: 학군지 패턴 강함",
            "WORKER_RATIO":        "Day 6: 자녀 가구 ↔ 직장가 음의 상관 (cosine 검증)",
            "VISITOR_RATIO":       "Day 6: C3 → C2(👻 도심 비거주) 음의 상관",
            "AVG_ASSET_LN":        "Day 6 C3: +1.73σ (학군지 가격 반영, 단 1극 쏠림 방지)",
            "AGE_40S_PCT":         "추론: 초·중등 자녀 학부모 = 40대 비율 동반 상승",
        },
        "anti_correlation_with": ["lone_lifestyle", "vibrant_atmosphere"],
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2) 출퇴근 편의
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "commute_convenience": {
        "label_kr": "출퇴근 편의",
        "icon": "🚇",
        "description": "직장 인접, 통근 시간 절약",
        "weights": {
            "WORKER_RATIO":        +2.0,   # HIGH: C1·C2 핵심 시그널
            "RESIDENT_RATIO":      -0.7,   # HIGH: 거주 비중 약함 = 직장가
            "VISITOR_RATIO":       +0.7,   # HIGH: 도심 활기 동반
            "TOTAL_POP_LN":        +0.5,   # HIGH: 도심 인구밀집
            "AGE_30S_PCT":         +0.7,   # 직장인 주력 연령
            "AGE_40S_PCT":         +0.5,
            "AGE_UNDER20_PCT":     -0.5,   # HIGH: 어린이 적음 = 도심 직장가
            "EXECUTIVE_PCT":       +0.5,   # HIGH: 사무직 비중
        },
        "sources": {
            "WORKER_RATIO":        "Day 6 K=5: C1(💼 고소득 도심 직장) +1.94σ, C2 +1.25σ",
            "RESIDENT_RATIO":      "Day 6 K=5: C2(👻 도심 비거주) -1.17σ",
            "VISITOR_RATIO":       "Day 6 viz3: 시군구 boxplot 중구 강세",
            "TOTAL_POP_LN":        "Day 6: V3 HIGH 변수 (도심 인구 밀집)",
            "AGE_30S_PCT":         "추론: 직장인 핵심 연령 (Day 5 ANOVA 보조)",
            "AGE_40S_PCT":         "추론: 직장인 보조 연령",
            "AGE_UNDER20_PCT":     "Day 6: 직장가 ↔ 어린이 음의 상관 (cosine 검증)",
            "EXECUTIVE_PCT":       "Day 6 V3 HIGH: 사무직/임원 비중 (도심 직장 시그널)",
        },
        "anti_correlation_with": ["quiet_residence", "kids_education"],
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3) 1인 라이프스타일 (카페·편의시설)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "lone_lifestyle": {
        "label_kr": "1인 라이프스타일",
        "icon": "☕",
        "description": "카페·편의시설·외식·온라인 라이프",
        "weights": {
            "VISITOR_RATIO":         +1.5,   # HIGH: 유동인구 = 상권
            "CREDIT_CARD_INTENSITY": +1.5,   # MEDIUM: 외식·소비 강도
            "AGE_20S_PCT":           +1.0,   # 20대 1인 가구
            "AGE_30S_PCT":           +1.0,   # 30대 1인 가구
            "WORKER_RATIO":          +0.7,   # 도심 회사원 1인 가구
            "HAS_ECOMMERCE":         +1.0,   # META: 온라인 주문 활동
            "AGE_UNDER20_PCT":       -0.7,   # HIGH: 어린이 적음
            "RESIDENT_RATIO":        -0.5,   # HIGH: 안정 거주 비중 약함
        },
        "sources": {
            "VISITOR_RATIO":         "Day 6 K=5: C0(🏘️ 도심 평균형) VISITOR↑ 시그널",
            "CREDIT_CARD_INTENSITY": "Day 5 ANOVA: 시군구 p<0.001 (V3에서 MEDIUM 강등됨, 정성적 시그널)",
            "AGE_20S_PCT":           "추론: 1인 가구 핵심 연령 + Day 5 카드소비 상관",
            "AGE_30S_PCT":           "추론: 1인 가구 핵심 연령",
            "WORKER_RATIO":          "Day 6 페르소나 1번 ✅ 7/7 중구 (1인 직장인)",
            "HAS_ECOMMERCE":         "Day 6 V3 META: 온라인 활동 활발 동네",
            "AGE_UNDER20_PCT":       "Day 6: 1인 가구 ↔ 어린이 음의 상관",
            "RESIDENT_RATIO":        "Day 6 C2(👻 도심 비거주) 패턴",
        },
        "anti_correlation_with": ["kids_education", "quiet_residence"],
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4) 자산 가치 안정/상승
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "asset_value": {
        "label_kr": "자산 가치 안정/상승",
        "icon": "💰",
        "description": "고자산층 거주, 시세 안정성, 신용도",
        "weights": {
            "AVG_ASSET_LN":      +2.0,   # HIGH: C3 +1.73σ
            "HIGH_INCOME_PCT":   +1.5,   # MEDIUM: C1 +1.94σ
            "AVG_INCOME_LN":     +1.0,   # MEDIUM
            "HIGH_CREDIT_PCT":   +1.5,   # HIGH: 신용도 = 자산 안정성
            "EXECUTIVE_PCT":     +1.0,   # HIGH: 임원/전문직
            "HAS_RICHGO_APT":    +1.5,   # META: 시세 데이터 있는 단지 = 안정성
            "MORTGAGE_LN":       +0.5,   # 대출 활성도 (시장 활기)
        },
        "sources": {
            "AVG_ASSET_LN":      "Day 6 K=5: C3(⭐ 프리미엄 가족) +1.73σ",
            "HIGH_INCOME_PCT":   "Day 6 K=5: C1(💼 고소득 도심) +1.94σ",
            "AVG_INCOME_LN":     "Day 6 K=5: C1 +1.63σ",
            "HIGH_CREDIT_PCT":   "Day 6 V3 HIGH: 자산층 신용도 = 시세 안정성 proxy",
            "EXECUTIVE_PCT":     "Day 6 V3 HIGH: C3 +1.73σ (전문직 비중)",
            "HAS_RICHGO_APT":    "Day 6 V3 META: 실거래 데이터 존재 = 매매 활성 단지",
            "MORTGAGE_LN":       "Day 6 K=5: C1 +1.67σ (대출 → 매매 활성)",
        },
        "anti_correlation_with": [],   # 다른 그룹과 부분 중첩 OK (자산은 보편 가치)
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5) 조용한 거주 환경
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "quiet_residence": {
        "label_kr": "조용한 거주 환경",
        "icon": "🌳",
        "description": "안정 거주, 유동인구 적음, 한적함",
        "weights": {
            "RESIDENT_RATIO":   +2.0,   # HIGH: C4 +1.18σ
            "WORKER_RATIO":     -1.5,   # HIGH: 직장가 강한 음수
            "VISITOR_RATIO":    -1.5,   # HIGH: 유동인구 강한 음수
            "TOTAL_POP_LN":     -0.5,   # HIGH: 인구밀집 회피
            "AGE_50S_PCT":      +0.7,   # 시니어 거주
            "AGE_60S_PCT":      +0.7,   # 시니어 거주
            "AGE_UNDER20_PCT":  +0.3,   # 가족 거주 약한 신호 (양수, 약함)
        },
        "sources": {
            "RESIDENT_RATIO":   "Day 6 K=5: C4(🏠 주거형 일반) +1.18σ",
            "WORKER_RATIO":     "Day 6 K=5: C4 -1.05σ (의도적 음의 상관)",
            "VISITOR_RATIO":    "Day 6 K=5: C4 -0.94σ (의도적 음의 상관)",
            "TOTAL_POP_LN":     "Day 6 viz3: 도심 인구밀집 회피",
            "AGE_50S_PCT":      "추론: 시니어 거주 + Day 5 ANOVA",
            "AGE_60S_PCT":      "추론: 시니어 거주",
            "AGE_UNDER20_PCT":  "Day 6 C4 패턴: 영등포 24동 (가족 거주 혼재)",
        },
        "anti_correlation_with": ["commute_convenience", "vibrant_atmosphere", "lone_lifestyle"],
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 6) 활기찬 분위기 (방문자/상권 많음)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "vibrant_atmosphere": {
        "label_kr": "활기찬 분위기",
        "icon": "🎉",
        "description": "유동인구·상권·외식·문화 활성",
        "weights": {
            "VISITOR_RATIO":         +2.0,   # HIGH: C0 시그널
            "TOTAL_POP_LN":          +1.0,   # HIGH: 도심 인구밀집
            "WORKER_RATIO":          +1.0,   # HIGH: 직장가 활기
            "CREDIT_CARD_INTENSITY": +1.5,   # MEDIUM: 소비 강도
            "RESIDENT_RATIO":        -1.0,   # HIGH: 안정 거주 음수
            "HAS_ECOMMERCE":         +0.5,   # META: 활성 동네
            "AGE_20S_PCT":           +0.7,   # 젊은층 활기
            "AGE_30S_PCT":           +0.5,
        },
        "sources": {
            "VISITOR_RATIO":         "Day 6 K=5: C0(🏘️ 도심 평균형) VISITOR↑",
            "TOTAL_POP_LN":          "Day 6 viz3: 시군구 boxplot 중구 강세",
            "WORKER_RATIO":          "Day 6 K=5: C2 +1.25σ (도심 활기)",
            "CREDIT_CARD_INTENSITY": "Day 5 ANOVA: 시군구 p<0.001 (외식 강도)",
            "RESIDENT_RATIO":        "Day 6: 활기 ↔ 거주 음의 상관",
            "HAS_ECOMMERCE":         "Day 6 V3 META: 도심 활성 동네 보조 시그널",
            "AGE_20S_PCT":           "추론: 젊은 인구 = 활기",
            "AGE_30S_PCT":           "추론: 젊은 인구",
        },
        "anti_correlation_with": ["quiet_residence", "kids_education"],
    },
}


# ============================================================================
# 가중치 적용 공식 (handoff §"단계 2 가중치 적용")
# ============================================================================
PRIORITY_WEIGHTS: Dict[int, float] = {
    1: 1.0,   # 1순위: 1.0×
    2: 0.6,   # 2순위: 0.6×
    3: 0.3,   # 3순위: 0.3×
}

# V3 가중치 카테고리 (handoff §"DONG_FEATURE_WEIGHTS_V3" 기반)
HIGH_VARS = {
    "TOTAL_POP_LN", "RESIDENT_RATIO", "HIGH_CREDIT_PCT",
    "AGE_UNDER20_PCT", "WORKER_RATIO", "AVG_ASSET_LN",
    "VISITOR_RATIO", "EXECUTIVE_PCT", "MEME_MOM_AVG_CBRT",
}
META_VARS = {
    "HAS_RICHGO_APT", "SPH_LOW_QUALITY", "IS_GHOST_ECONOMY",
    "LOW_RESIDENT_FLAG", "AGE_DATA_RELIABLE", "WORKER_DOMINANT",
    "HAS_ECOMMERCE",
}


# ============================================================================
# 헬퍼 함수
# ============================================================================

def list_groups() -> Dict[str, str]:
    """Streamlit 단계 2 드래그 카드 라벨 맵."""
    return {gid: g["label_kr"] for gid, g in PRIORITY_GROUPS.items()}


def get_group(group_id: str) -> Dict[str, Any]:
    if group_id not in PRIORITY_GROUPS:
        raise KeyError(
            f"Unknown group_id: {group_id}. "
            f"Available: {list(PRIORITY_GROUPS.keys())}"
        )
    return PRIORITY_GROUPS[group_id]


def build_priority_vector(
    user_priorities: list,
    feature_weights_v3: Dict[str, float],
    all_features: list,
) -> Dict[str, float]:
    """
    사용자 우선순위 (1~3순위 그룹 ID 리스트) → 페르소나 벡터 변환.
    
    공식 (handoff §"단계 2"):
        for rank, gid in enumerate(user_priorities, 1):
            multiplier = PRIORITY_WEIGHTS[rank]
            for var, z_pref in PRIORITY_GROUPS[gid]["weights"].items():
                persona_vec[var] += z_pref * multiplier * sqrt(V3_weight[var])
    
    Args:
        user_priorities: ["kids_education", "asset_value", "quiet_residence"] 등 1~3개
        feature_weights_v3: {var_name: weight} (V3 LIFE_MATCHER_USE=TRUE만)
        all_features: 전체 활성 변수 리스트 (55개)
    
    Returns:
        {var_name: priority_vec_value} — 0으로 채워진 sparse 벡터
    """
    if not (1 <= len(user_priorities) <= 3):
        raise ValueError(
            f"user_priorities는 1~3개여야 함 (현재: {len(user_priorities)}개)"
        )
    
    persona_vec = {var: 0.0 for var in all_features}
    
    for rank, gid in enumerate(user_priorities, 1):
        if gid not in PRIORITY_GROUPS:
            raise KeyError(f"Unknown priority group: {gid}")
        multiplier = PRIORITY_WEIGHTS[rank]
        z_prefs = PRIORITY_GROUPS[gid]["weights"]
        for var, z in z_prefs.items():
            if var not in feature_weights_v3:
                # V3 비활성 변수는 스킵 (FERTILITY_CBRT 등)
                continue
            weight = feature_weights_v3[var]
            persona_vec[var] += z * multiplier * math.sqrt(weight)
    
    return persona_vec


def validate_group_coverage() -> Dict[str, Dict[str, Any]]:
    """그룹별 변수 커버리지 + magnitude 점검."""
    report = {}
    for gid, g in PRIORITY_GROUPS.items():
        weights = g["weights"]
        report[gid] = {
            "n_vars": len(weights),
            "n_high": sum(1 for v in weights if v in HIGH_VARS),
            "n_meta": sum(1 for v in weights if v in META_VARS),
            "sum_abs_z": round(sum(abs(z) for z in weights.values()), 2),
            "max_abs_z": round(max(abs(z) for z in weights.values()), 2),
            "n_negative": sum(1 for z in weights.values() if z < 0),
            "anti_corr_with": g["anti_correlation_with"],
        }
    return report


def compute_group_orthogonality_matrix() -> Dict[str, Dict[str, float]]:
    """
    그룹 간 코사인 유사도 매트릭스 (정성적 직교성 점검).
    
    각 그룹 벡터를 변수공간에서 단순 dot product로 비교.
    음수면 의도적 대조, 양수면 중첩, 0 근처면 직교.
    """
    # 모든 그룹의 변수 합집합
    all_vars = set()
    for g in PRIORITY_GROUPS.values():
        all_vars.update(g["weights"].keys())
    all_vars = sorted(all_vars)
    
    # 각 그룹을 벡터로 변환
    group_vecs = {}
    for gid, g in PRIORITY_GROUPS.items():
        v = [g["weights"].get(var, 0.0) for var in all_vars]
        group_vecs[gid] = v
    
    # 코사인 유사도 매트릭스
    def cos_sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
    
    matrix = {}
    for g1 in group_vecs:
        matrix[g1] = {}
        for g2 in group_vecs:
            matrix[g1][g2] = round(cos_sim(group_vecs[g1], group_vecs[g2]), 3)
    return matrix


# ============================================================================
# 셀프 테스트
# ============================================================================

if __name__ == "__main__":
    print("=" * 78)
    print(" PRIORITY_GROUPS v1.0 — 셀프 점검")
    print("=" * 78)
    print(f"\n총 그룹 수: {len(PRIORITY_GROUPS)}")
    
    # 1) 커버리지 리포트
    print("\n[1] 변수 커버리지 리포트")
    print(f"{'group_id':25s} {'n_var':>6s} {'n_HIGH':>7s} {'n_META':>7s} "
          f"{'n_neg':>6s} {'Σ|z|':>7s} {'max|z|':>7s}")
    print("-" * 78)
    for gid, info in validate_group_coverage().items():
        print(f"{gid:25s} {info['n_vars']:>6d} {info['n_high']:>7d} "
              f"{info['n_meta']:>7d} {info['n_negative']:>6d} "
              f"{info['sum_abs_z']:>7.2f} {info['max_abs_z']:>7.2f}")
    
    # 2) Magnitude 정합성 (S1과 같은 [5, 12] 범위)
    print("\n[2] Magnitude 정합성 (S1 HOUSEHOLD_TEMPLATES와 동일 [5,12])")
    s1_range_ok = []
    for gid, info in validate_group_coverage().items():
        v = info["sum_abs_z"]
        if v < 5:
            print(f"  ⚠️  {gid}: Σ|z|={v} 너무 약함 (S2 비중 떨어짐)")
            s1_range_ok.append(False)
        elif v > 12:
            print(f"  ⚠️  {gid}: Σ|z|={v} 너무 강함 (S2 비중 과대)")
            s1_range_ok.append(False)
        else:
            print(f"  ✅  {gid}: Σ|z|={v} 정상 범위 (40:40 비율 보장)")
            s1_range_ok.append(True)
    
    # 3) 그룹 간 직교성 매트릭스
    print("\n[3] 그룹 간 코사인 유사도 매트릭스 (의도적 대조 검증)")
    matrix = compute_group_orthogonality_matrix()
    gids = list(PRIORITY_GROUPS.keys())
    short = {g: g[:8] for g in gids}
    print(f"{'':15s}", end="")
    for g in gids:
        print(f"{short[g]:>10s}", end="")
    print()
    for g1 in gids:
        print(f"{short[g1]:15s}", end="")
        for g2 in gids:
            v = matrix[g1][g2]
            mark = ""
            if g1 != g2:
                expected_anti = g2 in PRIORITY_GROUPS[g1]["anti_correlation_with"]
                if expected_anti and v < -0.1:
                    mark = "✓"   # 의도된 대조 정상
                elif expected_anti and v >= -0.1:
                    mark = "✗"   # 의도된 대조 실패
            print(f"{v:>9.2f}{mark}", end="")
        print()
    
    # 4) 의도된 대조 검증 (anti_correlation_with)
    print("\n[4] 의도된 음의 상관 (anti_correlation_with) 검증")
    anti_ok = []
    for g1 in gids:
        for g2 in PRIORITY_GROUPS[g1]["anti_correlation_with"]:
            v = matrix[g1][g2]
            if v < -0.1:
                print(f"  ✅ {g1} ↔ {g2}: cos={v:.3f} (의도된 대조 정상)")
                anti_ok.append(True)
            else:
                print(f"  ⚠️  {g1} ↔ {g2}: cos={v:.3f} (대조 약함, 검토 필요)")
                anti_ok.append(False)
    
    # 5) 종합 판정
    print("\n[5] 종합 판정")
    print(f"  Magnitude 정합성: {sum(s1_range_ok)}/{len(s1_range_ok)} 통과")
    print(f"  의도된 대조:     {sum(anti_ok)}/{len(anti_ok)} 통과")
    if all(s1_range_ok) and all(anti_ok):
        print("  ✅ 전체 정합성 OK — 작업 3 (Streamlit 매칭 엔진) 진입 가능")
    else:
        print("  ⚠️  일부 항목 검토 필요")
