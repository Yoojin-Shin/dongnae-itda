"""
============================================================================
HOUSEHOLD_TEMPLATES — 동네잇다 단계 1 페르소나 가구 템플릿 v1.0
============================================================================
Day 7 진입 직전 (2026-04-26 기준)

설계 원칙:
  1. Z-score 단위 가중치 (±0.0 ~ ±2.5)
     - persona_vec[var] = z_preference × √(LIFE_MATCHER_WEIGHT[var])
     - 즉 여기서 +1.5는 "그 변수에서 +1.5σ 이상 동네 선호"의 의미
  2. V3 LIFE_MATCHER_USE=TRUE 변수만 사용 (FERTILITY_CBRT EXCLUDE 준수)
  3. 변수당 6~10개 sparse 가중치 (페르소나 평탄화 방지)
  4. Day 6 페르소나 1번 실패(고소득 신혼부부 ⚠️ 1/7) 교훈 반영:
     - 신혼부부 템플릿에서 AVG_ASSET_LN을 +1.0(MEDIUM)으로 억제
     - 시그널 분산으로 강남 1극 쏠림 차단
  5. 각 템플릿은 expected_dna_cluster로 사전 검증 가능

호환:
  - Snowflake Notebook (Excel ❌, plain dict ✅)
  - Streamlit Cloud (CSV 4개 + 이 파일 import)

사용 예:
    from household_templates import HOUSEHOLD_TEMPLATES, build_template_vector
    
    template_id = "couple_kids_elementary"
    weights = HOUSEHOLD_TEMPLATES[template_id]["weights"]
    # weights = {"AGE_UNDER20_PCT": +2.0, "SEG_ADULT_CHILD_PCT": +2.0, ...}
============================================================================
"""

from typing import Dict, Any

# ----------------------------------------------------------------------------
# 메인 템플릿 딕셔너리 (8개)
# ----------------------------------------------------------------------------
# 각 항목 구조:
#   id: 식별자 (영문 snake_case)
#   label_kr: UI 표시 라벨
#   description: 한 줄 설명
#   icon: 이모지 아이콘 (Streamlit 카드 표시용)
#   age_band: 대표 연령대 (UI 표시용)
#   weights: {변수명: Z-score 단위 가중치} - 핵심 데이터
#   expected_dna_cluster: Day 6 K=5 클러스터 중 매칭 예상 (검증용)
#   anti_pattern_note: 이 템플릿이 피해야 할 흔한 오류 (개발자 주석)
# ----------------------------------------------------------------------------

HOUSEHOLD_TEMPLATES: Dict[str, Dict[str, Any]] = {

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1) 신혼부부 (자녀 없음, 20대 후반~30대 중반)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "newlywed_no_kids": {
        "label_kr": "신혼부부 (자녀 없음)",
        "description": "맞벌이 30대 부부, 향후 5년 내 자녀 계획 가능",
        "icon": "💑",
        "age_band": "27~35세",
        "weights": {
            # 연령대 시그널 (분산)
            "AGE_30S_PCT": +1.5,
            "AGE_20S_PCT": +0.5,
            # 소득/자산 — 강남 1극 쏠림 방지를 위해 ASSET 억제
            "AVG_INCOME_LN": +1.0,
            "AVG_ASSET_LN": +0.8,        # ⚠️ HIGH 변수지만 +1.0 미만으로 의도적 억제
            "HIGH_INCOME_PCT": +0.7,
            # 거주 환경 — 신축 아파트 매매·전세 시장
            "RESIDENT_RATIO": +0.8,
            "HAS_RICHGO_APT": +1.5,      # META: 시세 데이터 있는 아파트 단지
            # 어린이 비율은 0 (자녀 없으니 무관, 음수도 X)
            # 활기 — 적당한 도심 라이프
            "VISITOR_RATIO": +0.3,
        },
        "expected_dna_cluster": "C1(💼 고소득 도심 직장) 또는 C4(🏠 주거형 일반)",
        "anti_pattern_note": (
            "Day 6 페르소나 1번 실패 교훈: AVG_ASSET_LN을 +2.0으로 두면 "
            "C3(⭐ 프리미엄 가족) 클러스터 1극 쏠림 발생. "
            "신혼부부의 실제 분포는 영등포 신축(여의도·당산)·중구(신당) 포함."
        ),
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2) 부부 + 미취학 자녀 (0~6세)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "couple_kids_preschool": {
        "label_kr": "부부 + 미취학 자녀 (0~6세)",
        "description": "어린이집·놀이터 중심, 또래 엄마 풀 중요",
        "icon": "🍼",
        "age_band": "30~38세 (부모)",
        "weights": {
            # 어린이 핵심 시그널 (강하게)
            "AGE_UNDER20_PCT": +2.0,
            "MEME_MOM_AVG_CBRT": +2.0,    # 3040 여성 = 또래 엄마 풀
            "SEG_ADULT_CHILD_PCT": +1.8,
            # 거주환경 — 직장 통근보다 거주 우선
            "RESIDENT_RATIO": +1.5,
            "WORKER_RATIO": -0.7,
            "VISITOR_RATIO": -0.4,        # 조용한 환경 선호
            # 부모 연령대
            "AGE_30S_PCT": +1.0,
            # 자산 — 양육비 + 안정성 (적당히)
            "AVG_INCOME_LN": +0.8,
            "AVG_ASSET_LN": +0.7,
            "HAS_RICHGO_APT": +1.0,
        },
        "expected_dna_cluster": "C3(⭐ 프리미엄 가족) 또는 C4(🏠 주거형 일반)",
        "anti_pattern_note": (
            "FERTILITY_CBRT는 V3에서 LIFE_MATCHER_USE=FALSE이므로 사용 금지. "
            "MEME_MOM_AVG_CBRT(3040 여성)으로 또래 엄마 풀을 대신 시그널링."
        ),
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3) 부부 + 초등 자녀 (7~13세)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "couple_kids_elementary": {
        "label_kr": "부부 + 초등 자녀 (7~13세)",
        "description": "학군·학원·놀이터·도서관, 안전한 거주 동네",
        "icon": "🎒",
        "age_band": "35~45세 (부모)",
        "weights": {
            # 어린이 핵심 시그널
            "AGE_UNDER20_PCT": +2.0,
            "SEG_ADULT_CHILD_PCT": +2.0,
            # 거주 우선 (학군지)
            "RESIDENT_RATIO": +1.5,
            "WORKER_RATIO": -1.0,
            "VISITOR_RATIO": -0.5,
            # 부모 연령대
            "AGE_30S_PCT": +0.7,
            "AGE_40S_PCT": +0.7,
            # 자산 — 학군지 매매가는 비싸므로 자산 신호 살리되 압도적이지 않게
            "AVG_ASSET_LN": +0.8,
            "HIGH_CREDIT_PCT": +0.5,
            "HAS_RICHGO_APT": +1.0,
        },
        "expected_dna_cluster": "C3(⭐ 프리미엄 가족) 또는 C4(🏠 주거형 일반)",
        "anti_pattern_note": (
            "C3의 대표동(신원동·당산동4가)이 Top 7에 들어와야 정상. "
            "Day 6 검증에서 ✅ 6/7 매칭 확인됨."
        ),
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4) 부부 + 중고생 자녀 (14~19세)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "couple_kids_teen": {
        "label_kr": "부부 + 중고생 자녀 (14~19세)",
        "description": "입시·사교육 중심, 학원가 접근성 중요",
        "icon": "📚",
        "age_band": "42~52세 (부모)",
        "weights": {
            # 어린이 비율은 약간 낮춰 (10대 후반은 비율상 줄어듦)
            "AGE_UNDER20_PCT": +1.5,
            "SEG_ADULT_CHILD_PCT": +1.5,
            # 사교육 = 자산·소득 신호 강화
            "AVG_ASSET_LN": +1.2,
            "AVG_INCOME_LN": +1.0,
            "HIGH_INCOME_PCT": +1.0,
            "EXECUTIVE_PCT": +0.7,
            "HIGH_CREDIT_PCT": +0.7,
            # 부모 연령대
            "AGE_40S_PCT": +1.5,
            "AGE_50S_PCT": +0.5,
            # 거주 우선
            "RESIDENT_RATIO": +1.0,
            "WORKER_RATIO": -0.7,
            "HAS_RICHGO_APT": +0.8,
        },
        "expected_dna_cluster": "C3(⭐ 프리미엄 가족) — 서초 강세 예상",
        "anti_pattern_note": (
            "이 템플릿은 의도적으로 강남(서초) 편향을 허용 — "
            "사교육 의존도가 높은 가구의 실제 선호 반영."
        ),
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5) 1인 가구 직장인 (20~30대, 도심 통근)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "single_worker": {
        "label_kr": "1인 가구 직장인",
        "description": "통근 편의 + 외식·카페·편의시설 풍부",
        "icon": "🧑‍💼",
        "age_band": "26~38세",
        "weights": {
            # 도심형 시그널 (Day 6 검증 ✅ 7/7 중구)
            "WORKER_RATIO": +1.5,
            "VISITOR_RATIO": +1.0,
            "RESIDENT_RATIO": -0.5,       # 의도적 음수 (도심 비거주 동선)
            # 연령대
            "AGE_20S_PCT": +1.0,
            "AGE_30S_PCT": +1.0,
            # 어린이는 음수 (없는 곳 선호)
            "AGE_UNDER20_PCT": -0.5,
            # 라이프스타일 (V3 MEDIUM 변수)
            "CREDIT_CARD_INTENSITY": +0.7,
            "HAS_ECOMMERCE": +0.5,
            # 자산은 적당히 (강남 단신 직장인도 있지만 평균은 중구)
            "AVG_INCOME_LN": +0.5,
        },
        "expected_dna_cluster": "C2(👻 도심 비거주) 또는 C0(🏘️ 도심 평균형)",
        "anti_pattern_note": (
            "Day 6 페르소나 검증 ✅ 7/7 중구 매칭. "
            "이 템플릿은 안정적이므로 Streamlit 데모용 베이스라인으로 사용 가능."
        ),
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 6) 부부 시니어 (자녀 독립, 50대 후반~60대)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "senior_couple": {
        "label_kr": "부부 시니어 (자녀 독립)",
        "description": "조용한 거주, 자산 보존, 의료·녹지 접근",
        "icon": "👴👵",
        "age_band": "55~70세",
        "weights": {
            # 시니어 연령대
            "AGE_60S_PCT": +1.8,
            "AGE_50S_PCT": +1.0,
            # 어린이 비율은 약간 음수 (조용한 환경)
            "AGE_UNDER20_PCT": -0.5,
            "SEG_ADULT_CHILD_PCT": -0.3,
            # 거주 환경 — 조용함 최우선
            "RESIDENT_RATIO": +1.5,
            "WORKER_RATIO": -1.0,
            "VISITOR_RATIO": -0.7,
            # 자산 보존 — 시니어는 자산 형성 완료
            "AVG_ASSET_LN": +1.2,
            "HIGH_CREDIT_PCT": +0.7,
            "HAS_RICHGO_APT": +1.0,
        },
        "expected_dna_cluster": "C4(🏠 주거형 일반) — 영등포 강세 예상",
        "anti_pattern_note": (
            "주의: 서초 일부 동(방배동 등)도 시니어 자산층이지만, "
            "데이터는 영등포 노후 단지(양평동·당산동) 시니어 집중도가 더 높을 수 있음. "
            "검증 필수."
        ),
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 7) 1인 가구 학생/대학원
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "single_student": {
        "label_kr": "1인 가구 학생/대학원",
        "description": "저예산 전월세, 활기찬 분위기, 대중교통",
        "icon": "🎓",
        "age_band": "20~28세",
        "weights": {
            # 20대 강한 시그널
            "AGE_20S_PCT": +2.0,
            "AGE_30S_PCT": +0.3,
            # 라이프스타일 — 카페·편의시설
            "VISITOR_RATIO": +1.0,
            "WORKER_RATIO": +0.3,         # 대학가 직장인 혼재
            "CREDIT_CARD_INTENSITY": +0.5,
            # 자산 음수 — 저예산
            "AVG_INCOME_LN": -0.5,
            "AVG_ASSET_LN": -0.7,
            "HAS_RICHGO_APT": -0.5,       # 아파트보다 원룸/오피스텔
            # 어린이는 무관 (가중치 0)
        },
        "expected_dna_cluster": "C0(🏘️ 도심 평균형) 또는 C2(👻 도심 비거주)",
        "anti_pattern_note": (
            "주의: 본 데이터(서초·영등포·중구)는 대학가 비중이 낮음. "
            "이 템플릿은 매칭 약할 수 있음 — Day 8+ 강남구·관악구 확장 시 검증."
        ),
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 8) 자영업/프리랜서 (재택+이동 가능)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "freelancer": {
        "label_kr": "자영업/프리랜서",
        "description": "재택 + 카페·공유오피스, 자유로운 시간대",
        "icon": "💻",
        "age_band": "28~45세",
        "weights": {
            # 거주+이동 혼합 — 양쪽 0.5씩
            "RESIDENT_RATIO": +0.5,
            "VISITOR_RATIO": +1.0,        # 카페 접근성 우선
            "WORKER_RATIO": +0.3,
            # 연령대 폭넓게
            "AGE_30S_PCT": +1.0,
            "AGE_40S_PCT": +0.5,
            # 온라인·소비 시그널
            "HAS_ECOMMERCE": +1.5,        # META — 온라인 활동 많음
            "CREDIT_CARD_INTENSITY": +0.7,
            # 소득은 평균 (자영업 편차 큼)
            "AVG_INCOME_LN": +0.3,
        },
        "expected_dna_cluster": "C0(🏘️ 도심 평균형)",
        "anti_pattern_note": (
            "자영업/프리랜서는 데이터에서 분리 식별 변수가 약함 — "
            "HAS_ECOMMERCE + 카페소비 패턴으로 간접 시그널링. "
            "Day 8+ 사용자 피드백으로 보정 필요."
        ),
    },
}


# ============================================================================
# 헬퍼 함수
# ============================================================================

def list_templates() -> Dict[str, str]:
    """Streamlit 단계 1 카드 표시용 라벨 맵."""
    return {tid: t["label_kr"] for tid, t in HOUSEHOLD_TEMPLATES.items()}


def get_template(template_id: str) -> Dict[str, Any]:
    """템플릿 ID로 전체 메타데이터 조회."""
    if template_id not in HOUSEHOLD_TEMPLATES:
        raise KeyError(
            f"Unknown template_id: {template_id}. "
            f"Available: {list(HOUSEHOLD_TEMPLATES.keys())}"
        )
    return HOUSEHOLD_TEMPLATES[template_id]


def build_template_vector(
    template_id: str,
    feature_weights_v3: Dict[str, float],
    all_features: list
) -> Dict[str, float]:
    """
    가구 템플릿 → 페르소나 벡터 변환.
    
    공식: persona_vec[var] = z_pref × √(LIFE_MATCHER_WEIGHT[var])
    
    Args:
        template_id: HOUSEHOLD_TEMPLATES 키
        feature_weights_v3: {var_name: weight} (V3 가중치, HIGH=1.5 등)
        all_features: 전체 활성 변수 리스트 (55개)
    
    Returns:
        {var_name: persona_vec_value} — 0으로 채워진 sparse 벡터
    """
    import math
    
    template = get_template(template_id)
    z_prefs = template["weights"]
    
    persona_vec = {var: 0.0 for var in all_features}
    
    for var, z_pref in z_prefs.items():
        if var not in feature_weights_v3:
            # V3에서 비활성 변수 (FERTILITY_CBRT 등)는 스킵
            continue
        weight = feature_weights_v3[var]
        persona_vec[var] = z_pref * math.sqrt(weight)
    
    return persona_vec


def validate_template_coverage() -> Dict[str, Dict[str, Any]]:
    """
    각 템플릿의 변수 커버리지 점검.
    
    Returns:
        {template_id: {n_vars, n_high, n_meta, sum_abs_z, ...}}
    """
    high_vars = {
        "TOTAL_POP_LN", "RESIDENT_RATIO", "HIGH_CREDIT_PCT",
        "AGE_UNDER20_PCT", "WORKER_RATIO", "AVG_ASSET_LN",
        "VISITOR_RATIO", "EXECUTIVE_PCT", "MEME_MOM_AVG_CBRT",
    }
    meta_vars = {
        "HAS_RICHGO_APT", "SPH_LOW_QUALITY", "IS_GHOST_ECONOMY",
        "LOW_RESIDENT_FLAG", "AGE_DATA_RELIABLE", "WORKER_DOMINANT",
        "HAS_ECOMMERCE",
    }
    
    report = {}
    for tid, t in HOUSEHOLD_TEMPLATES.items():
        weights = t["weights"]
        report[tid] = {
            "n_vars": len(weights),
            "n_high": sum(1 for v in weights if v in high_vars),
            "n_meta": sum(1 for v in weights if v in meta_vars),
            "sum_abs_z": round(sum(abs(z) for z in weights.values()), 2),
            "max_abs_z": round(max(abs(z) for z in weights.values()), 2),
            "expected_cluster": t["expected_dna_cluster"],
        }
    return report


# ============================================================================
# 셀프 테스트 (`python household_templates.py` 실행 시)
# ============================================================================

if __name__ == "__main__":
    print("=" * 78)
    print(" HOUSEHOLD_TEMPLATES v1.0 — 셀프 점검")
    print("=" * 78)
    print(f"\n총 템플릿 수: {len(HOUSEHOLD_TEMPLATES)}")
    print("\n[변수 커버리지 리포트]")
    print(f"{'template_id':30s} {'n_var':>6s} {'n_HIGH':>7s} {'n_META':>7s} "
          f"{'Σ|z|':>7s} {'max|z|':>7s}  expected_cluster")
    print("-" * 78)
    for tid, info in validate_template_coverage().items():
        print(f"{tid:30s} {info['n_vars']:>6d} {info['n_high']:>7d} "
              f"{info['n_meta']:>7d} {info['sum_abs_z']:>7.2f} "
              f"{info['max_abs_z']:>7.2f}  {info['expected_cluster']}")
    
    # 가중치 절댓값 합이 너무 크거나 작은 템플릿 경고
    print("\n[정합성 점검]")
    for tid, info in validate_template_coverage().items():
        if info["sum_abs_z"] < 5:
            print(f"  ⚠️  {tid}: sum_abs_z={info['sum_abs_z']} 너무 약함 (시그널 부족)")
        elif info["sum_abs_z"] > 18:
            print(f"  ⚠️  {tid}: sum_abs_z={info['sum_abs_z']} 너무 강함 (페르소나 극단화)")
        else:
            print(f"  ✅  {tid}: sum_abs_z={info['sum_abs_z']} 정상 범위")
