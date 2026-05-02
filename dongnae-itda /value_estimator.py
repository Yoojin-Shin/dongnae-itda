"""
============================================================================
value_estimator.py — 동네잇다 적정가 모듈 v1.0
============================================================================
Day 7 — Value Estimator 통합 (Phase 2 모델 LightGBM Tuned 기반)

핵심 함수:
    load_value_predictions()    — 118개 동 사전 계산 적정가 (캐시)
    load_model_package()        — 실시간 추론용 .pkl 모델 로드
    get_value_for_dong()        — 단일 동 적정가 + 95% CI 조회
    apply_budget_filter()       — 추천 결과를 예산 범위로 필터링
    enrich_with_value()         — matching_engine 결과에 적정가 결합
    get_confidence_metadata()   — 신뢰도 → 색상/메시지 매핑
    load_model_info()           — 모델 카드 정보 (UI 표시용)
    predict_realtime()          — 실시간 추론 (가상 동 시뮬레이션)

성능 (Test 2024Q1-Q4, n=104):
    R²=0.928, MAPE=5.57%, Within ±10%=89.4%

============================================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"


# ============================================================================
# 1. 데이터 로드 (사전 계산 적정가)
# ============================================================================

def load_value_predictions() -> Optional[pd.DataFrame]:
    """
    data/dong_value_predictions.csv 로드.
    
    Returns:
        DataFrame with columns:
            DISTRICT_CODE, DISTRICT_KOR_NAME, SGG, COHORT,
            PRED_POINT, PRED_LOWER_95, PRED_UPPER_95,
            FINAL_CONFIDENCE, USAGE_GUIDE, MAHAL_DIST, ...
        None if file missing.
    """
    path = DATA_DIR / "dong_value_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    # DISTRICT_CODE 타입 통일 (matching_engine과 일치)
    if "DISTRICT_CODE" in df.columns:
        df["DISTRICT_CODE"] = df["DISTRICT_CODE"].astype(str)
    return df


def _make_mock_value_predictions(meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    예측 CSV 부재 시 mock 데이터 생성 (개발/테스트용).
    실제 운영 환경에서는 호출되지 않아야 함.
    """
    rng = np.random.default_rng(42)
    rows = []
    for _, m in meta_df.iterrows():
        # 시군구별 평균 시세 가정 (만원/평)
        sgg_base = {
            "서초구": 6500,
            "영등포구": 4500,
            "중구": 4000,
        }.get(m.get("SGG", ""), 4500)
        
        point = sgg_base + rng.normal(0, sgg_base * 0.15)
        ci_width = point * 0.20
        
        rows.append({
            "DISTRICT_CODE": str(m.get("DISTRICT_CODE", "")),
            "DISTRICT_KOR_NAME": m.get("DISTRICT_KOR_NAME", ""),
            "SGG": m.get("SGG", ""),
            "COHORT": "MOCK",
            "PRED_POINT": round(point, 0),
            "PRED_LOWER_95": round(point - ci_width, 0),
            "PRED_UPPER_95": round(point + ci_width, 0),
            "FINAL_CONFIDENCE": "MOCK",
            "USAGE_GUIDE": "Mock 데이터 (예측 CSV 부재)",
            "MAHAL_DIST": 10.0,
        })
    return pd.DataFrame(rows)


# ============================================================================
# 2. 단일 동 조회
# ============================================================================

def get_value_for_dong(
    district_code: str,
    value_df: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """
    단일 동의 적정가 + 95% CI + 신뢰도 정보 조회.
    
    Args:
        district_code: 동 코드 (str)
        value_df: load_value_predictions() 반환값
    
    Returns:
        {
            "name": str,
            "sgg": str,
            "cohort": str ("LEARN_26" or "IMPUTED_92"),
            "point": float (만원/평),
            "lower_95": float,
            "upper_95": float,
            "confidence": str ("HIGH"/"MEDIUM"/"LOW"/"EXTRAPOLATION"/"MOCK"),
            "usage_guide": str,
            "price_30py_billion": float (30평 기준 억원),
            "ci_width_pct": float (CI 폭 %),
        }
        None if district_code not found.
    """
    code_str = str(district_code)
    row = value_df[value_df["DISTRICT_CODE"] == code_str]
    
    if len(row) == 0:
        return None
    
    r = row.iloc[0]
    point = float(r["PRED_POINT"])
    lower = float(r["PRED_LOWER_95"])
    upper = float(r["PRED_UPPER_95"])
    
    return {
        "name": r.get("DISTRICT_KOR_NAME", ""),
        "sgg": r.get("SGG", ""),
        "cohort": r.get("COHORT", "UNKNOWN"),
        "point": point,
        "lower_95": lower,
        "upper_95": upper,
        "confidence": r.get("FINAL_CONFIDENCE", "MEDIUM"),
        "usage_guide": r.get("USAGE_GUIDE", ""),
        "mahal_dist": float(r.get("MAHAL_DIST", 0)),
        # 30평 환산 (만원/평 × 30평 × 10000원/만원 / 1억원)
        "price_30py_billion": round(point * 30 / 10000, 2),
        "lower_30py_billion": round(lower * 30 / 10000, 2),
        "upper_30py_billion": round(upper * 30 / 10000, 2),
        "ci_width_pct": round((upper - lower) / point * 100, 1) if point > 0 else 0,
    }


# ============================================================================
# 3. 예산 필터 (단계 3 활성화)
# ============================================================================

def apply_budget_filter(
    dong_codes: List[str],
    value_df: pd.DataFrame,
    budget_min_billion: float,
    budget_max_billion: float,
    pyeong: int = 30,
    use_lower_ci: bool = True,
) -> List[str]:
    """
    추천된 동 목록을 예산 범위로 필터링.
    
    Args:
        dong_codes: 추천 동 코드 리스트
        value_df: load_value_predictions() 반환값
        budget_min_billion, budget_max_billion: 예산 범위 (억원)
        pyeong: 평수 (기본 30평)
        use_lower_ci: True면 95% CI 하한으로 필터 (보수적, 권장)
                      False면 점추정값으로 필터
    
    Returns:
        예산 통과한 동 코드 리스트 (입력 순서 유지)
    """
    if value_df is None or len(value_df) == 0:
        return dong_codes  # 데이터 없으면 필터링 스킵
    
    passed = []
    for code in dong_codes:
        info = get_value_for_dong(code, value_df)
        
        if info is None:
            # 데이터 없는 동은 통과 (보수적으로 보존)
            passed.append(code)
            continue
        
        # 30평 환산 가격 (억원)
        if use_lower_ci:
            price_billion = info["lower_30py_billion"]
        else:
            price_billion = info["price_30py_billion"]
        
        if budget_min_billion <= price_billion <= budget_max_billion:
            passed.append(code)
    
    return passed


# ============================================================================
# 4. 추천 카드에 적정가 정보 결합
# ============================================================================

def enrich_with_value(
    candidates: List[Dict[str, Any]],
    value_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    matching_engine.recommend() 결과의 candidates에 적정가 정보 추가.
    
    원본 candidate dict에 'value_estimate' 키 추가:
        {
            ...기존 필드 (rank, district_code, district_kor_name, ...),
            'value_estimate': {
                'point': 8986.0,
                'lower_95': 5056.0,
                'upper_95': 9270.0,
                'confidence': 'HIGH',
                'price_30py_billion': 26.96,
                ...
            } or None (데이터 없음)
        }
    """
    enriched = []
    for c in candidates:
        info = get_value_for_dong(c.get("district_code", ""), value_df) if value_df is not None else None
        c_new = dict(c)  # 얕은 복사
        c_new["value_estimate"] = info
        enriched.append(c_new)
    return enriched


# ============================================================================
# 5. 신뢰도 메타데이터
# ============================================================================

CONFIDENCE_META = {
    "HIGH": {
        "color": "#10B981",
        "icon": "✅",
        "label": "신뢰 가능",
        "message": "학습 데이터와 매우 유사한 동입니다.",
    },
    "MEDIUM": {
        "color": "#F59E0B",
        "icon": "⚠️",
        "label": "보조 참고용",
        "message": "다른 데이터와 함께 사용을 권장합니다.",
    },
    "LOW": {
        "color": "#F97316",
        "icon": "⚠️",
        "label": "단독 비권장",
        "message": "외삽 위험이 있습니다. 단독 의존하지 마세요.",
    },
    "EXTRAPOLATION": {
        "color": "#EF4444",
        "icon": "❌",
        "label": "적용 부적합",
        "message": "학습 분포 밖이라 신뢰도가 낮습니다.",
    },
    "MOCK": {
        "color": "#6B7280",
        "icon": "🧪",
        "label": "Mock 데이터",
        "message": "실제 모델이 아닌 임시 데이터입니다.",
    },
}


def get_confidence_metadata(confidence: str) -> Dict[str, str]:
    """신뢰도 라벨 → 색상/아이콘/메시지 dict"""
    return CONFIDENCE_META.get(confidence, CONFIDENCE_META["MEDIUM"])


# ============================================================================
# 6. 모델 메타 정보
# ============================================================================

def load_model_info() -> Dict[str, Any]:
    """모델 카드 정보 (UI 사이드바 표시용)"""
    meta_path = DATA_DIR / "ve_model_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    
    # Fallback
    return {
        "model_name": "DongnaeVE_LightGBM_v1",
        "version": "1.0.0",
        "test_metrics": {
            "R2": 0.928,
            "MAPE": 5.57,
            "Within_10pct": 89.4,
            "Within_15pct": 95.2,
            "CI_Coverage": 86.5,
        },
        "training_period": "2017Q4-2023Q3",
        "test_period": "2024Q1-Q4",
        "n_dongs_predicted": 118,
        "is_mock": True,
    }


# ============================================================================
# 7. 실시간 추론 (선택적 — 가상 동 시뮬레이션)
# ============================================================================

_MODEL_CACHE: Dict[str, Any] = {}


def load_model_package() -> Optional[Dict[str, Any]]:
    """
    실시간 추론용 통합 모델 패키지 로드 (캐시).
    
    Returns:
        {
            "main_model": LightGBM,
            "quantile_lower": LightGBM,
            "quantile_median": LightGBM,
            "quantile_upper": LightGBM,
            "feature_order": List[str],
            "calibration": Dict (선택),
            ...
        }
        None if model file missing or load failed.
    """
    if "package" in _MODEL_CACHE:
        return _MODEL_CACHE["package"]
    
    pkl_path = MODELS_DIR / "ve_model_final.pkl"
    if not pkl_path.exists():
        return None
    
    try:
        import joblib
        pkg = joblib.load(pkl_path)
        _MODEL_CACHE["package"] = pkg
        return pkg
    except Exception as e:
        print(f"[value_estimator] Model load failed: {e}")
        return None


def predict_realtime(
    feature_dict: Dict[str, float],
    model_package: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, float]]:
    """
    실시간 추론 (가상 동 또는 사용자 시뮬레이션).
    
    Args:
        feature_dict: {feature_name: value} — 51개 피처 모두 있어야 함
        model_package: load_model_package() 반환값 (None이면 자동 로드)
    
    Returns:
        {
            "point": float,
            "lower_95": float,
            "upper_95": float,
            "median": float,
        }
        None if prediction failed.
    """
    if model_package is None:
        model_package = load_model_package()
    
    if model_package is None:
        return None
    
    try:
        feature_order = model_package.get("feature_order", [])
        
        # 입력 벡터 구성 (학습 순서대로)
        x = np.array([
            [feature_dict.get(f, 0.0) for f in feature_order]
        ])
        
        # 4개 모델 추론
        main_model = model_package.get("main_model")
        q_lower = model_package.get("quantile_lower")
        q_median = model_package.get("quantile_median")
        q_upper = model_package.get("quantile_upper")
        
        if main_model is None:
            return None
        
        result = {
            "point": float(main_model.predict(x)[0]),
        }
        
        if q_lower is not None:
            result["lower_95"] = float(q_lower.predict(x)[0])
        if q_median is not None:
            result["median"] = float(q_median.predict(x)[0])
        if q_upper is not None:
            result["upper_95"] = float(q_upper.predict(x)[0])
        
        return result
    
    except Exception as e:
        print(f"[value_estimator] Realtime prediction failed: {e}")
        return None


# ============================================================================
# 8. 통계 요약 (UI 대시보드용)
# ============================================================================

def get_value_summary(value_df: pd.DataFrame) -> Dict[str, Any]:
    """
    전체 적정가 데이터 통계 요약.
    
    Returns:
        {
            "n_dongs": int,
            "confidence_dist": {label: count},
            "sgg_dist": {sgg: count},
            "price_stats": {"min": ..., "max": ..., "median": ...},
            "ghost_economy_n": int,
        }
    """
    if value_df is None or len(value_df) == 0:
        return {"n_dongs": 0}
    
    summary = {
        "n_dongs": len(value_df),
        "confidence_dist": {
            str(k): int(v)
            for k, v in value_df["FINAL_CONFIDENCE"].value_counts().items()
        } if "FINAL_CONFIDENCE" in value_df.columns else {},
        "sgg_dist": {
            str(k): int(v)
            for k, v in value_df["SGG"].value_counts().items()
        } if "SGG" in value_df.columns else {},
        "price_stats": {
            "min": float(value_df["PRED_POINT"].min()),
            "max": float(value_df["PRED_POINT"].max()),
            "median": float(value_df["PRED_POINT"].median()),
            "mean": float(value_df["PRED_POINT"].mean()),
        } if "PRED_POINT" in value_df.columns else {},
    }
    
    if "IS_GHOST_ECONOMY" in value_df.columns:
        summary["ghost_economy_n"] = int(value_df["IS_GHOST_ECONOMY"].sum())
    
    return summary


# ============================================================================
# 셀프 테스트
# ============================================================================

if __name__ == "__main__":
    print("=" * 78)
    print(" value_estimator.py — 셀프 점검")
    print("=" * 78)
    
    # 1. 데이터 로드
    df = load_value_predictions()
    if df is None:
        print("\n❌ data/dong_value_predictions.csv 없음")
    else:
        print(f"\n✅ 적정가 데이터: {df.shape}")
        print(f"   동 수: {df['DISTRICT_CODE'].nunique()}")
        print(f"   신뢰도 분포: {df['FINAL_CONFIDENCE'].value_counts().to_dict()}")
        
        # 2. 단일 동 조회 테스트
        sample = df.iloc[0]
        info = get_value_for_dong(sample["DISTRICT_CODE"], df)
        print(f"\n✅ 샘플 조회: {info['name']} ({info['sgg']})")
        print(f"   적정가: {info['point']:,.0f}만원/평")
        print(f"   95% CI: [{info['lower_95']:,.0f}, {info['upper_95']:,.0f}]")
        print(f"   30평 환산: {info['price_30py_billion']:.1f}억원 "
              f"(범위 {info['lower_30py_billion']:.1f}~{info['upper_30py_billion']:.1f})")
        print(f"   신뢰도: {info['confidence']}")
        
        # 3. 예산 필터 테스트
        all_codes = df["DISTRICT_CODE"].tolist()[:10]
        passed = apply_budget_filter(all_codes, df, 10, 25, pyeong=30)
        print(f"\n✅ 예산 필터 (10~25억원, 30평): {len(passed)}/{len(all_codes)}동 통과")
        
        # 4. 통계 요약
        summary = get_value_summary(df)
        print(f"\n✅ 통계 요약:")
        print(f"   동 수: {summary['n_dongs']}")
        print(f"   가격 범위: {summary['price_stats']['min']:.0f} ~ "
              f"{summary['price_stats']['max']:.0f}만원/평")
        print(f"   중앙값: {summary['price_stats']['median']:.0f}만원/평")
    
    # 5. 모델 정보
    info = load_model_info()
    print(f"\n✅ 모델 정보:")
    print(f"   이름: {info.get('model_name')}")
    print(f"   R²: {info['test_metrics']['R2']}, MAPE: {info['test_metrics']['MAPE']}%")
    
    # 6. 실시간 추론 모델 로드 (선택)
    pkg = load_model_package()
    if pkg:
        print(f"\n✅ 실시간 추론 모델 로드 성공")
        print(f"   피처 수: {len(pkg.get('feature_order', []))}")
        print(f"   포함 모델: {[k for k in pkg.keys() if 'model' in k.lower() or 'quantile' in k.lower()]}")
    else:
        print(f"\nℹ️ 실시간 추론 모델 미로드 (precomputed CSV로만 작동)")
    
    print("\n" + "=" * 78)
    print(" ✅ value_estimator.py 셀프 점검 완료")
    print("=" * 78)
