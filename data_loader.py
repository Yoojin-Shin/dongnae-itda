"""
============================================================================
data_loader.py — 동네잇다 CSV 로더 + mock fallback v1.0
============================================================================
Day 7 진입 직전 (2026-04-26)

Day 8(GitHub 익스포트) 전이므로 CSV 파일이 아직 없음.
→ 파일 존재 시: 실제 로드
→ 파일 부재 시: mock 데이터 생성 (UI 데모 가능)

handoff §"Day 8: GitHub 익스포트" 4개 CSV:
  dong_vector_v2.csv     (~50KB, 추천 엔진 핵심)
  dong_metadata.csv      (~10KB, 식별자 + 클러스터)
  feature_weights_v3.csv (~5KB, 페르소나 → 벡터 변환용)
  dong_raw_stats.csv     (~30KB, UI 표시용 원본 값)
============================================================================
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).parent / "data"


# ============================================================================
# 1. 실제 CSV 로딩
# ============================================================================
def _load_real_csv(filename: str) -> Optional[pd.DataFrame]:
    """data/ 디렉토리에서 CSV 로딩, 없으면 None."""
    path = DATA_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


# ============================================================================
# 2. Mock 데이터 생성 (Day 8 전 데모용)
# ============================================================================
def _make_mock_data(seed: int = 42) -> Dict[str, pd.DataFrame]:
    """handoff §"Day 6 핵심 발견사항" 기반 현실적 mock 생성."""
    rng = np.random.default_rng(seed)
    
    # ─────────────────────────────────────────────────────────────────
    # Day 6 클러스터 분포 (handoff §"5개 DNA 클러스터 명명")
    # ─────────────────────────────────────────────────────────────────
    cluster_dist = [
        # (cluster_id, n, sgg_dist) — handoff §"K=5 결과 표"
        (0, 33, [("중구", 29), ("영등포구", 3), ("서초구", 1)]),
        (1, 11, [("중구", 11)]),
        (2, 25, [("중구", 25)]),
        (3, 17, [("서초구", 9), ("영등포구", 7), ("중구", 1)]),
        (4, 32, [("영등포구", 24), ("중구", 8)]),
    ]
    
    # 대표 동 이름 (handoff §"대표 동")
    represent_dong = {
        0: ["필동2가", "쌍림동", "남산동2가"],
        1: ["저동1가", "산림동", "을지로4가"],
        2: ["다동", "예관동", "태평로2가"],
        3: ["당산동4가", "신원동", "방배동"],
        4: ["양평동4가", "당산동3가", "양평동2가"],
    }
    
    # 활성 변수 리스트 (handoff §"DONG_VECTOR_V2" 핵심 변수만 24개)
    features = [
        "TOTAL_POP_LN", "RESIDENT_RATIO", "WORKER_RATIO", "VISITOR_RATIO",
        "AGE_UNDER20_PCT", "AGE_20S_PCT", "AGE_30S_PCT", "AGE_40S_PCT",
        "AGE_50S_PCT", "AGE_60S_PCT",
        "AVG_INCOME_LN", "AVG_ASSET_LN", "HIGH_INCOME_PCT", "HIGH_CREDIT_PCT",
        "EXECUTIVE_PCT", "MEME_MOM_AVG_CBRT", "SEG_ADULT_CHILD_PCT",
        "CREDIT_CARD_INTENSITY", "MORTGAGE_LN",
        "HAS_RICHGO_APT", "HAS_ECOMMERCE", "AGE_DATA_RELIABLE",
        "WORKER_DOMINANT", "IS_GHOST_ECONOMY",
    ]
    
    # 클러스터별 시그널 (Day 6 §"K=5 결과 표"의 σ값 그대로)
    cluster_signals = {
        0: {"VISITOR_RATIO": +0.5, "AGE_DATA_RELIABLE": +0.7, "AVG_ASSET_LN": -0.3},
        1: {"HIGH_INCOME_PCT": +1.94, "MORTGAGE_LN": +1.67, "AVG_INCOME_LN": +1.63,
            "EXECUTIVE_PCT": +1.0, "WORKER_RATIO": +1.5},
        2: {"WORKER_RATIO": +1.25, "RESIDENT_RATIO": -1.17, "TOTAL_POP_LN": -1.48,
            "IS_GHOST_ECONOMY": +1.0, "VISITOR_RATIO": +0.7},
        3: {"AGE_UNDER20_PCT": +1.86, "EXECUTIVE_PCT": +1.73, "AVG_ASSET_LN": +1.73,
            "MEME_MOM_AVG_CBRT": +1.5, "HAS_RICHGO_APT": +1.0,
            "SEG_ADULT_CHILD_PCT": +1.5, "RESIDENT_RATIO": +1.0},
        4: {"RESIDENT_RATIO": +1.18, "WORKER_RATIO": -1.05, "VISITOR_RATIO": -0.94,
            "AGE_50S_PCT": +0.5, "AGE_60S_PCT": +0.5},
    }
    
    # ─────────────────────────────────────────────────────────────────
    # DataFrame 생성
    # ─────────────────────────────────────────────────────────────────
    rows_meta = []
    rows_vec = []
    code_counter = 1100000000
    
    for c_id, n_dong, sgg_breakdown in cluster_dist:
        signal = cluster_signals.get(c_id, {})
        names_pool = represent_dong[c_id] + [f"가상{c_id}-{i}" for i in range(20)]
        name_idx = 0
        
        for sgg, n_in_sgg in sgg_breakdown:
            for _ in range(n_in_sgg):
                code = str(code_counter)
                code_counter += 1
                name = names_pool[name_idx % len(names_pool)]
                if name_idx >= len(represent_dong[c_id]):
                    name = f"{name}{name_idx-len(represent_dong[c_id])+1}"
                name_idx += 1
                
                # 거리 (centroid에서 떨어진 정도)
                distance = abs(rng.normal(0, 0.5)) + 0.2
                
                rows_meta.append({
                    "DISTRICT_CODE": code,
                    "DISTRICT_KOR_NAME": name,
                    "SGG": sgg,
                    "CITY_CODE": "11",
                    "CLUSTER_ID": c_id,
                    "DISTANCE_TO_CENTROID": round(distance, 4),
                    "K_VALUE": 5,
                })
                
                # 벡터 생성: cluster signal + noise
                vec_row = {
                    "DISTRICT_CODE": code,
                    "DISTRICT_KOR_NAME": name,
                    "SGG": sgg,
                    "CITY_CODE": "11",
                }
                for v in features:
                    base = signal.get(v, 0.0)
                    noise = rng.normal(0, 0.7)
                    z = base + noise
                    weight = 1.5 if v in {"TOTAL_POP_LN", "RESIDENT_RATIO", "HIGH_CREDIT_PCT",
                                          "AGE_UNDER20_PCT", "WORKER_RATIO", "AVG_ASSET_LN",
                                          "VISITOR_RATIO", "EXECUTIVE_PCT", "MEME_MOM_AVG_CBRT"} else 1.0
                    vec_row[f"{v}_W"] = round(z * np.sqrt(weight), 4)
                rows_vec.append(vec_row)
    
    df_meta = pd.DataFrame(rows_meta)
    df_vec = pd.DataFrame(rows_vec)
    
    # ─────────────────────────────────────────────────────────────────
    # feature_weights_v3
    # ─────────────────────────────────────────────────────────────────
    high_set = {"TOTAL_POP_LN", "RESIDENT_RATIO", "HIGH_CREDIT_PCT",
                "AGE_UNDER20_PCT", "WORKER_RATIO", "AVG_ASSET_LN",
                "VISITOR_RATIO", "EXECUTIVE_PCT", "MEME_MOM_AVG_CBRT"}
    meta_set = {"HAS_RICHGO_APT", "HAS_ECOMMERCE", "AGE_DATA_RELIABLE",
                "WORKER_DOMINANT", "IS_GHOST_ECONOMY"}
    
    weights_rows = []
    for v in features:
        if v in high_set:
            cat, w = "HIGH", 1.5
        elif v in meta_set:
            cat, w = "META", 1.0
        else:
            cat, w = "MEDIUM", 1.0
        weights_rows.append({
            "VARIABLE_NAME": v,
            "WEIGHT_CATEGORY": cat,
            "LIFE_MATCHER_WEIGHT": w,
            "LIFE_MATCHER_USE": True,
        })
    df_weights = pd.DataFrame(weights_rows)
    
    # ─────────────────────────────────────────────────────────────────
    # raw_stats (UI 표시용 — 시그너처 변수만)
    # ─────────────────────────────────────────────────────────────────
    raw_rows = []
    for _, vec_row in df_vec.iterrows():
        # _W 값을 역변환해서 그럴듯한 raw 값 생성
        # 실제로는 DONG_INTEGRATED_PROCESSED에서 가져와야 하지만 mock이므로 생성
        raw = {
            "DISTRICT_CODE": vec_row["DISTRICT_CODE"],
            "DISTRICT_KOR_NAME": vec_row["DISTRICT_KOR_NAME"],
            "SGG": vec_row["SGG"],
            "TOTAL_POP": int(np.exp(rng.normal(8.5, 1.0))),
            "RESIDENT_RATIO": round(np.clip(rng.normal(0.4, 0.2), 0, 1), 3),
            "WORKER_RATIO": round(np.clip(rng.normal(0.35, 0.2), 0, 1), 3),
            "VISITOR_RATIO": round(np.clip(rng.normal(0.25, 0.15), 0, 1), 3),
            "AGE_UNDER20_PCT": round(np.clip(rng.normal(15, 7), 1, 35), 1),
            "AVG_ASSET_MIL_KRW": int(rng.uniform(100, 800)),
            "AVG_INCOME_MIL_KRW": int(rng.uniform(40, 150)),
        }
        raw_rows.append(raw)
    df_raw = pd.DataFrame(raw_rows)
    
    return {
        "vector": df_vec,
        "meta": df_meta,
        "weights": df_weights,
        "raw_stats": df_raw,
    }


# ============================================================================
# 3. 통합 로딩 (실제 → mock fallback)
# ============================================================================
def load_data() -> Dict[str, pd.DataFrame]:
    """
    4개 CSV 로딩, 없으면 mock 생성.
    
    Returns:
        {"vector", "meta", "weights", "raw_stats", "is_mock"}
    """
    file_map = {
        "vector":    "dong_vector_v2.csv",
        "meta":      "dong_metadata.csv",
        "weights":   "feature_weights_v3.csv",
        "raw_stats": "dong_raw_stats.csv",
    }
    
    real_data = {k: _load_real_csv(v) for k, v in file_map.items()}
    
    if any(v is None for v in real_data.values()):
        mock = _make_mock_data()
        mock["is_mock"] = True
        mock["missing_files"] = [f for k, f in file_map.items() if real_data[k] is None]
        return mock
    
    real_data["is_mock"] = False
    return real_data


def get_feature_list_and_weights(
    weights_df: pd.DataFrame
) -> Tuple[List[str], Dict[str, float]]:
    """
    LIFE_MATCHER_USE=TRUE 변수만 추출.
    
    실제 CSV(VARIABLE/TIER)와 mock(VARIABLE_NAME/WEIGHT_CATEGORY) 둘 다 자동 인식.
    """
    # 변수명 컬럼 자동 탐지 (대소문자 무관)
    var_col = next(
        (c for c in weights_df.columns
         if c.upper() in ('VARIABLE', 'VARIABLE_NAME', 'FEATURE_NAME')),
        None
    )
    if var_col is None:
        raise ValueError(
            f"변수명 컬럼을 찾을 수 없음. 실제 컬럼: {list(weights_df.columns)}. "
            f"VARIABLE 또는 VARIABLE_NAME 중 하나가 있어야 함."
        )
    
    # LIFE_MATCHER_USE: bool 또는 'TRUE' 문자열 둘 다 처리
    if weights_df["LIFE_MATCHER_USE"].dtype == bool:
        active_mask = weights_df["LIFE_MATCHER_USE"] == True
    else:
        active_mask = weights_df["LIFE_MATCHER_USE"].astype(str).str.upper() == 'TRUE'
    
    active = weights_df[active_mask]
    features = active[var_col].tolist()
    weights = dict(zip(active[var_col], active["LIFE_MATCHER_WEIGHT"]))
    return features, weights


# ============================================================================
# 셀프 테스트
# ============================================================================
if __name__ == "__main__":
    print("=" * 78)
    print(" data_loader.py — 셀프 점검 (mock fallback 동작 확인)")
    print("=" * 78)
    
    data = load_data()
    print(f"\nis_mock: {data['is_mock']}")
    if data["is_mock"]:
        print(f"missing_files: {data.get('missing_files', [])}")
    
    print(f"\nvector:    {data['vector'].shape}")
    print(f"meta:      {data['meta'].shape}")
    print(f"weights:   {data['weights'].shape}")
    print(f"raw_stats: {data['raw_stats'].shape}")
    
    # 검증
    print("\n[검증]")
    n = len(data["meta"])
    assert len(data["vector"]) == n, "vector/meta 행 수 불일치"
    assert len(data["raw_stats"]) == n, "raw_stats/meta 행 수 불일치"
    print(f"  ✅ 모든 테이블 행 수 일치 ({n} 동)")
    
    # 클러스터 분포 (handoff §"5개 DNA 클러스터")
    print("\n[클러스터 분포]")
    cluster_count = data["meta"]["CLUSTER_ID"].value_counts().sort_index()
    expected = {0: 33, 1: 11, 2: 25, 3: 17, 4: 32}
    for c_id, count in cluster_count.items():
        exp = expected.get(c_id, 0)
        ok = "✅" if count == exp else "⚠️"
        print(f"  {ok} C{c_id}: {count}동 (예상 {exp})")
    
    # SGG 분포
    print("\n[SGG 분포]")
    print(data["meta"]["SGG"].value_counts().to_dict())
    
    # 활성 변수 추출
    features, weights = get_feature_list_and_weights(data["weights"])
    print(f"\n활성 변수 수: {len(features)}")
    print(f"HIGH 변수: {[v for v, w in weights.items() if w == 1.5]}")
    
    print("\n✅ 데이터 로더 정상 작동")
