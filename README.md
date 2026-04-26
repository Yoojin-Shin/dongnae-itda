# 동네잇다 (Dongnae:itda) — Day 7 진입 작업물

> 작업 1·2·3 완료 시점 (2026-04-26) — 매칭 엔진 골격 + Streamlit 앱

## 폴더 구조

```
dongnae/
├── household_templates.py      # 작업 1: 가구 템플릿 8개 (Z-score 가중치 dict)
├── priority_groups.py          # 작업 2: 라이프스타일 6개 그룹 (직교성 검증 포함)
├── matching_engine.py          # 작업 3a: 코사인 매칭 + 다양성 필터 + 설명 생성
├── data_loader.py              # 작업 3b: CSV 로딩 + mock fallback (Day 8 전 동작)
├── streamlit_app.py            # 작업 3c: 4단계 페르소나 입력 UI
├── requirements.txt            # streamlit, pandas, numpy
├── test_integration.py         # end-to-end 테스트 (페르소나 검증 5케이스)
└── data/                       # Day 8 GitHub 익스포트 후 채워짐
    ├── dong_vector_v2.csv      # (생성 예정)
    ├── dong_metadata.csv       # (생성 예정)
    ├── feature_weights_v3.csv  # (생성 예정)
    └── dong_raw_stats.csv      # (생성 예정)
```

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

`data/` 폴더가 비어있어도 mock 데이터로 작동합니다. Day 8에서 실제 CSV를 넣으면 자동으로 실제 데이터로 전환됩니다.

## 통합 테스트 실행

```bash
python test_integration.py
```

5개 페르소나 케이스를 돌려 handoff §"페르소나 검증 결과" 재현 여부 확인.

## Streamlit Cloud 배포 (Day 9~10)

1. GitHub repo에 위 파일 + data/ CSV 4개 push
2. Streamlit Cloud에서 repo 연결, `streamlit_app.py` 지정
3. requirements.txt 자동 인식 → 배포 완료 (영구 무료)

## 핵심 설계 결정 (작업 1~3 셀프 검증 통과)

- **단계 1 magnitude**: Σ\|z\| ∈ [5.8, 12.1] (8개 모두 통과)
- **단계 2 magnitude**: Σ\|z\| ∈ [6.1, 9.2] (6개 모두 통과, S1과 일관)
- **의도된 직교성**: 11/11 anti_correlation 통과
- **§1 A안**: lone+vibrant (cos=0.86) UI 경고 자동 표시
- **§3**: asset_value (직교) 다른 그룹과 자유 결합 가능
- **다양성 필터**: 같은 DNA 클러스터 최대 2개 (handoff §C 정확 구현)

## Day 7 (다음 단계) 연결점

`matching_engine.recommend()`의 `budget_filter` 파라미터가 placeholder 상태 — Day 7 LightGBM
학습 후 `dong_value_estimate.csv` 만들면 자동 활성화 (코드 변경 최소).
