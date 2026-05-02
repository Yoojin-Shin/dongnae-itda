# Value Estimator — Model Card

## 개요
- **버전**: v1.0
- **알고리즘**: LightGBM Regressor (Random Search 50 trials)
- **타깃**: MEME_PRICE_Q (분기별 평당 매매가, 만원/평)
- **학습**: 2017Q4-2023Q3, 26개 동 × 24분기 = 624행
- **검증**: 2023Q4 Val + 2024Q1-Q4 Test

## 성능 (Test 2024Q1-Q4, n=104)

| 메트릭 | 값 | 목표 | 상태 |
|---|---:|---:|:---:|
| R² | 0.928 | >0.65 | ✅ |
| MAPE | 5.57% | <15% | ✅ |
| Within ±10% | 89.4% | >80% | ✅ |
| Within ±15% | 95.2% | >90% | ✅ |
| 95% CI Coverage | 86.5% | 92-97% | ⚠️ |

## 모델 선정 (7개 모델 비교)

| 모델 | Test R² | MAPE | 통계적 유의 |
|---|---:|---:|:---:|
| Ridge | 0.869 | 13.11% | LightGBM 대비 ✅ 유의 |
| Lasso | 0.868 | 13.26% | ✅ 유의 |
| ElasticNet | 0.800 | 9.93% | ✅ 유의 |
| RandomForest | 0.916 | 6.80% | ✅ 유의 |
| GradBoosting | 0.930 | 5.63% | ❌ 동급 |
| **LightGBM** | **0.929** | **5.52%** | — (메인) |
| KNN(k=5) | 0.926 | 4.78% | ❌ 동급 |

LightGBM 채택 근거: 선형 모델 통계적 압도, Tree-Boosting 그룹 동급에서 Quantile/SHAP/속도 종합 우위.

## 알려진 한계

### 1. 2024년 폭등기 +9.8% 평균 과소추정
- 학습 데이터에 BOOM 사이클 시작점 부재
- 분기별 누적: Q1 +4% → Q4 +12%
- 영향: 강남 인접 동 +20-32% (반포동 +32%)

### 2. CI Coverage 86.5% (목표 92% 미달)
- raw 77.9% → Bias-aware Conformal 후 86.5%
- 시장 변동기 신뢰구간 좁게 추정

### 3. 외삽 영역
- 학습 26동 = "아파트+인구+부유" 클러스터
- imputed 92동 모두 학습 분포 95% 밖
- V2 라벨링: HIGH 39 / MEDIUM 40 / LOW 27 / EXTRAPOLATION 12
- 유령경제 42동 중 41동이 EXTRAPOLATION

## 사용 가이드

| 케이스 | 권장 |
|---|---|
| 학습 26동 단기 추정 | ✅ 신뢰 가능 |
| 사이클 변동기 | ⚠️ 거시 지표 함께 |
| imputed 92동 HIGH/MEDIUM | ⚠️ 보조 참고 |
| imputed 92동 LOW | ❌ 단독 비권장 |
| 유령경제 동 (EXTRAPOLATION) | ❌ 부적합 |

## 메타데이터

- 모델 파일: `models/ve_model_final.pkl` (메인 + Quantile 3개 + 메타)
- 피처 순서: `data/feature_order.csv` (51개, 변경 금지)
- 학습 통계: `data/training_stats.csv` (drift detection용)
- Random seed: 42

## 학습된 도메인 인사이트

1. **부르주아 코어** (자산+소득+임원)이 시세 핵심 (r=0.7-0.8)
2. **시군구 효과 42%** (서초 압도)
3. **시간 효과 < 동네 정체성** (1:1.69)
4. **2024년 폭등은 모델 학습 범위 밖**
5. **유령경제 동은 모델 적용 자체 부적합**
