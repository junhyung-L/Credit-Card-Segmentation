# 신용카드 고객 세그먼트 분류

[English](README.md)

> [프로젝트 자세히 보기](PORTFOLIO.ko.md)

고차원·결측치 다수·희소 클래스가 공존하는 정형 데이터에서 신용카드 고객 세그먼트를 분류한 대회형 분석 기록입니다. 그래디언트 부스팅, 고전 모델, 신경망, TabNet, 스태킹 앙상블을 비교합니다.

> 아래 성능과 순위는 [`reports/실험요약.md`](reports/실험요약.md)에 보관된 기록입니다. 원본 데이터, 제출 파일, 외부 리더보드 export는 저장소에 없습니다.

## 문제와 데이터 범위

실험 요약은 약 **24만 행**, **857개 컬럼**, 다중 클래스 `Segment` 타깃을 설명합니다. 결측치가 많고 클래스 불균형이 심한 데이터이며, 주요 스크립트는 여러 월별 Parquet 소스를 결합해 피처를 만듭니다. 원천 파일 자체는 제외돼 있습니다.

## 분석 흐름

```mermaid
flowchart LR
    A[월별 원천 테이블\n저장소 미포함] --> B[월·ID 기준 결합]
    B --> C[결측 원인 분석과 규칙 기반 전처리]
    C --> D[인코딩과 스케일링]
    D --> E[표본 기반 모델 비교]
    E --> F[XGBoost·CatBoost·신경망\nTabNet·스태킹]
    F --> G[weighted F1 검증]
    G --> H[대회 제출\n저장소 미포함]
```

## 저장소에 남은 방법론

- `src/baseline_xgb.py`가 월별 테이블을 도메인별로 이어 붙이고 월과 `ID` 기준으로 left merge합니다.
- `src/missing_mechanism_analysis.py`는 결측률 계산, 규칙 기반 채움, multi-output random forest 대체 실험을 포함합니다.
- 보고서는 결측률 50% 이상 컬럼 제거, 20~50% 구간의 모델 기반 처리, 20% 미만 범주의 `Unknown` 처리를 기록합니다.
- 2만 건 표본에서 CatBoost, Logistic Regression, XGBoost, Random Forest, DNN/MLP/CNN, TabNet, 여러 스태킹 구성을 비교했습니다.

## 보관된 결과

| 실험 | F1 | 평가 맥락 |
|---|---:|---|
| Baseline XGBoost | 0.607 | public score |
| 결측 인지 XGBoost | 0.625 | public score |
| CatBoost | 0.8893 | 2만 건 검증 표본 |
| TabNet | 0.8285 | 2만 건 테스트 표본 |
| CatBoost + Logistic Regression + MLP 스태킹 | **0.8936** | 2만 건 검증 표본 |

실험 요약에는 public score **0.64636**(75위), private score **0.6251**(58위·상위 25%)도 남아 있습니다. 이는 프로젝트 산출물에 기록된 대회 결과이며, 현재 체크아웃에서 독립 재실행한 값은 아닙니다.

![보관된 모델 실험의 검증 F1 비교](images/model_f1_comparison.png)

*그림. 0.8936은 원본 전체 데이터 재실행 결과가 아니라, 보관된 검증 조건의 스태킹 기록입니다.*

## 저장소 구조와 재현성

```text
src/                         # 내보낸 Colab 전처리·학습 스크립트
notebooks/                   # 탐색 노트북과 구조 실험
reports/실험요약.md           # 보관된 실험 요약과 점수
images/model_f1_comparison.png
data/                        # 자리표시자만 존재, 원본 소스 제외
```

기본선·결측 분석·TabNet 벤치마크·전처리 도구는 `src/project_config.py`와 명령행 인자로 경로를 관리합니다. 하지만 원본 Parquet/CSV와 고정된 패키지 버전이 없어, 현재 저장소만으로 처음부터 끝까지 재실행할 수는 없습니다.

## 문서

- [포트폴리오 사례 연구](PORTFOLIO.ko.md)
- [프로젝트 리뷰](docs/PROJECT_REVIEW.md)
- [아키텍처](docs/ARCHITECTURE.md)
- [실행 매니페스트](research/RUN_MANIFEST.md)
