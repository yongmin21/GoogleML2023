# Predict Health Outcomes of Horses
---
Kaggle Playground  
  
Final LeaderBoard Score : 464 / 1543  

<details>
  <summery>실패 원인 분석</summery>

    
  이번 competition은 추가 데이터를 사용이 가능했다.  
  kaggle에서 제공한 train set과 추가 데이터를 합치면 data shape은 약 (1600, 30) 정도로 매우 작은 데이터라 할 수 있다.  
  최종적으로 우리가 제출한 코드는 다음과 같은 방법들을 사용해 stacking, voting을 시도하여 제출하였다.  
  
  1. JH - Label Encoder, KNN imputer, Standard Scaler, HistGB  
  2. SS - NN  
  3. YM - Target Encoder, KNN imputer, (xgb, hgb, lgbm) ensemble
     
  높은 성적을 받은 코드를 확인해보니 별다른 feature engineering을 요하지 않았다.  
  그리고 shakeup과 관련한 다른 유저의 comment를 확인해보니 데이터가 작은 경우 최대한 simple 모델을 사용하고, 별다른 feature engineering을 가하지 않는 것이 overfit을 줄이는 방법이라 했다.  
  추가로 그 분이 올린 내용은 다음과 같다.  
  
  1. Public Score보다 CV Score를 판단 지표로 삼아라.  
  2. 일관성을 갖고 도전하고 과적합을 조심하라. 합성 데이터는 잡음이 많고 과적합이 매우 쉽다.  
  3. 다양한 모델을 최종 제출물로 제출하여 shakeup의 위험을 방지합니다.  

  그리고 Error Analysis를 잘 해야겠다.
</details>
---
### 파일 설명
input - 모델의 input으로 사용될 train, test, submission 파일들  
notebook - EDA나 테스트를 위해 사용한 jupyter notebook 파일들  
models - 학습된 모델 output pickle 파일들  
src - config, helpers, pipeline과 같은 사용자 지정 함수가 들어있는 파일들  
output - submission에 제출하기 위한 모델의 예측값 파일들  

---
### 버전별 기록들
추가 예정 ...
