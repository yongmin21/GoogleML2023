# Binary Classification with a Software Defects Dataset
---
Kaggle Playground  
  
Final LeaderBoard Score : 210 / 1704 (상위 20% 이내)

<details>
  <summery>실패 원인 분석</summery>

    
  이번 competition은 추가 데이터 사용이 가능했다.  
  그러나 저번 대회와는 달리 추가 데이터의 분포가 kaggle에서 제공한 train set과 다르다는 의견이 있어 사용하지 않았다.  
  또한 이번 대회는 train set의 shape이 (101763, 22) 으로 꽤 많은 정도에 속한 것 같다.  
  그러나 이번 대회에서는 feature engineering을 진행할 수록 score가 내려가고, 다른 유저들 역시 model ensemble에만 집중하였다.  
  따라서 overfitting에 유의하면서 model ensemble을 적절하게 구성하는 것에 목표를 두고 진행하였다.  
  
  제출한 코드는 두 가지였다.  
  Log Transform을 적용한 뒤, 
  1. Kernel Approximation을 추가한 LR + Tree Based Models + Hill Climbing  
  2. LGBM + XGB로만 구성된 앙상블 + Hill Climbing (다른 유저 코드)  
     
  높은 성적을 받은 코드를 확인해보니 사용한 모델은 다음과 같았다.  
  Random Forest  
  Extra Trees  
  HistGradientBoosting  
  LightGBM  
  XGBoost  
  CatBoost  
  그리고 여기에 hill climbing이라는 기법을 적용해 앙상블 점수를 끌어올렸다.  
  * hill climbing 이란 일종의 Greedy 알고리즘으로, 가장 점수가 높게 나타나는 모델별 앙상블 weight를 찾아주는 방법이다.  

  
  
  그리고 Error Analysis를 잘 해야겠다.  
</details>

  
  
### 파일 설명  
input - 모델의 input으로 사용될 train, test, submission 파일들  
notebook - EDA나 테스트를 위해 사용한 jupyter notebook 파일들  
models - 학습된 모델 output pickle 파일들  
src - config, helpers, pipeline과 같은 사용자 지정 함수가 들어있는 파일들  
output - submission에 제출하기 위한 모델의 예측값 파일들  

---
### 버전별 기록들
추가 예정 ...
