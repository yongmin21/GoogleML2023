# 10/5 TechTalk_Kaggle-Alumni

### 우연수 ( AI Engineer @니트로스튜디오 )

kaggle, NLP with Diaster Tweets 상위 3%

### 연락처

- woocosmos@gmail.com
- woo-niverse.tistory.com
- kakaotalk ID : 08565

### < 사공이 많아도 산을 차곡차곡 쌓기 >

### Tip 1. 명확한 목표 설정하기

- 무엇을 하고싶은지 명확하게 정하는 것이 중요
- 무엇을 해야하는지 모르는 사람도 있다 → 이럴 경우 무임승차, 혹은 반대상황이 발생할 수 있음

### Tip 2. 데이터 기반으로 실험하기

- 무지성으로 모델링 → 비효율
- 시간이 한정되어있는 상황에서는 체계적으로 하는 것이 도움
- 처음에 어떻게 시작하면 좋을까?
    - EDA
    - 많은 EDA 예제를 어떻게 활용?
        - 그대로 따라서 타이핑하지않고 스스로 데이터에 대한 궁금점이 생기는 것이 좋음
        - 주체적으로 스스로 찾아보는 것이 좋음
    - 캐글의 장점을 활용
        - 경쟁상대임에도 자신들의 코드를 공유해줌
        - 자신에게 가장 직관적이고 단순한 코드인 바닐라코드를 학습하면서 데이터 전처리, 모델 설계등의 스텝을 밟아나가기

### Tip 3. 공유하고, 귀담아듣고, 활용하기

- 진행상황과 결과를 귀담아 듣고 마지막 포인트로 활용하기
- 다른 팀원의 좋은 결과를 활용해서 더 좋은 결과를 낼 수 있음
- 많이 했던 실수
    
    → 다른 사람의 코드를 보지 않고 마이웨이로 코드를 짰을 때 결과가 안 좋았던 경우가 많음
    
    → 팀원들과 대화를 많이 나누는 것이 좋음
    
- 과정
    - 초기에 준수한 정도의 정확도를 보여주는 베이스 모델 만들기
    - 나쁘지 않다면 팀원들에게 공유
    - 각자 베이스코드를 활용해서 여러 방법들을 시도
    - 모델 앙상블 활용하여 좋은 결과를 만들어서 캐글 상위 3%까지
    

### 프로젝트를 통해 소소하게 배운 점

- 캐글 생각보다 할만하다
    - 어려울것이라 생각해서 겁을 많이 먹어서 조금 더 도전적인 Competition을 하지 않은 것이 아쉬움
- 데이터 언더샘플링의 마법
    - 데이터 언더샘플링이 도움이 되는구나 느낌
    - 데이터 부족 문제 ( 데이터 노이즈가 너무 많고 잘못 라벨링된 데이터도 많음 )에서 언더샘플링을 해보니까 오히려 정확도가 올라갔던 경험
    - 현업에서도 똑같은 경험을 겪음
    
    → 즉 생각을 다양하게 하자
    
- 보다 적극적으로 교류를 했다면?
    - 대부분 비대면으로 진행했기에 적극적으로 교류를 하지 못해서 아쉬움

### 이건 꼭 했으면 좋겠다!

- 기록을 했으면 좋겠음
- 내가 고민하고 실패하고 성공한 경험을 모두 기록하면 좋겠음
- 이상적인 기록
    - 코드 - Github
    - 과정 - 블로그
    - 정리 - 포트폴리오

---

### 이성진 ( NLP Engineer @BHSN.AI )

### 연락처

- [sjlee@bhsn.ai](mailto:sjlee@bhsn.ai)
- https://github.com/girinman
- https://linkedin.com/in/girinman

캐글은 어떻게 진행이 되나?

### EDA( Exploratory Data Analysis )

- EDA를 통해 데이터의 분포와 패턴 파악, 이상치 확인, 특성 간의 관계 이해, 모델링 전략 수립
- 일반적인 EDA 전략
    - Data Summary : 기술통계량 ( 평균, 중앙값, 표준편차 등 ), 데이터 타입 및 결측치 확인
    - Data Visualization : 히스토그램, 박스 플롯, 산점도, 상관 행렬
    - Anomaly Detection : Z-Score, IQR
- 데이터를 열심히 분석
- 데이터를 직접 까서 보는 것이 중요하다고 생각

### Data Preprocessing

- 모델이 학습하기 적절한 형태로 preprocessing 진행
- 데이터의 종류와 틀성에 따라 처리 과정이 다르기때문에 정답이 있는 아님
- 일반적인 방법
    - Handling Missing Values or Outliers
    - Feature Engineering
    - Data Scaling
    - Data Splitting(Train/Dev/Test)
- 직접 실험해보고 결과에 어떤 영향을 미치는지 파악하는 것이 중요
- 생각보다 모델링이나 하이퍼파라미터 튜닝보다는 데이터 전처리등이 모델 성능에 크리티컬함

### Selecting Baseline Model

- 데이터분석과 전처리가 끝나면 모델을 학습시킬 차례
- 기준이 되는 Baseline Model 먼저 구축 후 추가적인 데이터 전처리나 하이퍼파라미터 튜닝 등을 거쳐 얼마나 성능이 향상되었는지 비교
- 모델은 다양한 메트릭으로 평가하는 등 정량화된 수치로 평가할 수 없는 부분이 있는 것들이 있음
- 최대한 간단하고 잘 작동하는 Baseline모델을 구축하고 전처리들을 추가하는 등 여러 실험을 해보고 기록을 해보는 것이 좋음

### Tuning Hyperparameters

- 하이퍼파라미터를 조금씩 바꾸는 것에 따라 모델의 성능이 굉장히 달라짐
- 조정해야되는 하이퍼파라미터는 굉장히 많음 → 모든 경우의 수를 적용하는 것은 힘들다
- 한 번에 한 개의 파라미터를 조절하는 것을 추천
    - 체계적으로 정리가 됨
    - 기록도 잘 해놔야 됨

### Computing resources(GPU, TPU, etc…)

- 높은 성능을 위해 크고 복잡한 모델 사용, 처리해야하는 데이터 양이 매우 많은 경우
    
    → 학습이나 전처리에 엄청나게 많은 자원이 필요
    
- GPU, TPU 등 고성능 컴퓨팅 자원을 활용
    
    → 작업에 수행되는 시간 감소 효과
    
- Kaggle에서 무료로 제공하는 Notebook, Colab 등으로 충분히 간단한 태스크는 수행 가능

### Advanced Techniques

- Model Ensemble
    - 여러개의 모델의 추론값을 합쳐서 사용하는 방법
        - 한 가지 모델은 편협한 결과를 얻을 수 있음
        - 그러나 한 가지의 모델을 사용해서 해결하는 문제에 Model Ensemble을 사용하게 되면 비용이 늘어나게 되는 단점도 존재
- Random Seed Variation
    - Random Seed를 고정하는 것이 굉장히 중요
    - 머신러닝은 확률적인 부분이 들어가는 것이 굉장히 많은데 매번 실험을 할 때마다 Seed를 random하게 하면 다시 재현하기에 굉장히 어려움
    - 시드를 고정했는데 성능이 안 좋다면 시드를 변경하는 것도 방법
- Data Augmentation
    - 데이터가 너무 적은 경우 외부에서 유사한 데이터를 가져와서 추가로 학습시켜 더 좋은 성능을 내는 경우도 존재
    - 문제와 동떨어진 경우는 오히려 성능이 떨어질 수 있음
- AutoML
    - 특정한 모델이 고정되어있을 때 하이퍼파라미트와 같은 것들을 자동으로 바꿔가며 실험을 대신해줌

### Leveraging the Community

- 문제마다 데이터 유형도 다르고 적합한 정답은 다르기 때문에 캐글 커뮤니티에서 여러가지 디스커션이나 공유된 노트북을 활용하는 것이 문제를 잘 풀 수 있는 방법이 되기도 함

https://docs.google.com/presentation/d/1B0jIlxpcTyeIUqd1yoIxe2kE3M4W0cGQ50CB-MJ9lQE/edit?resourcekey=0-MnVtgVBNwbuCh_aTFTWSFg#slide=id.g231263e5d98_0_29
