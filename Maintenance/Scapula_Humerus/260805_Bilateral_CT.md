# - 8/5 Bilateral CT

## Issue

```
Unilateral CT만 분할 가능함. Bilateral CT가 입력되면 분할하지 못하고 오류 메시지 작성됨.
```

## Solution

```
모델을 새로 학습시키지 않고 DICOM data로부터 정보를 얻어내어 데이터 전처리로 수행함.
```

<br>

<img width="420" height="306" alt="image" src="https://github.com/user-attachments/assets/4c454f5d-84be-4dfc-97ea-955b1dfc668f" />


### A안 결과

<img width="656" height="746" alt="image" src="https://github.com/user-attachments/assets/2e78d4ec-c43e-4b9a-bc91-e6378d2f795d" />

<br>

### B안 결과

<img width="351" height="731" alt="image" src="https://github.com/user-attachments/assets/6738fd90-c251-4d79-84ac-c5d5ae26ef61" />
