# - 5/29 Inference Pipeline Development

```
Pipeline 파악 겸 개선
```

## Development

```
# 모델링 부분
- 모델 입력 전에 slice resampling 과정이 없이 후처리로만 계단 현상을 감당하려고 함.
        - slice thickness가 1mm 이상이면 1mm로 resample
- scapula와 humerus는 구조적으로 다른 특징을 가지고 있음에도 같은 파라미터로 후처리 함.
        - 각자 다른 파라미터로 후처리

# 효율성 부분
- 좌표 정합 과정에서 for문을 통해 vertex 하나마다 곱셈으로 옮겨주고 있었음.
        - 벡터화 해서 합성곱 연산
```

## 개선 전 Pipeline
<img width="2171" height="1041" alt="image" src="https://github.com/user-attachments/assets/2d4c665b-4e73-4432-b605-c4c2905bd5e3" />

<br>

## 개선 사항 Detail
<img width="2080" height="881" alt="image" src="https://github.com/user-attachments/assets/8f231e12-aa6e-4d39-9b33-7cdc6ceb031d" />
