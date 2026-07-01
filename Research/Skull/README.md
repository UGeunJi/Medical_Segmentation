# Skull_Segmentation


## - Purpose

```
두경부 결손 환자 재건 수술을 위한 부품 설계 보조 AI
```


## - Process

#### Skull Segmentation &#10145; Skull Implant

1. DB 준비
   - CQ500 (Open)
   - IMPACT (Open)
   - Samsung (Private)
     
2. Pre-processing
   - dcm to nii
   - Image Crop to skull
   - Crop Background
   - HU Clipping
   - Z-score Normalization
     
3. Segmentation
   - 5-fold Cross-Validation (Select Best DICE Score Model)
  
4. Post-processing
   - Mask
      - Isotrophic Resampling
      - Connected Component Analysis
      - Gaussian Blur
   - Mesh (After Marching-cube)
      - HC Laplacian Smoothing
      - Quadric Decimation
      - Hole Closing
      - Remeshing

5. 결손 Subject 생성 - [생성 코드](https://github.com/Jianningli/SciData) (Segmentation 전 부위 다 하면 수행 예정)
6. Imaplant Learning
   - Develop

<br>
   
--- 

<br>


### - DB

#### Open DB
- CQ500 (50명)
- IMPACT (60명)

#### Private DB
- 삼성병원 (약 400명)

<br>

### - Model

```
3D nnU-Net Version2
```

<br>

### - Pre-processing Table

| 항목 | U-Net | orient / resampling | nnUNet |
| :---: | :--: | :--: | :--: |
| HU clip | [-1000, 3000] | 동일 | 0.5/99.5 percentile (자동) |
| Norm | /4000 &#10145; uint8 | 동일 | Z-score (자동) |
| Resampling | X | 사용자 지정 | median spacing (자동) + anisotropy 처리 |
| Orientation | X | LPS/RAS | X |
| Foreground crop | X | X | V |
| Patch size | 고정 | 고정 | 자동 결정 (GPU 최적) |
| Augmentation | 사용자 정의 | 동일 | 풍부한 default set |
| 5-fold CV | 별도 구현 | 별도 구현 | 자동 |

<br>

### - Preprocessing / Model Performance Comparison Table

<table border="1" style="border-collapse: collapse; width: 100%; table-layout: fixed;">
    <thead style="background-color: #f2f2f2;">
        <tr>
            <th colspan="2" rowspan="2" style="padding: 10px; text-align: center; vertical-align: middle;">Dataset</th>
            <th rowspan="2" style="text-align: center; vertical-align: middle;">Subjects</th>
            <th rowspan="2" style="text-align: center; vertical-align: middle;">(Orient / Resample)<br>(2D &rarr; 3D)</th>
            <th colspan="2" style="text-align: center; vertical-align: middle;">Model</th>
            <th rowspan="2" style="text-align: center; vertical-align: middle;">Patchwork</th>
        </tr>
        <tr>
            <th style="text-align: center; vertical-align: middle;">U-Net</th>
            <th style="text-align: center; vertical-align: middle;">nnU-Net</th>
        </tr>
    </thead>
    <tbody style="text-align: center; vertical-align: middle;">
        <tr>
            <td rowspan="8" style="font-weight: bold; text-align: center; vertical-align: middle;">Experiments</td>
            <td rowspan="4" align="center" style="text-align: center; vertical-align: middle;">삼성서울</td>
            <td rowspan="2" align="center" style="text-align: center; vertical-align: middle;">13</td>
            <td align="center" style="height: 35px; text-align: center; vertical-align: middle;">-</td> 
            <td align="center" style="text-align: center; vertical-align: middle;">0.9468</td>
            <td align="center" style="text-align: center; vertical-align: middle;"><strong>0.9665</strong></td>
            <td align="center" style="text-align: center; vertical-align: middle;">-</td>
        </tr>
        <tr>
            <td align="center" style="text-align: center; vertical-align: middle;">V</td>
            <td align="center" style="text-align: center; vertical-align: middle;">0.8837</td>
            <td align="center" style="text-align: center; vertical-align: middle;">0.9577</td>
            <td align="center" style="text-align: center; vertical-align: middle;">-</td>
        </tr>
        <tr>
            <td align="center" rowspan="2" style="text-align: center; vertical-align: middle;">109</td>
            <td align="center" style="height: 35px; text-align: center; vertical-align: middle;">-</td>
            <td align="center" style="text-align: center; vertical-align: middle;">0.7859</td>
            <td align="center" style="text-align: center; vertical-align: middle;">0.7878</td>
            <td align="center" style="text-align: center; vertical-align: middle;">-</td>
        </tr>
        <tr>
            <td align="center" style="text-align: center; vertical-align: middle;">V</td>
            <td align="center" style="text-align: center; vertical-align: middle;">0.8835</td>
            <td align="center" style="text-align: center; vertical-align: middle;"><strong>0.9423</strong></td>
            <td align="center" style="text-align: center; vertical-align: middle;">-</td>
        </tr>
        <tr>
            <td align="center" rowspan="2" style="text-align: center; vertical-align: middle;">CQ500</td>
            <td align="center" rowspan="2" style="text-align: center; vertical-align: middle;">50</td>
            <td align="center" style="height: 35px; text-align: center; vertical-align: middle;">-</td>
            <td align="center" style="text-align: center; vertical-align: middle;">0.9533</td>
            <td align="center" style="text-align: center; vertical-align: middle;"><strong>0.9735</strong></td>
            <td align="center" style="text-align: center; vertical-align: middle;">-</td>
        </tr>
        <tr>
            <td align="center" style="text-align: center; vertical-align: middle;">V</td>
            <td align="center" style="text-align: center; vertical-align: middle;">0.9536</td>
            <td align="center" style="text-align: center; vertical-align: middle;">0.9641</td>
            <td align="center" style="text-align: center; vertical-align: middle;">-</td>
        </tr>
        <tr>
            <td align="center" rowspan="2" style="text-align: center; vertical-align: middle;">IMPACT</td>
            <td align="center" rowspan="2" style="text-align: center; vertical-align: middle;">60</td>
            <td align="center" style="height: 35px; text-align: center; vertical-align: middle;">-</td>
            <td align="center" style="text-align: center; vertical-align: middle;">0.9683</td>
            <td align="center" style="text-align: center; vertical-align: middle;"><strong>0.9782</strong></td>
            <td align="center" style="text-align: center; vertical-align: middle;">-</td>
        </tr>
        <tr>
            <td align="center" style="text-align: center; vertical-align: middle;">V</td>
            <td align="center" style="text-align: center; vertical-align: middle;">0.9018</td>
            <td align="center" style="text-align: center; vertical-align: middle;">0.9645</td>
            <td align="center" style="text-align: center; vertical-align: middle;">-</td>
        </tr>
        <tr style="background-color: #f9f9f9;">
            <td align="center" rowspan="2" style="font-weight: bold; text-align: center; vertical-align: middle;">Papers</td>
            <td align="center" style="text-align: center; vertical-align: middle;">Dot et al. (2022)</td>
            <td align="center" style="text-align: center; vertical-align: middle;">453</td>
            <td align="center" style="text-align: center; vertical-align: middle;">-</td>
            <td align="center" style="text-align: center; vertical-align: middle;">-</td>
            <td align="center" style="text-align: center; vertical-align: middle;">0.9622</td>
            <td align="center" style="text-align: center; vertical-align: middle;">-</td>
        </tr>
        <tr style="background-color: #f9f9f9;">
            <td align="center" style="text-align: center; vertical-align: middle;">Steybe et al. (2022)</td>
            <td align="center" style="text-align: center; vertical-align: middle;">20</td>
            <td align="center" style="text-align: center; vertical-align: middle;">별도 전처리 적용</td>
            <td align="center" style="text-align: center; vertical-align: middle;">-</td>
            <td align="center" style="text-align: center; vertical-align: middle;">-</td>
            <td align="center" style="text-align: center; vertical-align: middle;">0.94</td>
        </tr>
    </tbody>
</table>

<br>

```
2D U-Net보다 3D nnU-Net V2 모델의 성능이 모두 더 좋음.
Orientation / Voxel Resampling 전처리의 효과는 Thickness 등 이미 일관적으로 정리된 Open DB에는 악영향을, 정리되지 않은 Private DB엔 호영향을 끼침.

HU cliping, Normalization 등을 포함한 모든 전처리를 하지 않은 데이터셋부터 nnU-Net V2로 성능 다시 확인하고
전처리 하나씩 대입해보며 성능 나아지는 전처리 선정 예정
```

<br>

- Pure DB Performance Table

| Dataset | Subjects | nnUNet v2 |
| --- | --- | --- |
| 삼성서울 | 109 | 0.8151 |
| CQ500 | 50 | 0.9761 |
| IMPACT | 60 | 0.9876 |

```
앞으로 실험할 전처리, Method, Model과 비교하는 데에 Baseline 성능이 될 것임.
```

<br>

- Intra / Cross Dataset Test Performance Table

<table border="1" style="border-collapse: collapse; width: 100%; table-layout: fixed;">
    <thead>
        <tr>
            <th style="padding: 10px; text-align: center; vertical-align: middle;"></th>
            <th style="padding: 10px; text-align: center; vertical-align: middle;">Train Dataset</th>
            <th style="padding: 10px; text-align: center; vertical-align: middle;">Test Dataset</th>
            <th style="padding: 10px; text-align: center; vertical-align: middle;">nnU-Net V2</th>
        </tr>
    </thead>
    <tbody style="text-align: center; vertical-align: middle;">
        <tr>
            <td rowspan="9" style="font-weight: bold; text-align: center; vertical-align: middle;">Experiments</td>
            <td rowspan="3" style="text-align: center; vertical-align: middle;">삼성서울</td>
            <td style="padding: 10px; text-align: center; vertical-align: middle;">삼성서울</td>
            <td style="padding: 10px; text-align: center; vertical-align: middle;">0.9529</td>
        </tr>
        <tr>
            <td style="padding: 10px; text-align: center; vertical-align: middle; background-color: #f2f2f2;">CQ500</td>
            <td style="padding: 10px; text-align: center; vertical-align: middle; background-color: #f2f2f2;">0.9290</td>
        </tr>
        <tr>
            <td style="padding: 10px; text-align: center; vertical-align: middle; background-color: #f2f2f2;">IMPACT</td>
            <td style="padding: 10px; text-align: center; vertical-align: middle; background-color: #f2f2f2;">0.7700</td>
        </tr>
        <tr>
            <td rowspan="3" style="text-align: center; vertical-align: middle;">CQ500</td>
            <td style="padding: 10px; text-align: center; vertical-align: middle; background-color: #f2f2f2;">삼성서울</td>
            <td style="padding: 10px; text-align: center; vertical-align: middle; background-color: #f2f2f2;">0.7508</td>
        </tr>
        <tr>
            <td style="padding: 10px; text-align: center; vertical-align: middle;">CQ500</td>
            <td style="padding: 10px; text-align: center; vertical-align: middle;">0.9721</td>
        </tr>
        <tr>
            <td style="padding: 10px; text-align: center; vertical-align: middle; background-color: #f2f2f2;">IMPACT</td>
            <td style="padding: 10px; text-align: center; vertical-align: middle; background-color: #f2f2f2;">0.7396</td>
        </tr>
        <tr>
            <td rowspan="3" style="text-align: center; vertical-align: middle;">IMPACT</td>
            <td style="padding: 10px; text-align: center; vertical-align: middle; background-color: #f2f2f2;">삼성서울</td>
            <td style="padding: 10px; text-align: center; vertical-align: middle; background-color: #f2f2f2;">0.9141</td>
        </tr>
        <tr>
            <td style="padding: 10px; text-align: center; vertical-align: middle; background-color: #f2f2f2;">CQ500</td>
            <td style="padding: 10px; text-align: center; vertical-align: middle; background-color: #f2f2f2;">0.9383</td>
        </tr>
        <tr>
            <td style="padding: 10px; text-align: center; vertical-align: middle;">IMPACT</td>
            <td style="padding: 10px; text-align: center; vertical-align: middle;">0.9848</td>
        </tr>
    </tbody>
</table>

```
- 분석
데이터셋 간의 FOV(Field of View) 차이
학습 데이터셋의 FOV가 Inference FOV보다 넓으면 성능이 0.92 이상으로 나옴.
반대일 경우 0.7 부근으로 성능 하락함.

- 대응
ROI Crop 전처리를 통해 데이터 통일
FOV 넓은 데이터셋으로만 학습
Pre-training 기법 활용
범용성 넓은 Transformer-based 모델 활용
```

<br>

- Ordered Data Inference

| Train Dataset | DICE |
| --- | --- |
|삼성서울 | 0.9330 |
| CQ500 | 0.9293 |
| IMPACT | 0.9367 |

- Visualization

<img width="778" height="465" alt="image" src="https://github.com/user-attachments/assets/d652861d-29d1-4a1d-bcd8-34dbf7e2cd48" />

```
학습 데이터셋들이 Ordered Data보다 모두 FOV가 넓었기에 Inference 결과가 좋았음.
```

<br>

- DB 간 FOV 시각화

<img width="626" height="596" alt="image" src="https://github.com/user-attachments/assets/8e9bacb1-e5df-4e9c-b417-aafcc8b66da9" />

<br>

---

<br>

- 최종 결과 정리

<table>
  <thead>
    <tr>
      <th rowspan="2" align="center"></th>
      <th rowspan="2" align="center">Method</th>
      <th rowspan="2" align="center">Training Dataset</th>
      <th rowspan="2" align="center">Subjects</th>
      <th rowspan="2" align="center">Validation</th>
      <th colspan="3" align="center">Test</th>
      <th rowspan="2" align="center">Test</th>
    </tr>
    <tr>
      <th align="center">Combined Dataset 1</th>
      <th align="center">Added Samsung</th>
      <th align="center">Ordered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6" align="center">Experiments</td>
      <td rowspan="5" align="center">Head Crop</td>
      <td align="center">CQ500</td>
      <td align="center">45 / 5</td>
      <td align="center"><u>0.9761</u></td>
      <td align="center">0.9525</td>
      <td align="center">0.9154</td>
      <td align="center">0.9171</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center">IMPACT</td>
      <td align="center">54 / 6</td>
      <td align="center"><b>0.9876</b></td>
      <td align="center">0.9532</td>
      <td align="center">0.9328</td>
      <td align="center">0.9328</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center">Samsung</td>
      <td align="center">99 / 10</td>
      <td align="center">0.8151</td>
      <td align="center">0.9065</td>
      <td align="center"><u>0.9678</u></td>
      <td align="center"><b>0.9619</b></td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center">Combined Dataset 1</td>
      <td align="center">132 / 15</td>
      <td align="center">0.9641</td>
      <td align="center"><b>0.9748</b></td>
      <td align="center">0.9571</td>
      <td align="center">0.9432</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center">Combined Dataset 2</td>
      <td align="center">350 / 27</td>
      <td align="center">0.9688</td>
      <td align="center">0.9736</td>
      <td align="center"><b>0.9729</b></td>
      <td align="center"><u>0.9599</u></td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center">Pre-training</td>
      <td align="center">Combined Dataset 1</td>
      <td align="center">147</td>
      <td align="center">0.9756</td>
      <td align="center"><u>0.9744</u></td>
      <td align="center">0.9634</td>
      <td align="center">0.9550</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td rowspan="2" align="center">Paper</td>
      <td align="center">Dot et al. (2022)</td>
      <td align="center">Private</td>
      <td align="center">300 / 153</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">0.9622</td>
    </tr>
    <tr>
      <td align="center">Steybe et al. (2022)</td>
      <td align="center">Private</td>
      <td align="center">15 / 5</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">0.94</td>
    </tr>
  </tbody>
</table>

```
Combined Dataset2로 학습한 모델이 대체적으로 높은 DICE Score를 유지할 뿐 아니라 Generalizaibility 측면에서도 우수한 것으로 분석되어 해당 모델을 서비스화 함.
```

<br>

- Visualization

<img width="815" height="411" alt="image" src="https://github.com/user-attachments/assets/4d88bdbe-6382-409d-8c7f-48e8d732087e" />

<br>

---

<br>

## Final Skull Auto Segmentation Pipeline

<img width="884" height="401" alt="image" src="https://github.com/user-attachments/assets/a6c5a98d-37ad-4d59-a9a6-e6ed9ad510ae" />

<br>

- Post-Processing Before/After Visualization

<img width="847" height="286" alt="image" src="https://github.com/user-attachments/assets/185628b6-9a0a-4162-aa9f-ccdcd55b9734" />
