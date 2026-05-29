# - 5/13 Stair Artifact Issue

```
Scapula, humerus Segmentation Model Stair Artifact Issue
```

## - file list

- /code/

```
dicom_sort.py
select_series.py
dicomtovtk.py
dicomtovtk__.py
__dicomtovtk.py
makevti_m.py
dicom_compression.py
dicom_image_Conversion.py
nifti_ti_stl_ignore_direction.py
Downlaod_stl_applied_trasnformer.py
train_multi.py
test.py
precess_medical_data.py
autoseg_pipeline.py
```

- /code/python

```
unet_model.py
dataloader.py
train_multi.py
test.py
run_pipeline.py
list_all_series.py
select_best_series.py
copy_selected_series.py
dicom_sort.py
select_series.py
nifti_to_stl_ignore_direction.py
process_medical_data.py
autoseg_pipeline.py
```

## 의존성 그래프 (inference)

```
[autoseg_pipeline].py                    [run_pipeline].py
        │                                       │
        ├─ dicom_sort.py                        │
        ├─ select_series.py                     │
        ├─ test.py   ◄──────────────────────────┤
        │    └─ unet_model.py                   │
        ├─ dicom_sort.py                        │
        ├─ nifti_to_stl_ignore_direction.py ◄───┘


[train_multi.py]
      ├─ dataloader.py
      └─ unet_model.py
```

<br>

## Solution

```
원인: 다양한 입력 CT 이미지에 대한 python 기반 framework의 대응 능력 부족
- Slice thickness resampling
        - resampling 발동 조건 (If문 수정)
- Image size
        - zero-padding 조건적으로 발동

```
      
