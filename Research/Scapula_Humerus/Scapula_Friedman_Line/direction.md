# CT → Mesh → Landmark 좌표계 변환 흐름 정리

> DICOM 입력부터 STL mesh와 landmark 좌표가 최종적으로 **LPS 좌표계**로 확정되기까지,
> 각 코드 파일과 함수가 좌표계를 **어디서 어떻게 건드리는지** 실행 순서대로 정리한 문서.

---

## 0. 전체 흐름 요약

```
DICOM (patient/LPS + Direction matrix)
  │
  │  [run_pipeline.py] to_identity_direction()   ← Direction 제거 (D=I 격자 재샘플)
  │  [run_pipeline.py] resample_z()              ← z-spacing 1mm 통일
  ▼
CT image + segmentation mask  (동일 격자, D=I, 1mm)
  │
  ├─────────────────── [STL 경로] ───────────────────┐
  │  nifti_to_stl_ignore_direction()                 │
  │    · marching cubes → voxel index (ZYX)          │
  │    · index → mm (ZYX→XYZ 재배열, spacing 곱)      │
  │    · Direction 무시 (대개) → axis-aligned mm      │
  │  restore_direction_on_stl()  → LPS 확정 ★         │
  │                                                   ▼
  │                                          scapula.stl / humerus.stl (LPS)
  │
  └────────────────── [Landmark 경로] ───────────────┐
     _crop_*()  affine translation만 갱신 (world 보존) │
     _landmark_core: L/R 분리 (nibabel RAS 기준)       │
     EPFL worker → PKL (EPFL 내부 frame)               │
     normalize_to_lps() → LPS 확정 ★                   ▼
                                          landmarks (LPS)
```

**핵심**: 서로 다른 두 경로(SimpleITK 기반 STL / nibabel 기반 landmark)가
마지막에 모두 **LPS**로 수렴한다. `verify_against_stl()`이 두 경로의 정합성을 검증한다.

---

## 1. `run_pipeline.py` — 입력 CT 좌표계 정규화 (출발점)

좌표계 처리의 시작이자 가장 중요한 정규화 단계. 이후 모든 산출물이 여기서 만든
공통 격자를 기준으로 한다.

### 1-1. `to_identity_direction()` — Direction matrix 제거

```python
ct_image = to_identity_direction(ct_image, sitk.sitkLinear)
```

- CT의 Direction matrix에 off-diagonal 성분(gantry tilt 등)이 있으면,
  **world 좌표(mm)는 유지한 채 Direction = 단위행렬(D=I)인 격자로 재샘플**.
- off-diagonal이 없으면 **no-op**(원본 그대로 반환).
- 목적: "tilt 분기 근본 제거" — 이후 단계가 Direction을 신경 쓰지 않도록
  tilt를 격자에 미리 구워 넣는다.

### 1-2. `resample_z()` — z 방향 해상도 통일

```python
ct_image = inference.resample_z(ct_image, target_z_spacing=1.0)
```

- z-spacing > 1mm면 계단(stair) artifact 방지를 위해 z축 재샘플.
- 방향 변환이 아니라 **격자 해상도 조정**이지만, 이후 mask/STL의 물리 좌표에 직접 영향.

> **이 시점 결과**: CT와 mask가 **동일한 identity-direction 격자 + 1mm z-spacing**을 공유.
> STL 경로와 landmark 경로 모두 이 공통 좌표계를 기준으로 분기한다.

---

## 2. `nifti_to_stl_ignore_direction.py` — voxel index → 물리공간(mm)

mask NIFTI를 3D mesh로 변환하며 voxel 인덱스를 물리 좌표로 바꾸는 단계.
함수 이름 그대로 **Direction을 무시**하는 것이 기본 방침(Mimics 방식).

### 2-1. Reference 헤더 적용 — Origin/Spacing만

```python
image.SetOrigin(ref.GetOrigin())
image.SetSpacing(ref.GetSpacing())
```

- reference CT의 **Origin·Spacing만** 강제 적용, **Direction은 의도적으로 제외**.

### 2-2. Marching Cubes — voxel index 좌표 생성

```python
verts, faces, _, _ = measure.marching_cubes(..., spacing=(1.0, 1.0, 1.0), ...)
```

- `spacing=(1,1,1)`로 호출 → `verts` 단위 = **(리샘플된) voxel index**.
- 축 순서: `verts[:,0]=Z_idx`, `verts[:,1]=Y_idx`, `verts[:,2]=X_idx`.

### 2-3. index → mm 변환 (§4 좌표 변환 블록)

```python
index_xyz = np.stack([verts[:, 2], verts[:, 1], verts[:, 0]], axis=1)  # ZYX → XYZ
scaled    = index_xyz * effective_spacing_xyz                          # voxel → mm
```

- marching cubes의 **(Z,Y,X) → (X,Y,Z) 재배열**.
- `effective_spacing_xyz`를 곱해 mm로 스케일.
  - isotropic 경로: zoom 반올림 오차를 보정한 effective spacing.
  - thin 경로: 원본 spacing 그대로.

### 2-4. Direction 적용 분기 — 진짜 tilt만 감지

```python
off_diagonal_max = np.abs(direction[off_diagonal_mask]).max()
is_real_tilt = off_diagonal_max > 0.005

if is_real_tilt:
    verts_physical = np.array(origin) + (direction @ scaled.T).T
else:
    verts_physical = np.array(origin) + scaled
```

- off-diagonal > 0.005인 **"진짜 gantry tilt"일 때만** Direction 적용.
- `diag(±1,±1,±1)` 형태(단순 축 부호/정렬)는 무시.
- **주의**: §1에서 이미 D=I로 재샘플했으므로, 정상 흐름에서는 `is_real_tilt = False`가
  되어 `origin + scaled` 단순 경로를 탄다. 이 tilt 처리는 사실상 **§1을 거치지 않고
  이 함수를 직접 호출할 때의 방어 장치**에 가깝다.

### 2-5. Reflection 보정 — face winding

```python
if np.linalg.det(direction) < 0:
    faces = faces[:, ::-1]
```

- Direction 행렬식이 음수(reflection 포함)면 face winding을 뒤집어 **법선 방향 보존**.
- tilt 여부와 무관하게 적용.

> **이 시점 결과**: `verts_physical`은 origin 기준 **axis-aligned 물리공간(mm)** 좌표.
> 아직 "LPS 확정"은 아니며, Direction을 무시한 상태.

---

## 3. `restore_direction_on_stl()` — STL을 LPS로 확정 ★

`run_pipeline.py`에서 STL 생성 직후 호출 (구현은 `landmark_align.py`에 위치).

```python
if stl_path.exists():
    restore_direction_on_stl(stl_path, mask_nifti_path, ct_nifti_path)
    created_files.append({ ..., "coordinate_system": "LPS" })
```

- §2에서 Direction을 무시하고 만든 mesh에 **NIFTI 헤더의 방향 정보를 다시 입혀
  LPS 좌표계로 정렬**하는 역할.
- 이 호출 직후 STL은 **LPS로 선언**된다(`"coordinate_system": "LPS"`).

> ⚠️ **미첨부**: 이 함수의 실제 구현은 `landmark_align.py`에 있어 세부 로직은 확인 필요.
> STL 경로의 "LPS 확정" 핵심 로직이 이 파일에 있음.

---

## 4. Landmark 경로 — 별도 격자에서 계산 후 LPS로 되돌림

Landmark는 STL과 **다른 라이브러리(nibabel)·다른 좌표 규약**으로 처리된다.

### 4-1. ROI crop — `_crop_ct_and_mask_by_mask()` (run_pipeline) / `_crop_to_scapula_roi()` (_landmark_core)

```python
new_affine[:3, 3] = affine[:3, :3] @ np.array([sx, sy, sz]) + affine[:3, 3]
```

- crop 시작 voxel에 맞춰 **affine의 translation만 재계산**.
- **world 좌표(mm)는 원본과 동일하게 보존** → crop된 볼륨에서 나온 landmark도
  원본 좌표계에서 그대로 유효. 좌표계를 평행이동시키지 않는 설계.

### 4-2. L/R 분리 — `_landmark_core.py` (nibabel RAS 기준)

```python
def _world_x_centroid(bool_mask, affine):
    return float(affine[0, :3] @ c_vox + affine[0, 3])
...
side = "right" if x_hum > x_scap else "left"   # RAS +x = Right
```

- **좌표 규약 전환 지점**: `_landmark_core`는 **nibabel**로 로드 → affine이 **RAS** 규약.
  (STL 경로는 SimpleITK 기반 LPS).
- 즉 landmark 내부 계산은 **RAS 좌표**에서 이루어진다 (`RAS +x = Right`).

### 4-3. EPFL worker 실행 → PKL 저장

- `_stage_files()`가 EPFL 규약대로 파일 배치 (`{pid}_0000.nii.gz`, `{pid}R/L.nii.gz` 등).
- `measureScase()` / `predictLandmarksSCases()`가 landmark 계산 →
  `scapulaLandmarksAuto{R/L}.pkl`, `glenoidLandmarksAuto{R/L}.pkl` 저장.
- `_parse_landmark_pkls()`가 GL1~GL4, glenoid_center, trigonum_spinae, friedman line 회수.
- 이 PKL 좌표는 **EPFL 내부 규약 좌표**(스테이징에 쓴 affine 기준).

### 4-4. `normalize_to_lps()` — 최종 LPS 변환 ★ (run_pipeline)

```python
frame = detect_landmark_frame(raw_landmarks, mask_for_landmark, label_value=1)
landmarks_lps = normalize_to_lps(raw_landmarks, frame)
...
verify = verify_against_stl(landmarks_lps, scap_stl)
```

- `detect_landmark_frame()`으로 raw landmark의 **소스 좌표계(frame)를 탐지**.
- `normalize_to_lps()`로 **LPS로 변환** → STL과 좌표계 통일.
- `verify_against_stl()`로 scapula STL과 **대조 검증**.

> ⚠️ **미첨부**: `detect_landmark_frame`, `normalize_to_lps`, `verify_against_stl`,
> `check_plausibility`, `to_stl_frame`, `get_stl_frame` 모두 `landmark_align.py`에 위치.

---

## 5. 독립 검증 도구 (파이프라인 본류 아님)

### 5-1. Friedman line 추출 스크립트 (`get_glenoid_center_geometric` 등)

```python
if side == "right":
    idx_medial = np.argmax(verts[:, 0])   # LPS에서 X 큰 쪽이 medial
else:
    idx_medial = np.argmin(verts[:, 0])
```

- STL만으로 geometric heuristic 기반 landmark를 **추정**하는 오프라인 검증 도구.
- 입력 STL이 §3에서 **이미 LPS로 확정된 상태**임을 전제하고, 그 좌표계에서
  medial 방향을 판단.
- 좌표계를 변환하지 않고 **LPS 가정 하에 소비**만 함. PKL(official)과 diff(mm) 출력.

### 5-2. `epfl_landmark_worker.py`

- 좌표 변환 로직 없음. `_landmark_core.predict_landmarks_from_mask()`를
  subprocess에서 호출하는 **얇은 CLI 래퍼**.

---

## 6. 좌표계 변환 지점 요약표

| 순서 | 코드 / 함수 | 좌표계 동작 | 결과 좌표계 |
|:---:|---|---|---|
| 입력 | `run_pipeline` (SimpleITK 로드) | DICOM 로드 | patient/LPS + Direction |
| 1-1 | `to_identity_direction()` | Direction=I 재샘플, world 유지 | axis-aligned (D=I) |
| 1-2 | `resample_z()` | z-spacing 1mm 통일 | 동일 격자 |
| 4-1 | `_crop_ct_and_mask_by_mask()` / `_crop_to_scapula_roi()` | affine translation만 갱신 | 불변 (world 보존) |
| 2-3 | `nifti_to_stl_ignore_direction()` §4 | voxel index → mm, ZYX→XYZ | axis-aligned mm |
| 2-4 | 동 §4 tilt 분기 | 진짜 tilt만 Direction 적용 | axis-aligned mm (대개) |
| 2-5 | 동 §4 reflection | det<0 시 face winding 반전 | 법선 보존 |
| **3** | **`restore_direction_on_stl()`** | NIFTI 방향 재적용 | **LPS** ★ |
| 4-2 | `_landmark_core` L/R 분리 | nibabel affine 기준 판정 | RAS 내부 |
| 4-3 | EPFL worker → PKL | EPFL 규약 좌표 계산 | EPFL frame |
| **4-4** | **`normalize_to_lps()`** | frame 탐지 후 변환 | **LPS** ★ |

---

## 7. 확인·주의 사항

1. **좌표계 확정 핵심 로직이 미첨부 파일에 있음**
   `restore_direction_on_stl`, `normalize_to_lps`, `detect_landmark_frame` 등
   "LPS로 어떻게 정확히 변환되는가"의 세부는 **`landmark_align.py`**에 있다.
   정확한 문서화를 위해서는 이 파일 확인이 필요.

2. **두 경로의 라이브러리·규약이 다름**
   - STL 경로: **SimpleITK → LPS**
   - Landmark 경로: **nibabel → RAS 내부 → LPS**
   서로 다른 규약을 거쳐 마지막에 LPS로 만나므로, **두 경로의 LPS 정의가
   완전히 일치하는지**가 검증의 핵심. `verify_against_stl()`이 그 정합성 확인 장치.

3. **STL 단계의 tilt 처리는 방어 장치**
   §1에서 이미 D=I로 정규화하므로 정상 흐름에서는 §2-4의 tilt 분기가 거의
   발동하지 않는다. `nifti_to_stl_ignore_direction()`을 파이프라인 밖에서 단독
   호출할 때를 대비한 안전장치로 이해할 것.
