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
  ├─────────────────── [STL 경로] ─────────────────────┐
  │  nifti_to_stl_ignore_direction()                   │
  │    · marching cubes → voxel index (ZYX)            │
  │    · index → mm (ZYX→XYZ 재배열, spacing 곱)        │
  │    · Direction 무시(기본) → axis-aligned mm         │
  │  restore_direction_on_stl()  → LPS 확정 ★           │
  │    · p_true = O + D·(p_stl − O)                     ▼
  │                                          scapula.stl / humerus.stl (LPS)
  │
  └────────────────── [Landmark 경로] ─────────────────┐
     _crop_*()  affine translation만 갱신 (world 보존)   │
     _landmark_core: L/R 분리 (nibabel RAS 기준)         │
     EPFL worker → PKL (RAS or LPS, 미확정)              │
     detect_landmark_frame()  → 실측으로 frame 판별       │
     normalize_to_lps()       → LPS 확정 ★               │
     verify_against_stl()     → STL과 정합성 검증         ▼
                                          landmarks (LPS)
```

**핵심 원리**

- STL 경로(SimpleITK/LPS)와 landmark 경로(nibabel/RAS)가 서로 다른 규약을 거쳐
  마지막에 모두 **LPS**로 수렴한다.
- 두 경로가 만나는 좌표계가 일치하는지를 `verify_against_stl()`이 실제 mesh 거리로 검증한다.

**좌표계 변환의 수학적 정의** (`landmark_align.py` 기준)

```
f_stl(i)  = O + S·i        ← Direction 무시하고 만든 STL (S: spacing 대각행렬)
f_true(i) = O + D·S·i       ← 진짜 LPS (D: direction matrix)

⇒ STL → LPS :  p_true = O + D·(p_stl − O)          # restore_direction_on_stl
⇒ LPS → STL :  p_stl  = O + D⁻¹·(p_true − O)        # world_lps_to_stl_frame
⇒ RAS ↔ LPS :  p_lps  = p_ras · [-1, -1, 1]         # RAS2LPS (X·Y 부호 반전)
```

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
- 목적: "tilt 분기 근본 제거" — tilt를 격자에 미리 구워 넣어 이후 단계가
  Direction을 신경 쓰지 않도록 한다.

### 1-2. `resample_z()` — z 방향 해상도 통일

```python
ct_image = inference.resample_z(ct_image, target_z_spacing=1.0)
```

- z-spacing > 1mm면 계단(stair) artifact 방지를 위해 z축 재샘플.
- 방향 변환이 아니라 **격자 해상도 조정**이지만, 이후 mask/STL의 물리 좌표에 직접 영향.

> **이 시점 결과**: CT·mask가 **identity-direction 격자(D=I) + 1mm z-spacing**을 공유.
> D=I이므로 축이 이미 LPS와 정렬된 axis-aligned 상태다. STL·landmark 두 경로가
> 여기서 분기한다.

---

## 2. `nifti_to_stl_ignore_direction.py` — voxel index → 물리공간(mm)

mask NIFTI를 3D mesh로 변환하며 voxel 인덱스를 물리 좌표로 바꾸는 단계.
함수 이름대로 **Direction 무시**가 기본 방침(Mimics 방식).

### 2-1. Reference 헤더 적용 — Origin/Spacing만

```python
image.SetOrigin(ref.GetOrigin())
image.SetSpacing(ref.GetSpacing())
```

- reference CT의 **Origin·Spacing만** 강제 적용, **Direction은 의도적으로 제외**.
- ⚠️ 이 두 줄은 §3의 `get_stl_frame()`과 **완전히 동일**해야 프레임 복원이 성립한다.

### 2-2. Marching Cubes — voxel index 좌표 생성

```python
verts, faces, _, _ = measure.marching_cubes(..., spacing=(1.0, 1.0, 1.0), ...)
```

- `spacing=(1,1,1)` 호출 → `verts` 단위 = **(리샘플된) voxel index**.
- 축 순서: `verts[:,0]=Z_idx`, `verts[:,1]=Y_idx`, `verts[:,2]=X_idx`.

### 2-3. index → mm 변환 (§4 좌표 변환 블록)

```python
index_xyz = np.stack([verts[:, 2], verts[:, 1], verts[:, 0]], axis=1)  # ZYX → XYZ
scaled    = index_xyz * effective_spacing_xyz                          # voxel → mm
```

- marching cubes의 **(Z,Y,X) → (X,Y,Z) 재배열** 후 spacing을 곱해 mm 스케일.
- isotropic 경로는 zoom 반올림 보정 spacing, thin 경로는 원본 spacing 사용.

### 2-4. Direction 적용 분기 — 진짜 tilt만 감지

```python
is_real_tilt = np.abs(direction[off_diagonal_mask]).max() > 0.005
if is_real_tilt:
    verts_physical = np.array(origin) + (direction @ scaled.T).T   # f_true = O + D·S·i
else:
    verts_physical = np.array(origin) + scaled                     # f_stl  = O + S·i
```

- off-diagonal > 0.005인 **"진짜 gantry tilt"일 때만** Direction 적용.
  `diag(±1,±1,±1)` 형태(단순 축 부호/정렬)는 무시.
- **정상 흐름에서는** §1에서 D=I로 정규화했으므로 `is_real_tilt = False` →
  `origin + scaled`(Direction 무시) 경로를 탄다. tilt 분기는 이 함수를
  **파이프라인 밖에서 단독 호출**할 때를 위한 방어 장치.

### 2-5. Reflection 보정 — face winding

```python
if np.linalg.det(direction) < 0:
    faces = faces[:, ::-1]
```

- Direction 행렬식이 음수(reflection 포함)면 face winding 반전 → **법선 방향 보존**.

> **이 시점 결과**: `verts_physical`은 origin 기준 **axis-aligned 물리공간(mm)** 좌표.
> Direction을 무시한 `f_stl` 상태이며, 아직 "LPS 확정"은 아니다.

---

## 3. `landmark_align.py :: restore_direction_on_stl()` — STL을 LPS로 확정 ★

`run_pipeline.py`가 STL 생성 직후 호출. §2에서 버린 Direction을 되살려 STL을 진짜 LPS로 만든다.

```python
o, d, tilt = get_stl_frame(nifti_path, reference_path)
if tilt:                                  # is_real_tilt=True → §2-4에서 이미 D 적용
    return False                          #   → 이미 LPS, skip
mesh = trimesh.load(str(stl_path), process=False)
v = np.asarray(mesh.vertices, float)
mesh.vertices = o + (d @ (v - o).T).T     # p_true = O + D·(p_stl − O)
if np.linalg.det(d) < 0:
    mesh.faces = np.asarray(mesh.faces)[:, ::-1]   # det(D)<0 → winding flip
```

### 3-1. `get_stl_frame()` — STL 프레임 파라미터 복원

- §2-1과 **동일하게** reference의 Origin/Spacing을 적용한 뒤 `(origin, direction, is_real_tilt)` 반환.
- off-diagonal > `tilt_eps(0.005)` 이면 `is_real_tilt=True` (§2-4와 동일 기준).

### 3-2. 변환 로직

- `is_real_tilt=True`: §2-4에서 STL이 이미 `O + D·S·i`로 만들어졌으므로 **항등 → skip**.
- `is_real_tilt=False`: STL이 `f_stl = O + S·i` 상태 → `p_true = O + D·(p_stl − O)`로 되살림.
- `det(D) < 0`이면 face winding도 뒤집어 법선 보존.

> **정상 흐름에서의 실제 동작**: §1에서 D=I로 정규화 → 여기 들어오는 direction ≈ 단위행렬 →
> `p_true = O + I·(p_stl − O) = p_stl`, 즉 **좌표는 사실상 불변**이고 "LPS"라는 라벨만 확정된다.
> 실질적 회전 복원은 이 함수를 **direction이 살아있는 데이터에 단독 적용**할 때 의미가 있다.
> 호출 직후 `run_pipeline`은 `"coordinate_system": "LPS"`로 기록한다.

---

## 4. Landmark 경로 — 별도 격자에서 계산 후 LPS로 되돌림

Landmark는 STL과 **다른 라이브러리(nibabel)·다른 좌표 규약(RAS)** 으로 처리된다.

### 4-1. ROI crop — world 좌표 보존

`_crop_ct_and_mask_by_mask()`(run_pipeline) / `_crop_to_scapula_roi()`(_landmark_core):

```python
new_affine[:3, 3] = affine[:3, :3] @ np.array([sx, sy, sz]) + affine[:3, 3]
```

- crop 시작 voxel에 맞춰 **affine translation만 재계산** → **world 좌표(mm)는 원본과 동일**.
- 좌표계를 평행이동시키지 않으므로, crop 볼륨에서 나온 landmark도 원본 좌표계에서 그대로 유효.

### 4-2. L/R 분리 — `_landmark_core.py` (nibabel RAS 기준)

```python
def _world_x_centroid(bool_mask, affine):
    return float(affine[0, :3] @ c_vox + affine[0, 3])
...
side = "right" if x_hum > x_scap else "left"   # RAS +x = Right
```

- **좌표 규약 전환 지점**: `_landmark_core`는 **nibabel** 로드 → affine이 **RAS** 규약.
  (STL 경로는 SimpleITK 기반 LPS.) 이후 landmark 계산은 RAS 좌표에서 진행된다.

### 4-3. EPFL worker 실행 → PKL 저장

- `_stage_files()`가 EPFL 규약대로 파일 배치, `measureScase()` / `predictLandmarksSCases()`가
  landmark 계산 → `scapulaLandmarksAuto{R/L}.pkl`, `glenoidLandmarksAuto{R/L}.pkl` 저장.
- `_parse_landmark_pkls()`가 GL1~GL4, glenoid_center, trigonum_spinae, friedman line 회수.
- 이 PKL 좌표가 **RAS인지 LPS인지는 이 시점에 확정되지 않음** → 다음 단계에서 실측 판별.

### 4-4. `detect_landmark_frame()` — RAS vs LPS 실측 판별

```python
inv  = np.linalg.inv(img.affine)                       # nibabel affine = RAS
dist = distance_transform_edt(~binm, sampling=sp)      # 뼈 마스크까지의 거리장
for name, sgn in [("RAS", np.ones(3)), ("LPS", RAS2LPS)]:
    vi = np.round((inv @ np.append(p * sgn, 1))[:3])   # 후보 부호로 voxel 투영
    # in-bounds 개수 + 뼈까지 중앙 거리로 점수화
best = max(res, key=lambda k: (res[k][0], -res[k][1])) # 뼈에 더 가까운 프레임 선택
```

- landmark 점들을 **RAS 가정**과 **LPS 가정** 두 부호로 각각 voxel에 투영.
- 실제 scapula 마스크(뼈)에 **더 가깝게 떨어지는 쪽**을 참 프레임으로 판정.
- 반환: `"RAS" | "LPS" | None`.

### 4-5. `normalize_to_lps()` — LPS 확정 ★

```python
RAS2LPS = np.array([-1.0, -1.0, 1.0])   # 자기역원 (X·Y 부호 반전)
def normalize_to_lps(landmarks, frame):
    if frame != "RAS":
        return copy.deepcopy(landmarks)   # 이미 LPS면 그대로
    f = lambda p: p * RAS2LPS             # 점·벡터 동일 (RAS↔LPS는 선형 부호반전)
    return _map_points(..., f, f)
```

- frame이 RAS면 `RAS2LPS`로 부호 반전, LPS면 변환 없음 → **모두 LPS로 통일**.
- 이후 `run_pipeline`은 `landmarks_lps`를 그대로 결과에 담고 `verify_against_stl()`로 검증.

### 4-6. 점 / 벡터 구분 처리 (좌표 매핑 공통 규칙)

`_map_points()` / `_iter_points()`는 길이-3 좌표에만 변환을 적용하되, **키 이름으로 점과 벡터를 구분**한다.

- 벡터 키 힌트: `"direction", "normal", "axis", "vector", "versor", "_dir"`
  (예: `friedman_direction`).
- **점**: 평행이동 포함 변환 (`O + D⁻¹·(p−O)` 등).
- **벡터**: 회전만 적용, 평행이동 금지 (뼈 위에 있을 이유가 없으므로).
- RAS↔LPS 부호반전은 선형이라 점·벡터가 동일한 `f`를 쓴다.

---

## 5. 좌표계 관련 검증 함수 (변환 아님)

### 5-1. `verify_against_stl()` — landmark ↔ STL 정합성

```python
tree = cKDTree(np.asarray(mesh.vertices))
d = float(tree.query(np.asarray(v, float))[0])   # landmark → 최근접 mesh 정점 거리
if d > tol_mm(25.0): ok = False                  # 벡터 키는 skip
```

- LPS로 통일된 landmark가 **LPS STL 표면에 실제로 얹히는지**를 mesh 거리로 확인.
- 두 경로(STL/landmark)의 LPS 정의가 일치하는지 검증하는 최종 관문.

### 5-2. `check_plausibility()` — 해부학적 타당성 (mesh 독립)

```python
lat = "left" if gc[0] > ts[0] else "right"   # LPS: +X = Left
```

- Friedman 길이(85~125mm), glenoid rim SI/AP·직교각·대칭성, 좌우 판정 등을 검사.
- LPS 규약(`+X = Left`)을 전제로 좌우를 추론해 입력 `side`와 대조. 좌표 변환은 하지 않음.

---

## 6. 사용/미사용 경로 구분 (혼동 주의)

`landmark_align.py`는 STL과 landmark를 맞추는 **두 가지 전략**을 담고 있으나, 파이프라인은 하나만 쓴다.

| 전략 | 방법 | 파이프라인 사용 |
|---|---|---|
| **A (사용)** | `restore_direction_on_stl()`로 **STL을 LPS로** 올리고, landmark는 LPS 유지 → 직접 비교 | ✅ run_pipeline |
| B (미사용) | STL은 그대로 두고 `to_stl_frame()`/`world_lps_to_stl_frame()`로 **landmark를 STL 프레임(D 무시)으로** 내림 | ❌ 대안 경로 |

- `run_pipeline`은 **전략 A**만 실행한다: `restore_direction_on_stl()` 호출 후
  `verify_against_stl(landmarks_lps, ...)`로 LPS끼리 비교.
- `to_stl_frame()`, `world_lps_to_stl_frame()`, `vec_lps_to_stl_frame()`는 전략 B용으로
  현재 파이프라인 경로에서는 호출되지 않는다.

---

## 7. 좌표계 변환 지점 요약표

| 순서 | 코드 / 함수 | 좌표계 동작 | 결과 좌표계 |
|:---:|---|---|---|
| 입력 | `run_pipeline` (SimpleITK 로드) | DICOM 로드 | patient/LPS + Direction |
| 1-1 | `to_identity_direction()` | Direction=I 재샘플, world 유지 | axis-aligned (D=I) |
| 1-2 | `resample_z()` | z-spacing 1mm 통일 | 동일 격자 |
| 4-1 | `_crop_ct_and_mask_by_mask()` / `_crop_to_scapula_roi()` | affine translation만 갱신 | 불변 (world 보존) |
| 2-3 | `nifti_to_stl_ignore_direction()` §4 | voxel index → mm, ZYX→XYZ | axis-aligned mm (`f_stl`) |
| 2-4 | 동 §4 tilt 분기 | 진짜 tilt만 `O+D·S·i` 적용 | axis-aligned mm (대개) |
| 2-5 | 동 §4 reflection | det(D)<0 시 winding 반전 | 법선 보존 |
| **3** | **`restore_direction_on_stl()`** | `p_true = O + D·(p_stl − O)` | **LPS** ★ |
| 4-2 | `_landmark_core` L/R 분리 | nibabel affine 기준 판정 | RAS 내부 |
| 4-3 | EPFL worker → PKL | landmark 계산 | RAS/LPS 미확정 |
| **4-4** | **`detect_landmark_frame()`** | 뼈 근접도로 frame 실측 | RAS or LPS 판별 |
| **4-5** | **`normalize_to_lps()`** | RAS면 `·[-1,-1,1]` | **LPS** ★ |
| 5-1 | `verify_against_stl()` | mesh 최근접 거리 검증 | (검증) |
| 5-2 | `check_plausibility()` | 해부학적 타당성 (LPS +X=Left) | (검증) |

---

## 8. 요점 정리

1. **모든 산출물은 LPS로 수렴한다.** STL은 §3(`restore_direction_on_stl`), landmark는
   §4-5(`normalize_to_lps`)에서 LPS로 확정된다.

2. **STL 경로와 landmark 경로의 좌표 규약이 다르다.**
   - STL: **SimpleITK → LPS**
   - Landmark: **nibabel → RAS 내부 → (frame 실측) → LPS**
   두 경로가 마지막에 LPS로 만나며, `verify_against_stl()`이 그 정합성을 mesh 거리로 확인한다.

3. **`get_stl_frame()`은 `nifti_to_stl_ignore_direction`의 §1·§4와 반드시 동일 로직이어야 한다.**
   reference Origin/Spacing 적용 방식과 `tilt_eps` 기준이 어긋나면 프레임 복원이 깨진다.

4. **정상 흐름에서 STL의 Direction 복원은 거의 항등이다.** §1에서 D=I로 정규화하므로
   `restore_direction_on_stl`은 좌표를 바꾸지 않고 "LPS 라벨"만 확정한다. 실질 회전 복원은
   direction이 살아있는 데이터에 단독 적용할 때 발동한다.

5. **점과 벡터를 구분해 다룬다.** 키 이름 힌트로 벡터(`*_direction` 등)를 식별해
   평행이동을 배제하고 회전만 적용한다.

6. **전략 A만 사용된다.** `to_stl_frame` 계열(전략 B)은 구현돼 있으나 현재 파이프라인에서는
   호출되지 않는 대안 경로다.
