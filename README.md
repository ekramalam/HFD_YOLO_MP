## Algorithm: Proposed Human Fall-Detection Framework

**Input:** Video stream $V$  
**Output:** Fall / ADL decision for each detected person

### Step 1: Person Detection and ROI Extraction

For each input frame:

1. Apply **YOLO12n** to detect persons.
2. Retain detections belonging to the person class.
3. Discard detections whose confidence is below $T_{YlCS}=0.4$.
4. Add padding around the detected bounding box.
5. Crop the padded person region as the ROI.

### Step 2: Pose Estimation and Validation

6. Apply **MediaPipe Pose** to the person ROI.
7. Extract the 33 body landmarks.
8. Map the required landmarks back to the original image coordinates.
9. Validate the required landmarks (nose, shoulders, hips, knees, and ankles).
10. Retain landmarks only when their visibility is above the required threshold.
11. Smooth the shoulder midpoint temporally using exponential moving average.

### Step 3: Feature Extraction

12. Calculate the following spatial and temporal features:

   - Body angle $\theta$
   - Normalized vertical drop $\Delta y$
   - Height ratio $HR$
   - Angle change $\Delta\theta$
   - Vertical velocity $v_y$
   - Knee-ankle vertical distance $d_{ka}$
   - Knee-hip vertical distance $d_{kh}$
   - Normalized nose position $y_{nose}$
   - Bounding-box aspect ratio $AR$
   - Knee-ankle angle $\theta_{ka}$
   - Knee-hip angle $\theta_{kh}$

13. Maintain the temporal buffer required for velocity, displacement, and angle-change calculations.
14. Update the shoulder-to-hip baseline and apply spike filtering before calculating $HR$.

### Step 4: Early Override Filtering

For every valid person detection, evaluate the following overrides in sequence:

**HPO — Head Position Override**

```text
if y_nose < T_top:
    skip fall evaluation
```

**ARO — Aspect Ratio Override**

```text
if AR > T_AR:
    skip fall evaluation
```

**VLO — Vertical Leg Override**

```text
if θ_ka < T_θka AND θ_kh < T_θkh:
    skip fall evaluation
```

**LAO — Leg Alignment Override**

```text
if θ_ka < T_θka_la
   AND d_ka > T_dkao
   AND d_kh > T_dkh:
    skip fall evaluation
```

If none of the four override conditions is satisfied, continue with fall-criterion evaluation.

### Step 5: Multi-Criteria Fall Evaluation

Initialize:

```text
fall_crit_count = 0
```

Evaluate the seven criteria independently:

| Criterion | Condition |
|---|---|
| **C1 — Body Angle** | $\theta > T_\theta$ |
| **C2 — Vertical Drop** | $\Delta y > T_{\Delta y}$ |
| **C3 — Height Ratio** | $HR < T_{HR}$ |
| **C4 — Angle Change** | $\Delta\theta > T_{\Delta\theta}$ |
| **C5 — Knee-Ankle Distance** | $d_{ka} < T_{d_{ka}}$ |
| **C6 — Vertical Velocity** | $v_y > T_{v_y}$ |
| **C7 — Head Proximity** | $y_{nose} \geq T_{y_{nose}}$ |

For every satisfied criterion:

```text
fall_crit_count = fall_crit_count + 1
```

### Step 6: Temporal Confirmation

```text
if fall_crit_count >= 3:
    consecutive_fall_frames += 1
else:
    consecutive_fall_frames = 0
```

A fall is declared only when:

```text
fall_crit_count >= 3
AND
consecutive_fall_frames >= 2
```

Otherwise, the current frame is treated as a non-confirmed fall/ADL frame.

### Overall Decision Rule

```text
Fall = 1
if fall_crit_count >= 3
AND consecutive_fall_frames >= 2

Fall = 0
otherwise
```

This multi-stage process combines human detection, pose estimation, temporal feature extraction, early false-positive rejection, multi-criteria voting, and temporal confirmation.
