# A Pose Landmark-Based Lightweight Human Fall-Detection Framework Using Spatiotemporal Features

This repository contains the implementation materials associated with
the paper:

**A Pose Landmark-Based Lightweight Human Fall-Detection Framework Using
Spatiotemporal Features**

**Authors:** Ekram Alam, Abu Sufian, Paramartha Dutta, and Marco Leo

**Journal:** *Information*\
**Year:** 2026\
**Volume:** 17\
**Article:** 976\
**DOI:** 10.3390/info17100976

## Paper

-   **MDPI:** https://www.mdpi.com/2078-2489/17/10/976
-   **DOI:** https://doi.org/10.3390/info17100976

## Overview

The proposed human fall-detection framework is a lightweight, pose
landmark-based approach that combines **YOLO12n** for human detection
and region-of-interest localization with **MediaPipe Pose** for human
pose landmark extraction.

The framework analyzes spatial and temporal body features and uses
traceable rule-based fall criteria together with early override
conditions to distinguish falls from Activities of Daily Living (ADL).

The system does **not** use YOLO12n to directly classify fall/non-fall
events. YOLO12n is used for human detection, while the fall decision is
obtained from pose-derived spatiotemporal features and rule-based
evaluation.

### Main components

-   YOLO12n for human detection
-   MediaPipe Pose for 33-point pose landmark extraction
-   Spatiotemporal feature extraction
-   Four early override conditions
-   Seven fall-detection criteria
-   Temporal confirmation over consecutive frames
-   Traceability information panel for interpreting detection decisions
-   CPU-oriented lightweight implementation

## Fall-Detection Criteria

The framework evaluates seven criteria:

1.  **C1 --- Body Angle (`θ`)**
2.  **C2 --- Vertical Drop (`Δy`)**
3.  **C3 --- Height Ratio (`HR`)**
4.  **C4 --- Angle Change (`Δθ`)**
5.  **C5 --- Knee-Ankle Vertical Distance (`dka`)**
6.  **C6 --- Vertical Velocity (`vy`)**
7.  **C7 --- Head Proximity to Ground (`ynose`)**

A fall is declared when at least **three of the seven criteria** are
satisfied for **two consecutive frames**.

## Override Conditions

Four early rejection/override conditions are used to reduce false
positives from non-fall activities:

-   **HPO --- Head Position Override**
-   **ARO --- Aspect Ratio Override**
-   **VLO --- Vertical Leg Override**
-   **LAO --- Leg Alignment Override**

These conditions are applied before the seven fall criteria are
evaluated.

## Datasets

The experiments reported in the paper use two publicly available
fall-detection datasets.

### 1. GMDCSA-24

**GMDCSA-24: A Dataset for Human Fall Detection in Videos**

-   81 ADL clips
-   79 fall clips
-   4 subjects
-   3 different home setups

Dataset:

https://zenodo.org/records/12921216

Dataset publication:

https://doi.org/10.1016/j.dib.2024.110892

### 2. UR Fall Detection Dataset (URFD)

The URFD dataset contains:

-   40 ADL video clips
-   30 fall video clips
-   5 subjects

For the experiments reported in the paper, the `cam0` clips were used
for both ADL and fall events.

Dataset:

https://fenix.ur.edu.pl/\~mkepski/ds/uf.html

**Important:** Please check and comply with the original dataset license
and usage conditions before redistributing or using the datasets.



## System Pipeline

The overall processing pipeline is:

``` text
Input Video
     |
     v
Person Detection using YOLO12n
     |
     v
Person ROI Cropping
     |
     v
MediaPipe Pose Estimation
     |
     v
Pose Landmark Validation / Pre-processing
     |
     v
Spatiotemporal Feature Extraction
     |
     v
Override Conditions
     |
     v
Seven Fall Criteria
     |
     v
Temporal Confirmation
     |
     v
Fall / ADL Decision
```

## Software Environment

The implementation described in the paper was developed using:

-   Python 3.10.19
-   PyTorch 2.9.1
-   Ultralytics YOLO12n
-   MediaPipe Pose 0.10.20
-   OpenCV 4.11.0

The experiments were conducted on a Windows 11 Pro (64-bit) laptop with:

-   Intel Core i5-8265U @ 1.60 GHz
-   8 GB RAM

The framework is designed to operate without requiring GPU acceleration.




## Citation

If you use this implementation or the methodology in your research,
please cite:

``` bibtex
@article{alam2026pose,
  title   = {A Pose Landmark-Based Lightweight Human Fall-Detection Framework Using Spatiotemporal Features},
  author  = {Alam, Ekram and Sufian, Abu and Dutta, Paramartha and Leo, Marco},
  journal = {Information},
  year    = {2026},
  volume  = {17},
  number  = {10},
  pages   = {976},
  doi     = {10.3390/info17100976}
}
```

## Related Dataset Citation

If you use GMDCSA-24, please also cite its original dataset publication:

``` bibtex
@article{alam2024gmdcsa24,
  title   = {GMDCSA-24: A dataset for human fall detection in videos},
  author  = {Alam, Ekram and Sufian, Abu and Dutta, Paramartha and Leo, Marco and Hameed, Ibrahim Al},
  journal = {Data in Brief},
  volume  = {57},
  pages   = {110892},
  year    = {2024},
  doi     = {10.1016/j.dib.2024.110892}
}
```

## License and Dataset Usage

This repository contains implementation materials associated with the
research paper. The licenses of third-party software, models, and
datasets remain applicable.

The datasets used in the study are hosted by their respective providers
and are **not included in this repository**.

Users are responsible for complying with the terms and licenses of:

-   GMDCSA-24
-   URFD
-   YOLO12n / Ultralytics
-   MediaPipe
-   PyTorch
-   OpenCV

## Acknowledgment

This repository accompanies the research article published in
*Information* (MDPI):

**A Pose Landmark-Based Lightweight Human Fall-Detection Framework Using
Spatiotemporal Features**

Paper: https://www.mdpi.com/2078-2489/17/10/976
