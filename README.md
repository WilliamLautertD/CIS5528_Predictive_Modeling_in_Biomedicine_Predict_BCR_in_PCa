# CIS5528_Predictive_Modeling_in_Biomedicine_Predict_BCR_in_PCa
Repository for CIS5528 project class. Our goal is to predict BCR in prostate cancer patients using multimodal data, including H&E-stained whole slide images, mpMRI, and clinical information, based on CHIMERA Challenge (https://chimera.grand-challenge.org). 

# Data sets
- CHIMERA Challenge (https://chimera.grand-challenge.org): s3://chimera-challenge/v2/task1/
- The LEOPARD challenge (https://leopard.grand-challenge.org):
- PI-CAI challenge (https://pi-cai.grand-challenge.org): https://pi-cai.grand-challenge.org/DATA/
- The PANDA challenge (https://panda.grand-challenge.org): https://panda.grand-challenge.org/data/
- TCGA-PRAD | The Cancer Genome Atlas Prostate Adenocarcinoma Collection: https://www.cancerimagingarchive.net/collection/tcga-prad/

# Summary data sets

## PI-CAI challenge:

-   Public Training and Development Dataset (1500 cases)
-   Imaging data: zenodo.org/record/6624726 (DOI: 10.5281/zenodo.6624726)
-   Annotations: github.com/DIAGNijmegen/picai_labels (expert-derived annotations for all 1500 cases)
-   1075 cases have benign tissue or indolent PCa and 425 cases have csPCa.
-   All expert-derived csPCa annotations carry granular or multi-class labels (ISUP ≤ 1, 2, 3, 4, 5), while all automated AI-derived annotations carry binary labels (ISUP ≤ 1 or ≥ 2).

### Labels

| Label                       | [Expert-Derived Annotations] | [AI-Derived Annotations] or [Additional Expert-Derived Annotations] ([Pooch et al., 2025](https://www.medrxiv.org/content/early/2025/05/13/2025.05.13.25327456))  |
|:---------------------------:|:---------------------:|:---------------------:|
| 0                           | [ISUP][ISUP] ≤ 1      | [ISUP][ISUP] ≤ 1      |
| 1                           | N/A                   | [ISUP][ISUP] ≥ 2      |
| 2                           | [ISUP][ISUP] 2        | N/A                   |
| 3                           | [ISUP][ISUP] 3        | N/A                   |
| 4                           | [ISUP][ISUP] 4        | N/A                   |
| 5                           | [ISUP][ISUP] 5        | N/A                   |

### PI-CAI data set summary 

| Characteristic              | Frequency       |
|:----------------------------|:---------------:|
| Number of patients             | 1476            |
| Number of cases                | 1500            |
| — Benign or indolent PCa       | 1075            |
| — csPCa (ISUP ≥ 2)             | 425             |
| Median age (years)             | 66 (IQR: 61–70) |
| Median PSA (ng/mL)             | 8.5 (IQR: 6–13) |
| Median prostate volume (mL)    | 57 (IQR: 40–80) |
| Number of positive MRI lesions | 1087            |
| — PI-RADS 3                    | 246 (23%)       |
| — PI-RADS 4                    | 438 (40%)       |
| — PI-RADS 5                    | 403 (37%)       |
| Number of ISUP-based lesions   | 776             |
| — ISUP 1                       | 311 (40%)       |
| — ISUP 2                       | 260 (34%)       |
| — ISUP 3                       | 109 (14%)       |
| — ISUP 4                       | 41 (5%)         |
| — ISUP 5                       | 55 (7%)         |

-   Citation: __[Saha A, Bosma JS, Twilt JJ, et al. Artificial intelligence and radiologists in prostate cancer detection on MRI (PI-CAI): an international, paired, non-inferiority, confirmatory study. Lancet Oncol 2024; 25: 879–887](https://www.thelancet.com/journals/lanonc/article/PIIS1470-2045(24)00220-1/fulltext)__  

## CHIMERA Challenge (Task 1) [WIP]

### Chimera data set summary

| Characteristic                        | Frequency              |
|:--------------------------------------|:----------------------:|
| Number of patients                    | 95                     |
| — BCR negative (no recurrence)        | 68 (72%)               |
| — BCR positive (recurrence)           | 27 (28%)               |
| Median age (years)                    | 66 (IQR: 60–69)        |
| Median pre-operative PSA (ng/mL)      | 7.8 (IQR: 5.2–12.0)   |
| Median follow-up time (months)        | 38.0 (IQR: 18.6–58.4) |
| **ISUP grade group**                  |                        |
| — ISUP 1                              | 8 (8%)                 |
| — ISUP 2                              | 44 (46%)               |
| — ISUP 3                              | 27 (28%)               |
| — ISUP 4                              | 7 (7%)                 |
| — ISUP 5                              | 9 (9%)                 |
| **Pathological T stage**              |                        |
| — pT2                                 | 34 (36%)               |
| — pT2a                                | 1 (1%)                 |
| — pT2c                                | 16 (17%)               |
| — pT3a                                | 28 (29%)               |
| — pT3b                                | 11 (12%)               |
| — pT4                                 | 4 (4%)                 |
| — pT4b                                | 1 (1%)                 |
| **Pathological features**             |                        |
| Positive surgical margins             | 50/95 (53%)            |
| Capsular penetration                  | 39/91 (43%)            |
| Seminal vesicle invasion              | 14/95 (15%)            |
| Lymphovascular invasion               | 17/95 (18%)            |
| Positive lymph nodes                  | 10/42 (24%)            |
| **Earlier therapy**                   |                        |
| — None                                | 92 (97%)               |
| — Other/unknown                       | 3 (3%)                 |

### WSI pyramidal TIF structure
- 190 whole-slide images (H&E), stored as pyramidal TIFFs with multiple resolution levels
- **WSI levels** (most slides have 5): level 0 = full resolution (~100k–170k px), then 4x, 16x, 64x, 256x downsamples
- **Tissue mask levels** (most have 7): start at 4x downsample (= WSI level 1), then step by 2x down to 64x
- Mask level 0 spatially matches WSI level 1 — no resize needed at that scale
- Pre-extracted patch features (`.pt`) and coordinates (`.npy`) are provided for all 190 slides

`extract_wsi_previews.py` generates thumbnail and level-3 (64x) PNG previews for all WSIs into `previews/`.

###MRI
`MRI.ipynb` extracts image features and saves a csv file containing them.
`encoding.ipynb` compresses MRI radiometric features into 2- and 8-dimensional latent embeddings with an unsupervised autoencoder.

# Algorithms

- Prostate segmentation: https://grand-challenge.org/algorithms/prostate-segmentation/

# Citations 

- [Common Limitations of Image Processing Metrics: A Picture Story. REINKE, A. et al. (2023).](https://arxiv.org/pdf/2104.05642)
