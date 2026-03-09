# Wing Morphometrics and Segmentation Pipeline

## Repository Organisation

### 1. Image Annotation & Segmentation
* **[`coordinates.ipynb`](coordinates.ipynb)**: An interactive Python notebook designed to capture and retrieve image coordinates via mouse clicks.
* **[`image_predict_sam2.py`](image_predict_sam2.py)**: A batch SAM2 (Segment Anything Model 2) script that applies a standardised set of coordinates to segment images consistently across multiple species.

### 2. Feature Extraction & Normalisation
* **[`momocs_efd.R`](momocs_efd.R)**: An R script for batch image processing that directly extracts Elliptic Fourier Descriptor (EFD) coefficients from the segmented image masks.
* **[`normalisation_scale_rotate_phase.R`](normalisation_scale_rotate_phase.R)**: A comprehensive 3-step normalisation script that performs scale, rotation, and phase normalisation on the extracted morphological features.

### 3. Classification & Modeling
* **[`5fold_lda.R`](5fold_lda.R)**: Generates 5-fold cross-validation sets and executes Linear Discriminant Analysis (LDA) on each set. Converts and outputs the processed data into the required format for GUIDE.

### 📁 Data & Model Inputs
* **`dm_guide/`**: Directory containing input files specifically formatted for GUIDE decision tree and random forest modeling for `dm` cells.
* **`pa2r_guide/`**: Directory containing input files specifically formatted for GUIDE decision tree and random forest modeling for `pa2r` cells.

---
*Note: This pipeline integrates Python-based computer-vision segmentation (SAM2) with R-based morphometric analysis tools (Momocs, LDA, GUIDE).*
