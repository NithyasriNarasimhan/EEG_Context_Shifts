# EEG-Based Analysis of Neurocognitive Context Shifts

**Author:** Nithyasri Narasimhan  
**Email:** nn7985@rit.edu  
**Institution:** Rochester Institute of Technology  
**Date:** April 17, 2025  

---

## Project Overview
This project analyzes EEG signals to investigate neurocognitive context shifts in language processing, comparing individuals with ADHD to healthy controls. The dataset includes EEG recordings from male and female subjects, divided into ADHD and control groups.

The **baseline implementation** uses machine learning models (Logistic Regression, Random Forest, SVM) with statistical features (mean, standard deviation).  
The **core implementation** explores advanced methods to enhance classification performance.(needs more updates)

---

## Repository Structure

- **baseline.ipynb**  
  Baseline implementation with:
  - Data loading from `data/*.mat`
  - Feature extraction (mean, standard deviation per channel)
  - Training and evaluation of Logistic Regression, Random Forest, and SVM models
  - Visualizations: Confusion matrix, ROC curves, raw EEG signals, and signal statistics

- **coreimplementation.ipynb**  
  Core implementation with advanced methods (e.g., spectral features, pending confirmation)

- **data/**  
  - `MC.mat`: Male control EEG data
  - `MADHD.mat`: Male ADHD EEG data
  - `FC.mat`: Female control EEG data
  - `FADHD.mat`: Female ADHD EEG data

- **figures/**  
  - `confusion_matrix_rf.png`: Confusion matrix for Random Forest (baseline)
  - `roc_curve_comparison.png`: ROC curves for baseline models
  - `sample_raw_eeg_signals.png`: Sample raw EEG signals
  - `signal_strength_and_variability.png`: Signal strength and variability plot

- **src/**  
  - `preprocess.py`: Preprocessing functions (data loading, feature extraction)
  - `train.py`: Model training and evaluation functions

- **requirements.txt**  
  Python dependencies for running the notebooks.

---

## Prerequisites

- **Python**: Version 3.11 or higher
- **Dependencies**: Install required packages:
  ```bash
  pip install -r requirements.txt
  ```

  **Contents of `requirements.txt`**:
  ```
  numpy
  scipy
  pandas
  scikit-learn
  matplotlib
  ```

- **Jupyter Notebook**: For running `.ipynb` files:
  ```bash
  pip install jupyter
  ```

- **Dataset**: Included in `data/`.  
  If missing, download from Mendeley Data and place it in the `data/` folder.

---

## How to Run

### Setup

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/NithyasriNarasimhan/EEG_Context_Shifts.git
    cd EEG_Context_Shifts
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Verify Dataset**:
    Ensure `data/` contains:
    - `MC.mat`
    - `MADHD.mat`
    - `FC.mat`
    - `FADHD.mat`

---

### Running the Baseline (`baseline.ipynb`)

1. **Start Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

2. **Open `baseline.ipynb`** in the Jupyter interface.

3. **Run all cells in order**:
    - Loads EEG data from `data/`
    - Extracts features (mean, standard deviation per channel)
    - Trains and evaluates models: Logistic Regression, Random Forest, and SVM
    - Generates performance metrics and visualizations

4. **Expected Outputs**:

| Model              | Accuracy | F1 Score | AUC   |
|--------------------|----------|----------|-------|
| Logistic Regression| 73.75%   | 0.6557   | 0.8528|
| Random Forest      | 78.75%   | 0.7463   | 0.8559|
| SVM (RBF Kernel)   | 72.50%   | 0.6207   | 0.8672|

- **Figures (saved to `figures/`)**:
  - `confusion_matrix_rf.png`
  - `roc_curve_comparison.png`
  - `sample_raw_eeg_signals.png`
  - `signal_strength_and_variability.png`

- **Console Output**:  
  Classification report with precision, recall, and F1 scores.

---

### Running the Core Implementation (`coreimplementation.ipynb`)

1. **Open `coreimplementation.ipynb`** in Jupyter Notebook.

2. **Run all cells in order**:
    - Loads and preprocesses data
    - Implements advanced methods (e.g., spectral features, pending confirmation)
    - Trains and evaluates models
    - Generates visualizations (if implemented)

3. **Expected Outputs**:
    - Performance metrics (to be updated based on implementation)
    - Figures (e.g., ROC curves, confusion matrices, EEG plots)


---

## Notes

- Ensure `data/` is in the repository root with all `.mat` files.
- If `src/` scripts are used, they are imported in the notebooks (e.g., `from src.preprocess import ...`).
- Update `requirements.txt` if `coreimplementation.ipynb` uses additional libraries (e.g., `pywt`, `tensorflow`).

---

## Results

- **Baseline Performance**:
  - Random Forest: Highest accuracy (78.75%) and F1 score (0.7463)
  - SVM (RBF kernel): Highest AUC (0.8672)

- **Visualizations**:
  - See `figures/` for confusion matrix, ROC curves, raw EEG signals, and signal statistics.

- **Core Implementation**:
  - Performance metrics pending final implementation
  - Expected to improve on baseline with advanced methods

---

## Figures

- `confusion_matrix_rf.png`: Confusion matrix for Random Forest (baseline)
- `roc_curve_comparison.png`: ROC curves for baseline models
- `sample_raw_eeg_signals.png`: Sample raw EEG signals
- `signal_strength_and_variability.png`: Signal strength and variability plot
