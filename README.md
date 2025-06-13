# RNA-Seq Cancer Classification Analysis

A comprehensive machine learning pipeline for cancer type classification using RNA-Seq gene expression data from the PANCAN dataset.

## 📊 Project Overview

This project implements a complete end-to-end pipeline for analyzing RNA-Seq gene expression data to classify different cancer types. Using the PANCAN dataset from UCI Machine Learning Repository, we developed and evaluated multiple machine learning models to distinguish between 5 cancer types based on gene expression profiles.

### 🎯 Objectives
- Perform comprehensive exploratory data analysis (EDA) on RNA-Seq data
- Implement proper preprocessing pipeline for gene expression data
- Develop and compare multiple machine learning classification models
- Identify key biomarker genes for cancer classification
- Create ensemble models for robust predictions

## 📁 Dataset Information

**Source:** [UCI ML Repository - Gene Expression Cancer RNA-Seq](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq)

**Dataset Details:**
- **Samples:** 801 patients
- **Features:** 20,531 genes
- **Cancer Types:** 5 classes
  - BRCA: Breast Invasive Carcinoma
  - KIRC: Kidney Renal Clear Cell Carcinoma  
  - COAD: Colon Adenocarcinoma
  - LUAD: Lung Adenocarcinoma
  - PRAD: Prostate Adenocarcinoma

## 🚀 Key Results

- **Best Model Accuracy:** >95%
- **Cross-Validation Score:** Consistently high across all folds
- **Feature Reduction:** From 20,531 to optimized gene subset
- **Ensemble Performance:** Enhanced robustness through model combination

## 📋 Requirements

```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
scipy>=1.7.0
jupyter>=1.0.0
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/hmzi67/rna-seq-cancer-classification.git
cd rna-seq-cancer-classification
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
- Download `data.csv` and `labels.csv` from the UCI repository
- Place them in the project root directory

## 📖 Usage

### Running the Complete Pipeline

Execute the Jupyter notebook cells in order:

1. **Data Loading and EDA** (`cell_1_imports.py` - `cell_4_gene_expression_analysis.py`)
2. **Preprocessing** (`cell_5_distribution_analysis.py` - `cell_8_normalization.py`)
3. **Dimensionality Reduction** (`cell_9_dimensionality_reduction.py`)
4. **Machine Learning** (`cell_1_ml_imports.py` - `cell_11_save_results.py`)

### Quick Start
```python
# Load preprocessed data
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
X = pd.read_csv('processed_gene_expression_data.csv', index_col=0)
y = pd.read_csv('processed_labels.csv', index_col=0)

# Load trained model
model = joblib.load('best_cancer_classifier.pkl')

# Make predictions
predictions = model.predict(X)
```


## 🔬 Methodology

### 1. Data Preprocessing
- **Quality Control:** Missing value analysis, outlier detection
- **Gene Filtering:** Removed low-expression and low-variance genes
- **Normalization:** Log2(x + 1) transformation for RNA-Seq data
- **Feature Selection:** Variance-based and PCA-based selection

### 2. Exploratory Data Analysis
- Expression distribution analysis
- Class balance evaluation
- Sample quality assessment
- Gene variance characterization

### 3. Machine Learning Pipeline
- **Models Evaluated:**
  - Logistic Regression
  - Random Forest
  - Support Vector Machine
  - Gradient Boosting
  - Extra Trees
  - K-Nearest Neighbors
  - Naive Bayes
  - Decision Tree
  - Neural Network (MLP)

- **Validation Strategy:**
  - 5-fold stratified cross-validation
  - 80-20 train-test split
  - Hyperparameter tuning with GridSearchCV

### 4. Model Evaluation
- Accuracy, Precision, Recall, F1-score
- ROC curves and AUC scores (One-vs-Rest)
- Confusion matrices
- Feature importance analysis

## 📈 Results Summary

### Model Performance
| Model | Accuracy | F1-Score | AUC |
|-------|----------|----------|-----|
| Random Forest | 0.993789 | 0.993758 | 1.000 |
| SVM | 0.987578 | 0.987503 | 0.999 |
| Logistic Regression | 0.993789 | 0.993758 | 1.000 |
| Decision Tree | 0.987578 | 0.987539 | 0.9910 |

### Key Findings
- ✅ High classification accuracy across all cancer types
- ✅ Identified biologically relevant biomarker genes
- ✅ Ensemble methods improved model robustness
- ✅ Clear separation between cancer types in dimensionality reduction

## 🧬 Biological Insights

### Top Contributing Genes
The analysis identified several genes with high importance for cancer classification:
- Biological pathway analysis reveals cancer-relevant gene signatures
- Results align with known cancer biomarkers in literature

## 📊 Visualizations

The project generates comprehensive visualizations:
- **Class Distribution:** Cancer type prevalence
- **Expression Patterns:** Gene expression distributions
- **PCA Analysis:** Principal component visualization
- **t-SNE Plot:** Non-linear dimensionality reduction
- **ROC Curves:** Model performance evaluation
- **Confusion Matrices:** Classification accuracy breakdown

## 🔧 Configuration

### Preprocessing Parameters
```python
MIN_SAMPLES_EXPRESSED = 0.1  # 10% of samples
MIN_EXPRESSION_LEVEL = 1.0   # Minimum expression threshold
VARIANCE_THRESHOLD = 0.1     # Bottom 10% variance filter
```

### Model Parameters
```python
CV_FOLDS = 5                 # Cross-validation folds
TEST_SIZE = 0.2             # Train-test split ratio
RANDOM_STATE = 42           # Reproducibility seed
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 📚 References

1. **Dataset:** Weinstein, John N., et al. "The cancer genome atlas pan-cancer analysis project." Nature genetics 45.10 (2013): 1113-1120.
2. **RNA-Seq Analysis:** Love, Michael I., Wolfgang Huber, and Simon Anders. "Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2." Genome biology 15.12 (2014): 1-21.
3. **Machine Learning:** Pedregosa, F., et al. "Scikit-learn: Machine learning in Python." Journal of machine learning research 12.Oct (2011): 2825-2830.

## 👤 Author

**hmzi67**
- GitHub: [@hmzi67](https://github.com/hmzi67)
- LinkedIn: [Connect with me](https://linkedin.com/in/hmzi67)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- The Cancer Genome Atlas (TCGA) Research Network
- Open-source community for excellent machine learning tools

## 📊 Project Status

**Status:** ✅ Complete  
**Last Updated:** 2025-06-13 07:12:05 UTC  
**Version:** 1.0.0  
**Created By:** hmzi67

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐
