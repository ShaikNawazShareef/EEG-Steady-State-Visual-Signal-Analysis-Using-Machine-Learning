# 🧠 EEG Steady-State Visual Signal Analysis Using Machine Learning

## 📌 Project Overview

This project focuses on analysing **Electroencephalography (EEG) Steady-State Visual Evoked Potential (SSVEP)** signals using Machine Learning algorithms.

EEG signals are electrical activities produced by the human brain. Machine Learning techniques are applied to analyse these signals, build predictive models, and evaluate their performance.

The project implements multiple regression and classification algorithms to explore different approaches to EEG signal analysis.

## 🎯 Objectives

- 📊 Load and analyse EEG datasets.
- 🧹 Handle missing values using SimpleImputer.
- 📈 Visualize data using Pairplot and Correlation Heatmap.
- 🔍 Perform data preprocessing and train-test splitting.
- 🤖 Implement multiple Machine Learning algorithms.
- 📉 Evaluate model performance using appropriate metrics.
- ⚙️ Optimize SVM hyperparameters using GridSearchCV.

## 🛠️ Technologies & Libraries Used

- 🐍 Python
- 📊 Pandas
- 🔢 NumPy
- 📉 Matplotlib
- 🎨 Seaborn
- 🤖 Scikit-learn
- ☁️ Google Colab

## 🤖 Machine Learning Algorithms Implemented

This project implements **6 distinct Machine Learning algorithms**.

| No. | Algorithm | Type |
|---|---|---|
| 1 | Linear Regression | Regression |
| 2 | RANSAC Regressor | Regression |
| 3 | Perceptron | Classification |
| 4 | Logistic Regression | Classification |
| 5 | Decision Tree Classifier | Classification |
| 6 | Support Vector Machine (SVM) | Classification |

*Note: Linear Regression is implemented twice in the original code.*

## 🔬 Data Preprocessing

The following preprocessing techniques are used:

1. **Dataset Loading:** Import EEG data from a CSV file.
2. **Missing Value Detection:** Identify missing values using `isna()`.
3. **Data Imputation:** Replace missing values using `SimpleImputer(strategy='most_frequent')`.
4. **Data Visualization:** Use Seaborn Pairplot.
5. **Correlation Analysis:** Generate a correlation heatmap.
6. **Train-Test Split:** Divide data into 70% training and 30% testing.

## ⚙️ Model Evaluation Metrics

The following evaluation metrics are implemented:

- 📌 Mean Squared Error (MSE)
- 📌 R² Score
- 📌 Training Accuracy
- 📌 Testing Accuracy

## 🔍 SVM Optimization

Support Vector Machine is further optimized using **GridSearchCV**.

The following kernels are explored:

- Linear Kernel
- Polynomial Kernel
- RBF Kernel

Hyperparameters:
- `C`
- `gamma`

Cross-validation: 4-Fold (`cv=4`)

## 📂 Project Structure

```text
EEG-Steady-State-Visual/
│
├── A001SB1_1.csv
├── EEG_Steady_State_Visual.ipynb
└── README.md
```

## 🚀 How to Run the Project

**Step 1:** Clone the repository.

```bash
git clone https://github.com/your-username/EEG-Steady-State-Visual.git
```

**Step 2:** Install the required libraries.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**Step 3:** Open the notebook using Google Colab or Jupyter Notebook.

**Step 4:** Upload the EEG dataset (`A001SB1_1.csv`).

**Step 5:** Execute all notebook cells.

## 📈 Expected Outcomes

- Understand EEG dataset characteristics.
- Visualize relationships between EEG features.
- Train multiple Machine Learning algorithms.
- Compare regression and classification approaches.
- Identify suitable SVM hyperparameters through GridSearchCV.

*Actual model performance depends on the dataset features and target labels.*

## 💡 Applications

- 🧠 Brain-Computer Interface (BCI)
- 📡 EEG Signal Processing
- 👁️ Visual Stimulus Analysis
- 🤖 Biomedical Machine Learning
- 🔬 Neuroscience Research

## 👨‍💻 Author

**Nawaz Shareef Shaik**

GitHub: https://github.com/ShaikNawazShareef

## ⭐ Acknowledgements

- Scikit-learn Documentation
- Pandas Documentation
- NumPy Documentation
- Seaborn Documentation

---

⭐ If you find this project useful, consider giving it a star!
