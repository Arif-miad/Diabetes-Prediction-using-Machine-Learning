# Diabetes Prediction using Machine Learning

This repository contains the code and resources to predict the likelihood of diabetes using machine learning models. The project follows a structured approach starting from exploratory data analysis (EDA) to training multiple classification models and deploying the best one.

## Dataset

The dataset used in this project includes multiple features such as:
- Number of pregnancies
- Plasma glucose concentration
- Diastolic blood pressure
- Triceps skin fold thickness
- Serum insulin
- BMI
- Diabetes pedigree function
- Age

Here’s how you can structure your GitHub repository and `README.md` for your diabetes prediction project. This includes the 15 steps of code implementation and an overview of the project.

### **GitHub Repository Structure:**
```
diabetes-prediction/
│
├── data/
│   └── diabetes.csv           # Your dataset
│
├── notebooks/
│   └── diabetes_analysis.ipynb # Jupyter notebook for data analysis and model building
│
├── models/
│   └── best_model.pkl         # Pickled model for deployment
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project description and setup
└── .gitignore                 # Ignore unnecessary files
```

### **README.md**

```markdown


## Steps in this Project

### 1. **Import Libraries**
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
```

### 2. **Load Dataset**
```python
df = pd.read_csv('/kaggle/input/diabetes-classification-dataset/diabetesData.csv')
```

### 3. **Check for Missing Values**
```python
df = pd.read_csv('/kaggle/input/diabetes-classification-dataset/diabetesData.csv')
```

### 4. **Exploratory Data Analysis (EDA)**
```python
sns.set_theme(style="whitegrid")

# --- 1. Histogram ---
df.hist(bins=15, figsize=(15, 10), color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Features", fontsize=16)
plt.tight_layout()
plt.show()
```

### 5. **Handling Categorical Variables**
```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df['target'] = label_encoder.fit_transform(df['target'])
```

### 6. **Removing Outliers**
```python
def remove_outliers(df, column):
    """
    Removes outliers from a column in a DataFrame using the IQR method.
    """
    Q1 = df[column].quantile(0.25)  # First quartile (25th percentile)
    Q3 = df[column].quantile(0.75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1  # Interquartile range
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df


numeric_columns = ['num_preg', 'plasma_glucose_conc', 'bp', 'tricepsthickness', 
                   'insulin', 'BMI', 'pedigree_func', 'age']





for col in numeric_columns:
    df = remove_outliers(df, col)


print(f"Original dataset shape: {df.shape}")
print(f"Cleaned dataset shape: {df.shape}")


plt.figure(figsize=(15, 10))
sns.boxplot(data=df[numeric_columns], orient='h', palette="Set2")
plt.title("Boxplot After Removing Outliers", fontsize=16)
plt.show()

```

### 7. **Feature Scaling (MinMax Scaling)**
```python
from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# List of numerical columns
numeric_columns = ['num_preg', 'plasma_glucose_conc', 'bp', 'tricepsthickness', 
                   'insulin', 'BMI', 'pedigree_func', 'age']

# Apply Min-Max Scaling
df = df.copy()  # Create a copy of the dataset
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])


print("Scaled Data (First 5 Rows):")
print(df.head())


print("\nMin-Max Values After Scaling:")
for col in numeric_columns:
    print(f"{col}: Min = {df[col].min()}, Max = {df[col].max()}")
```

### 8. **Splitting Data into Training and Testing Sets**
```python
X = df[numeric_columns]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```



### 10. **Train at Classifier**
```python
# Step 7: Define Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier()
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Make predictions
    acc = accuracy_score(y_test, y_pred)  # Evaluate accuracy
    results[name] = acc  # Store the result
    print(f"{name} Accuracy: {acc:.2f}")
}
```




### 12. **Evaluate Models Using Accuracy Score**
```python
cv_scores = cross_val_score(best_model, X, y, cv=5)
print(f"\nCross-Validation Scores for {best_model_name}: {cv_scores}")
print(f"Mean CV Score: {np.mean(cv_scores):.2f}"))

```

### 13. **Confusion Matrix and Classification Report**
```python
# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_best), annot=True, fmt='d', cmap="Blues")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.show()
# Step 10: Detailed Evaluation of the Best Model
y_pred_best = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))
```

### 14. **Model Deployment (Save the Best Model)**
```python
joblib.dump(xgb_model, 'models/best_model.pkl')
```

### 15. **Load the Model and Make Predictions**
```python
cv_scores = cross_val_score(best_model, X, y, cv=5)
print(f"\nCross-Validation Scores for {best_model_name}: {cv_scores}")
print(f"Mean CV Score: {np.mean(cv_scores):.2f}")
```

## Requirements

To install the dependencies, run:
```bash
pip install -r requirements.txt
```

### Example of `requirements.txt`:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
```

## Conclusion

This repository demonstrates the entire pipeline for predicting diabetes using machine learning, from data preprocessing and exploratory data analysis (EDA) to training models and deploying the best one.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

### Key Elements in `README.md`:
- **Project Overview**: Describes the dataset and the aim of the project.
- **Step-by-Step Instructions**: 15 steps of code implementation for clarity and reproducibility.
- **Dependencies**: Information about required libraries and how to install them.
- **Model Saving and Deployment**: How to save and load models using `joblib`.
- **Results**: Accuracy evaluation and classification metrics.

