# ----------------------------------------------------------Code to predict Material Level ------------------------------------------------------------------

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the generated dataset
try:
    df = pd.read_csv('Material_Level.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'refined_material_level_data_stronger_relative.csv' not found.")
    print("Please ensure the dataset generation script has been run and the file exists.")
    exit() # Exit if the file isn't found

# --- 1. Data Preparation ---
print("\n--- Data Preparation ---")

# Separate features (X) and target variable (y)
X = df.drop('Material Level', axis=1)
y = df['Material Level']

# Check for and drop any columns that were intermediate calculation steps
potential_leakage_cols = [
    'Consistency_Num', 'Student_Level_Num', 'Course_Level_Num',
    'Present_Material_Level_Num', 'Material_Level_Num',
    'Present Material Level'
]
cols_to_drop = [col for col in potential_leakage_cols if col in X.columns and col != 'Present Material Level' and col != 'Relative Performance'] # Keep Present Material and Relative Perf
if cols_to_drop:
    print(f"Dropping intermediate/leakage columns: {cols_to_drop}")
    X = X.drop(columns=cols_to_drop)

# Identify numerical and categorical features *after potential drops*
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

print(f"Numerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features) # sparse_output=False often easier
    ],
    remainder='passthrough'
)

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)
print(f"Data preprocessed. Shape: {X_processed.shape}")

# Get feature names after preprocessing for later use (importance)
try:
    feature_names = preprocessor.get_feature_names_out()
except AttributeError:
    print("Warning: Could not get feature names using get_feature_names_out(). Check scikit-learn version.")
    feature_names = numerical_features + [f"cat_{col}" for col in categorical_features]


# Label Encoding for the target variable ('Beginner', 'Intermediate', 'Advanced')
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_
num_classes = len(class_names)
print(f"Target variable encoded. Classes: {class_names}")

# --- 2. Data Splitting (Train, Validation, Test) ---
print("\n--- Data Splitting ---")

# Split into Training + Validation set (80%) and Test set (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

# Split Training + Validation set into Training (e.g., 80% of 80% = 64%) and Validation (e.g., 20% of 80% = 16%)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")


print("\n--- Model Training ---")

# Instantiate the XGBoost classifier
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=num_classes,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

# Define a more comprehensive parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
}

# Configure GridSearchCV

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=-1
)

print("Starting GridSearchCV (this may take some time)...")
# Perform Grid Search on the training data
grid_search.fit(X_train, y_train)

print("\nGridSearchCV complete.")
print(f"Best Parameters found: {grid_search.best_params_}")
print(f"Best Cross-validation Accuracy: {grid_search.best_score_:.4f}")

# Get the best estimator found by GridSearchCV
best_xgb_params = grid_search.best_params_

params_for_constructor = best_xgb_params.copy()


fit_only_params = ['early_stopping_rounds', 'eval_set', 'verbose']

for param in fit_only_params:
    if param in params_for_constructor:
        print(f"Note: Removing fit-specific param '{param}' from constructor params.")
        params_for_constructor.pop(param)

print("\nInstantiating final model with best constructor parameters...")
final_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=num_classes,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42,
    **params_for_constructor
)

# Define the evaluation set needed for early stopping
eval_set = [(X_val, y_val)]


print("Training final model with early stopping (using explicit args)...")
final_model.fit(
    X_train,
    y_train,
    eval_set=eval_set,
    verbose=False
)

print("Final model trained successfully!")

# --- 4. Model Evaluation on the Test Set ---
print("\n--- Model Evaluation ---")

y_pred_encoded = final_model.predict(X_test)

y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
y_test_labels = label_encoder.inverse_transform(y_test)

print("\nPredictions on the test set (first 10):")
print(y_pred_labels[:10])

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred_encoded)
print(f"\nAccuracy on Test Set: {accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_encoded)
print("\nConfusion Matrix (Test Set):")
# Plotting the confusion matrix for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Level')
plt.ylabel('True Level')
plt.title('Confusion Matrix')
plt.show()


# Classification Report
cr = classification_report(y_test, y_pred_encoded, target_names=class_names)
print("\nClassification Report (Test Set):")
print(cr)

# --- 5. Feature Importance ---
print("\n--- Feature Importance ---")

feature_importance = final_model.feature_importances_


if 'feature_names' not in locals():
     print("Warning: 'feature_names' not found. Importance plot may lack labels.")
     try:
        feature_names = preprocessor.get_feature_names_out()
     except Exception:
        feature_names = [f'feature_{i}' for i in range(X_processed.shape[1])]

feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importance Ranking:")
print(feature_importance_df)


plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15)) # Display top 15 features
plt.title('Top 15 Feature Importances')
plt.tight_layout()
plt.show()




# --- 6. Prediction Function and Usage ---
print("\n--- Prediction Function ---")


original_feature_columns = X.columns.tolist()
print(f"Original features expected by the preprocessor: {original_feature_columns}")

def predict_material_level(input_data, model, processor, encoder, feature_columns):

    try:

        input_df = pd.DataFrame([input_data], columns=feature_columns)
        print(f"\nInput DataFrame created:\n{input_df}")


        input_processed = processor.transform(input_df)
        print(f"Input data preprocessed. Shape: {input_processed.shape}")

        prediction_encoded = model.predict(input_processed)
        print(f"Model prediction (encoded): {prediction_encoded}")


        predicted_label = encoder.inverse_transform(prediction_encoded)
        print(f"Predicted label (decoded): {predicted_label}")

        return predicted_label[0]

    except KeyError as e:
         print(f"\nError: Missing feature in input_data: {e}")
         print(f"Please ensure your input includes all required features: {feature_columns}")
         return None
    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")
        return None

print("\n--- Prediction Example ---")


new_student_data_1 = {
    'Age': 14,
    'IQ': 115.0,
    'Time per Day (hrs)': 2.1,
    'Assessment Score': 85,
    # --- Categorical features ---
    'Level of Student': 'Intermediate',
    'Level of Course': 'Intermediate',
    'Course Name': 'Math',
    'Consistency': 'Regular',
    'Present Material Level': 'Intermediate',
    'Relative Performance': 10.0
}

# Make the prediction using the function
predicted_level_1 = predict_material_level(
    new_student_data_1,
    final_model,
    preprocessor,
    label_encoder,
    original_feature_columns
)

if predicted_level_1:
    print(f"\n---> Final Predicted Material Level for student 1: {predicted_level_1}")

print("\n--- Another Prediction Example ---")

new_student_data_2 = {
    'Age': 9,
    'IQ': 92.0,
    'Time per Day (hrs)': 0.7,
    'Assessment Score': 58,
    'Level of Student': 'Beginner',
    'Level of Course': 'Beginner',
    'Course Name': 'English',
    'Consistency': 'Irregular',
    'Present Material Level': 'Beginner',
    'Relative Performance': -2.0
}

predicted_level_2 = predict_material_level(
    new_student_data_2,
    final_model,
    preprocessor,
    label_encoder,
    original_feature_columns
)

if predicted_level_2:
    print(f"\n---> Final Predicted Material Level for student 2: {predicted_level_2}")
