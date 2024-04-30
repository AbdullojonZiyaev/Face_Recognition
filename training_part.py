import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data from the CSV file
df = pd.read_csv('face_data.csv')

# Extract features (face encodings) and labels
face_encodings = df.iloc[:, :-1].values
labels = df['label'].values

# Scale features
scaler = StandardScaler()
face_encodings_scaled = scaler.fit_transform(face_encodings)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(face_encodings_scaled, labels, test_size=0.2, random_state=42, stratify=labels)

# Define the parameter grid for the grid search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': [0.1, 1, 10]
}

# Create the SVM model
svm_model = SVC()

# Perform grid search with cross-validation
"""grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', return_train_score=False)
grid_search.fit(X_train, y_train)

# Print results for parameter combinations with accuracy between 0.75 and 1.0
results_df = pd.DataFrame(grid_search.cv_results_)
filtered_results = results_df[(results_df['mean_test_score'] > 0.75) & (results_df['mean_test_score'] < 1.0)]
print("Results for parameter combinations with accuracy between 0.75 and 1.0:")
print(filtered_results[['param_C', 'param_kernel', 'param_gamma', 'mean_test_score']])
"""
# Train the SVM model with the best hyperparameters
best_model = SVC(C=10, kernel="rbf", gamma=0.1)  # Update with the best parameters
best_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(best_model, 'face_recognition_model.joblib')

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')

# Make predictions on the testing set
y_pred = best_model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Cross-validation (optional)
cv_scores = cross_val_score(best_model, face_encodings_scaled, labels, cv=5)
print("Cross-Validation Mean Accuracy:", cv_scores.mean())
print("Unique Labels in Training Set:", np.unique(y_train))
print("Unique Labels in Testing Set:", np.unique(y_test))
