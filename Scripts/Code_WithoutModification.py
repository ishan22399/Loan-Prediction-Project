import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import joblib


# Load data from CSV
data = pd.read_csv("Data\\dataset.csv")

# Preprocessing
# Handling missing values
imputer = SimpleImputer(strategy='most_frequent')
data['Dependents'].replace('3+', 3, inplace=True)  # replace '3+' with 3
data[['Dependents', 'SelfEmployed', 'LoanAmountTerm']] = imputer.fit_transform(data[['Dependents', 'SelfEmployed', 'LoanAmountTerm']])
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())
data['CreditHistory'] = data['CreditHistory'].fillna(data['CreditHistory'].mode()[0])

# Encoding categorical variables
label_encoders = {}
categorical_cols = ['Gender', 'Married', 'Education', 'SelfEmployed', 'PropertyArea', 'LoanStatus']
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Feature scaling
scaler = StandardScaler()
data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'LoanAmountTerm']] = scaler.fit_transform(data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'LoanAmountTerm']])

# Split data into train and test sets
X = data.drop(['Loan_ID', 'LoanStatus'], axis=1)
y = data['LoanStatus']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train, y_train)

# Evaluate model
y_pred = knn.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Predicted No', 'Predicted Yes'])
plt.yticks([0, 1], ['Actual No', 'Actual Yes'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment='center', color='white' if conf_matrix[i, j] > conf_matrix.max()/2 else 'black')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# Plot ROC Curve
y_probs = knn.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()


# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Sensitivity (True Positive Rate)
sensitivity = recall

# Specificity (True Negative Rate)
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

# Print metrics
print("Accuracy:", accuracy)
print("F1-score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

# Plot metrics
metrics_names = ['Accuracy', 'F1-score', 'Precision', 'Recall', 'Sensitivity', 'Specificity']
metrics_values = [accuracy, f1, precision, recall, sensitivity, specificity]

plt.figure(figsize=(10, 6))
plt.bar(metrics_names, metrics_values, color='darkorange')
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Evaluation Metrics')
plt.ylim(0, 1)
plt.show()

# Save the trained KNN model to a file
joblib.dump(knn, 'Models\\knn_model_WithoutModification.pkl')
