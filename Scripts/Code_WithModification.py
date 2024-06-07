import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from collections import Counter
import joblib



# Define custom distance function
def custom_distance(x1, x2):
    x1_numeric = np.asarray(x1, dtype=np.float64)
    x2_numeric = np.asarray(x2, dtype=np.float64)
    x1_clean = x1_numeric[~np.isnan(x1_numeric)]
    x2_clean = x2_numeric[~np.isnan(x2_numeric)]
    sum_of_squares = np.sum((x1_clean - x2_clean)**2)
    sum_of_quadratics = np.sum((x1_clean - x2_clean)**4)
    distance = 10**((np.log(np.exp(1)) * (np.log(np.abs(sum_of_quadratics)) - np.log(np.abs(sum_of_squares)))) / (2 * np.log(10)))
    return distance


# Define KNN with custom distance and decision function
class CustomKNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [custom_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return max(set(k_nearest_labels), key=k_nearest_labels.count)  # Return the most common label

    def decision_scores(self, X):
        decision_scores = []
        for x in X:
            distances = [custom_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            decision_score = Counter(k_nearest_labels)[1] / self.n_neighbors  # Ratio of positive labels in neighbors
            decision_scores.append(decision_score)
        return decision_scores

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

# Train KNN model with custom distance
knn_custom = CustomKNN(n_neighbors=5)
knn_custom.fit(X_train.values, y_train.values)

# Evaluate model
y_pred_custom = knn_custom.predict(X_test.values)
conf_matrix_custom = confusion_matrix(y_test, y_pred_custom)

# Plot Confusion Matrix for custom KNN
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix_custom, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Custom KNN Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Predicted No', 'Predicted Yes'])
plt.yticks([0, 1], ['Actual No', 'Actual Yes'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(conf_matrix_custom[i, j]), horizontalalignment='center', color='white' if conf_matrix_custom[i, j] > conf_matrix_custom.max()/2 else 'black')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# Calculate metrics for custom KNN
accuracy_custom = accuracy_score(y_test, y_pred_custom)
f1_custom = f1_score(y_test, y_pred_custom)
precision_custom = precision_score(y_test, y_pred_custom)
recall_custom = recall_score(y_test, y_pred_custom)
sensitivity_custom = recall_custom
specificity_custom = conf_matrix_custom[0, 0] / (conf_matrix_custom[0, 0] + conf_matrix_custom[0, 1])



# Plot ROC Curve
# Use decision scores to compute ROC curve
decision_scores_custom = knn_custom.decision_scores(X_test.values)

# Plot ROC Curve
fpr_custom, tpr_custom, _ = roc_curve(y_test, decision_scores_custom)
roc_auc_custom = auc(fpr_custom, tpr_custom)

plt.figure()
plt.plot(fpr_custom, tpr_custom, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_custom)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# Plot Precision-Recall Curve
# Use decision scores to compute precision-recall curve
precision_custom, recall_custom, _ = precision_recall_curve(y_test, decision_scores_custom)

plt.figure()
plt.plot(recall_custom, precision_custom, color='darkorange', lw=2, label='PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Print metrics for custom KNN
print("Custom KNN Metrics:")
print("Accuracy:", accuracy_custom)
print("F1-score:", f1_custom)
print("Precision:", precision_custom)
print("Recall:", recall_custom)
print("Sensitivity:", sensitivity_custom)
print("Specificity:", specificity_custom)

# Define the metrics and their corresponding values
metrics_names = ['F1-score', 'Accuracy', 'Sensitivity', 'Specificity']
metrics_values = [f1_custom, accuracy_custom, sensitivity_custom, specificity_custom]

# Plot the metrics
plt.figure(figsize=(10, 6))
plt.bar(metrics_names, metrics_values, color='darkorange')
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Evaluation Metrics for Custom KNN')
plt.ylim(0, 1)  # Set y-axis limit to match the range of values for metrics
plt.show()

# Save the trained KNN model to a file
joblib.dump(knn_custom, 'Models\\knn_model_WithModification.pkl')