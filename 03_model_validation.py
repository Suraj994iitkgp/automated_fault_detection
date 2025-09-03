import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv('fault_data.csv')

# 2. Select Features and Target
X = df.drop('fault_label', axis=1)  # Change as per your actual feature columns
y = df['fault_label']               # Change as per your actual target column

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Best Model (Random Forest example)
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# 5. Training Accuracy
train_pred = rf.predict(X_train)
print("Training Accuracy:", accuracy_score(y_train, train_pred))

# 6. Test Accuracy and Metrics
test_pred = rf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_pred))

print("Classification Report:\n", classification_report(y_test, test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, test_pred))

# 7. ROC-AUC and ROC Curve (For binary classification ONLY)
if len(y.unique()) == 2:
    y_probs = rf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_probs)
    print("ROC-AUC Score:", roc_auc)

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()
else:
    print("ROC-AUC and ROC curve require binary classification target.")

# End of script
