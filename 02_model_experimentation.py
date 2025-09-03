import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 1. Loading Data
df = pd.read_csv('fault_data.csv')

# 2. Selecting Features and Target
X = df.drop('fault_label', axis=1) 
y = df['fault_label']               

# 3. Spliting Data into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Training Models with Basic Hyperparameters
# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print('Random Forest Accuracy:', accuracy_score(y_test, rf_pred))

# SVM
svm = SVC(kernel='rbf', C=1)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print('SVM Accuracy:', accuracy_score(y_test, svm_pred))

# Logistic Regression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print('Logistic Regression Accuracy:', accuracy_score(y_test, lr_pred))

# 5. Save models for later use 
joblib.dump(rf, 'rf_model.pkl')
joblib.dump(svm, 'svm_model.pkl')
joblib.dump(lr, 'lr_model.pkl')
