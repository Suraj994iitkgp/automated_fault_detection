# Experiment 1: Default models  
- Random Forest (n_estimators=100, max_depth=5): Accuracy=0.995  
- SVM (rbf, C=1): Accuracy=0.985  
- Logistic Regression (max_iter=200): Accuracy=0.990
- Data Used: fault_data
- Best Model Choosen: RF

# Model Validation
Training Accuracy: 1.0
Test Accuracy: 0.9949238578680203
Classification Report:
               precision    recall  f1-score   support

          -1       0.99      1.00      0.99        88
           1       1.00      0.99      1.00       109

    accuracy                           0.99       197
   macro avg       0.99      1.00      0.99       197
weighted avg       0.99      0.99      0.99       197

Confusion Matrix:
 [[ 88   0]
 [  1 108]]
ROC-AUC Score: 0.9998957464553795