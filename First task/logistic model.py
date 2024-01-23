# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, precision_recall_curve, roc_curve
from sklearn.datasets import make_classification

## Generate a dataset for a binary classification task:
X, y = make_classification(
    n_samples = 2000,
    n_features = 2,
    n_informative = 2,
    n_redundant = 0,
    n_repeated = 0,
    n_classes = 2,
    n_clusters_per_class = 1,
    weights = (0.20, 0.80),
    class_sep = 6.0,
    hypercube = False,
    random_state = 2,
)
Train_X, Test_x, Train_y, Test_y = train_test_split(X, y, test_size=0.35, random_state=42)

## Implementation of the logistic regression method:
lr = LogisticRegression()
lr.fit(Train_X, Train_y)
lr_pred = lr.predict(Test_x)
print("Логистическая регрессия:")
lr_accuracy = accuracy_score(Test_y, lr_pred)
lr_conf_matrix = confusion_matrix(Test_y, lr_pred)
lr_precision = precision_score(Test_y, lr_pred)
lr_recall = recall_score(Test_y, lr_pred)
lr_f1 = f1_score(Test_y, lr_pred)
lr_avg_precision = average_precision_score(Test_y, lr_pred)
lr_roc_auc = roc_auc_score(Test_y, lr_pred)
print(f"Верные ответы: {lr_accuracy}")
print(f"Ошибки:\n{lr_conf_matrix}")
print(f"Точность: {lr_precision}")
print(f"Полнота: {lr_recall}")
print(f"F-мера: {lr_f1}")
print(f"Средняя точность: {lr_avg_precision}")
print(f"ROC-кривая: {lr_roc_auc}")

## Visualization of the PR curve:
lr_precision, lr_recall, _ = precision_recall_curve(Test_y, lr.decision_function(Test_x))
plt.figure(figsize=(10, 6))
plt.plot(lr_recall, lr_precision, label=f'Логистическая регрессия (AP={lr_avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR-кривая')
plt.legend()
plt.show()

## Visualization of the ROC curve:
lr_fpr, lr_tpr, _ = roc_curve(Test_y, lr.decision_function(Test_x))
plt.figure(figsize=(10, 6))
plt.plot(lr_fpr, lr_tpr, label=f'Логистическая регрессия (AUC={lr_roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend()
plt.show()

