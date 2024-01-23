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
Train_X, Test_x, Train_y, Test_y= train_test_split(X, y, test_size=0.35, random_state=42)

## Implementing the k-nearest neighbors method:
knn = KNeighborsClassifier()
knn.fit(Train_X, Train_y)
knn_pred = knn.predict(Test_x)
print("\nМетод k ближайших соседей:")
knn_accuracy = accuracy_score(y_test, knn_pred)
knn_conf_matrix = confusion_matrix(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred)
knn_recall = recall_score(y_test, knn_pred)
knn_f1 = f1_score(y_test, knn_pred)
knn_avg_precision = average_precision_score(y_test, knn_pred)
knn_roc_auc = roc_auc_score(y_test, knn_pred)
print(f"Верные ответы: {knn_accuracy}")
print(f"Ошибки:\n{knn_conf_matrix}")
print(f"Точность: {knn_precision}")
print(f"Полнота: {knn_recall}")
print(f"F-мера: {knn_f1}")
print(f"Средняя точность: {knn_avg_precision}")
print(f"ROC-кривая: {knn_roc_auc}")
knn_precision, knn_recall, _ = precision_recall_curve(y_test, knn.predict_proba(Test_x)[:, 1])

## Visualization of the PR curve:
plt.figure(figsize=(10, 6))
plt.plot(knn_recall, knn_precision, label=f'k ближайших соседей (AP={knn_avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR-кривая')
plt.legend()
plt.show()

## Visualization of the ROC curve:
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn.predict_proba(Test_x)[:, 1])
plt.figure(figsize=(10, 6))
plt.plot(knn_fpr, knn_tpr, label=f'k ближайших соседей (AUC={knn_roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend()
plt.show()
