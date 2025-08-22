import pickle
import joblib
import numpy as np
import pandas as pd
import sklearn.ensemble as ek
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset = pd.read_csv('datasets/dataset_1.csv', sep=',', low_memory=False)

# Data preprocessing
X = dataset.drop(['ID', 'md5', 'legitimate'], axis=1).values
y = dataset['legitimate'].values

# Feature selection using ExtraTreesClassifier
extratrees = ek.ExtraTreesClassifier().fit(X, y)
model = SelectFromModel(extratrees, prefit=True)
X_new = model.transform(X)
nbfeatures = X_new.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2)

# Collect feature importance
features = []
index = np.argsort(extratrees.feature_importances_)[::-1][:nbfeatures]
for f in range(nbfeatures):
    features.append(dataset.columns[2 + f])

# Train the RandomForest model
model = ek.RandomForestClassifier(n_estimators=33)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("Accuracy:", (score * 100), '%')

# Save the model and features
joblib.dump(model, "model/model.pkl")
with open('model/features.pkl', 'wb') as f:
    pickle.dump(features, f)

# Confusion matrix
res = model.predict(X_new)
mt = confusion_matrix(y, res)

# Print false positive and false negative rates
print("False positive rate : %f %%" % ((mt[0][1] / float(sum(mt[0]))) * 100))
print("False negative rate : %f %%" % ((mt[1][0] / float(sum(mt[1]))) * 100))

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(nbfeatures), extratrees.feature_importances_[index], align='center')
plt.xticks(range(nbfeatures), [dataset.columns[2 + i] for i in index], rotation=90)
plt.tight_layout()
plt.show()

# Plot confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(mt, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Legitimate', 'Malicious'],
            yticklabels=['Legitimate', 'Malicious'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve and AUC
y_pred_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()