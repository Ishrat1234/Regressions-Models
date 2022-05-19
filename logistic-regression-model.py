import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
cancer = datasets.load_breast_cancer()
train_x, test_x, train_y, test_y = train_test_split(cancer.data,cancer.target, test_size=0.2, random_state=42)
lr = LogisticRegression()
lr.fit(train_x, train_y)
pred_y = lr.predict(test_x)
print("Classification Report:\n",classification_report(test_y, pred_y))
print("\nConfusion Matrix:\n",confusion_matrix(test_y, pred_y))
print("\nAccuracy:",accuracy_score(test_y, pred_y))
cm=confusion_matrix(test_y, pred_y)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=lr.classes_)
disp.plot()
plt.show()
