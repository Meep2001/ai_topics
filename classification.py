from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

X,y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=4,
    n_redundant=0,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, predictions))
print('Recall:', recall_score(y_test, predictions))
print('Confusion Matrix:')
print(confusion_matrix(y_test, predictions))


      
