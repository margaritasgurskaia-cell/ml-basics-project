import pandas as pd
plants = pd.read_csv("../data/train.csv", sep=";")

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

y_true = [1, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 1, 0]

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print("TP:", tp)
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)

cm = confusion_matrix(y_true, y_pred)

print(cm) 

accuracy = (tn+tp)/(tn+tp+fp+fn)
print(accuracy)

precision = tp/(tp+fp)
print(precision)

recall = tp/(fn+tp)
print(recall)

F1 = 2*((precision*recall)/(precision+recall))
print(F1)

features = plants[["height_cm", "leaf_count"]]
labels = plants["species"]

X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=42
)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(predictions)

#Використала вспоміжні джерела, щоб зрозуміти де помилка, як її виправити і щось додати до коду, щоб він працював (наприклад: from sklearn.neighbors import KNeighborsClassifier, from sklearn.metrics import confusion_matrix і тд...)