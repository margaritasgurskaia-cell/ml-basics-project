import pandas as pd
plants = pd.read_csv("data/train.csv", sep="\t")

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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