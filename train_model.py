import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data_file = "data.txt"
# Carga info del .txt
data = np.loadtxt(data_file)

# Separa la info entre info real y etiquetas
X = data[:, :-1]  # Features are all columns except the last one
y = data[:, -1]   # Labels are the last column

# Separa en entrenamiento y testeo
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    shuffle=True,
                                                    stratify=y)

#Random Forest Classifier
rf_classifier = RandomForestClassifier()

# Entrenamiento en la info
rf_classifier.fit(X_train, y_train)

# Prediccion en testeo
y_pred = rf_classifier.predict(X_test)

# Evaluacion de la exactitud del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(confusion_matrix(y_test, y_pred))

with open('./model', 'wb') as f:
    pickle.dump(rf_classifier, f)
