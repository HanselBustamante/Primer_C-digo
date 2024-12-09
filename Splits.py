import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

df_resampled = pd.read_csv('proy_balanceado.csv')

# Seleccionar características (X) y etiqueta (y)
X = df_resampled.drop('categoria_jugador', axis=1)  # Características
y = df_resampled['categoria_jugador']  # Etiqueta

accuracies = []

for _ in range(100):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=None)  

    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

median_accuracy = np.median(accuracies)

print(f'La mediana de las confiabilidades (accuracy) es: {median_accuracy * 100:.2f}%')

import matplotlib.pyplot as plt

plt.hist(accuracies, bins=20, edgecolor='black')
plt.title('Distribución de las Confiabilidades (Accuracy) - 100 Splits')
plt.xlabel('Accuracy')
plt.ylabel('Frecuencia')
plt.show()
