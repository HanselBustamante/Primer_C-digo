import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Cargar el dataset balanceado (proveniente de la etapa anterior)
df_resampled = pd.read_csv('proy_balanceado.csv')

# Seleccionar características (X) y etiqueta (y)
X = df_resampled.drop('categoria_jugador', axis=1)  # Características
y = df_resampled['categoria_jugador']  # Etiqueta

# Lista para almacenar las confiabilidades
accuracies = []

# Realizar 100 ejecuciones con splits 50/50
for _ in range(100):
    # Dividir los datos en entrenamiento y prueba (50% entrenamiento, 50% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=None)  # random_state=None para variabilidad

    # Inicializar el clasificador Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Entrenar el clasificador
    clf.fit(X_train, y_train)

    # Predecir las etiquetas para el conjunto de prueba
    y_pred = clf.predict(X_test)

    # Calcular la confiabilidad (accuracy)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Calcular la mediana de las confiabilidades
median_accuracy = np.median(accuracies)

# Mostrar la mediana de las confiabilidades
print(f'La mediana de las confiabilidades (accuracy) es: {median_accuracy * 100:.2f}%')

# Opcional: Ver la distribución de las confiabilidades
import matplotlib.pyplot as plt

plt.hist(accuracies, bins=20, edgecolor='black')
plt.title('Distribución de las Confiabilidades (Accuracy) - 100 Splits')
plt.xlabel('Accuracy')
plt.ylabel('Frecuencia')
plt.show()
