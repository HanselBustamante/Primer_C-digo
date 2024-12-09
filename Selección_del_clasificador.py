import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Cargar el dataset balanceado (proveniente de la etapa anterior)
df_resampled = pd.read_csv('proy_balanceado.csv')

# Seleccionar características (X) y etiqueta (y)
X = df_resampled.drop('categoria_jugador', axis=1)  # Características
y = df_resampled['categoria_jugador']  # Etiqueta

# Dividir los datos en entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar el clasificador Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el clasificador
clf.fit(X_train, y_train)

# Predecir las etiquetas para el conjunto de prueba
y_pred = clf.predict(X_test)

# Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Matriz de confusión
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# Reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
