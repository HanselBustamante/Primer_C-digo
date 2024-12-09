import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Cargar el dataset balanceado (proveniente de la etapa anterior)
df_resampled = pd.read_csv('proy_balanceado.csv')

# Seleccionar características (X) y etiqueta (y)
X = df_resampled.drop('categoria_jugador', axis=1)  # Características
y = df_resampled['categoria_jugador']  # Etiqueta

# Dividir los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar el clasificador Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el clasificador
clf.fit(X_train, y_train)

# Predecir las etiquetas para el conjunto de prueba
y_pred = clf.predict(X_test)

# Evaluación del modelo:
# 1. Calcular la confiabilidad (accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# 2. Generar la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# 3. Reporte de clasificación: precisión, recall, F1-score
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Gráfico de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Titular', 'Suplente', 'Reservista'], yticklabels=['Titular', 'Suplente', 'Reservista'])
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()
