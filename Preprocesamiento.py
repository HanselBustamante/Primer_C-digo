import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# Cargar el dataset desde un archivo CSV
df = pd.read_csv('proy.csv')

# Preprocesamiento:

# 1. Verificar si hay valores nulos en las columnas
print("Valores nulos por columna:")
print(df.isnull().sum())

# 2. Convertir columnas categóricas a tipo 'category' y codificarlas
label_encoder = LabelEncoder()

# Columnas categóricas a codificar
categorical_columns = ['equipo', 'formacion', 'zona_controlada', 'categoria_jugador']

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# 3. Normalización de columnas numéricas
scaler = StandardScaler()
df[['posicion_x', 'posicion_y', 'velocidad', 'aceleracion', 'frecuencia_cardiaca', 
    'distancia_al_balon', 'posesion_equipo', 'presion_rival']] = scaler.fit_transform(
    df[['posicion_x', 'posicion_y', 'velocidad', 'aceleracion', 'frecuencia_cardiaca', 
        'distancia_al_balon', 'posesion_equipo', 'presion_rival']])

# 4. Verificar la distribución de categorías en 'categoria_jugador'
print("\nDistribución de 'categoria_jugador':")
print(df['categoria_jugador'].value_counts())

# Balanceo de datos (si es necesario):
# 4.1 Técnicas de balanceo:
# Verificar si la clase está desbalanceada, si es necesario aplicar técnicas de balanceo

# Separar las clases mayoritaria y minoritaria
df_minority = df[df['categoria_jugador'] == 2]  # 'Reservista' codificado como 2
df_majority = df[df['categoria_jugador'] != 2]

# Realizar undersampling de la clase mayoritaria (si se desea)
df_majority_undersampled = resample(df_majority, 
                                    replace=False,    # No reemplazar
                                    n_samples=len(df_minority),  # Igualar el tamaño
                                    random_state=42)  # Para reproducibilidad

# Combinar las clases balanceadas
df_balanced = pd.concat([df_majority_undersampled, df_minority])

# O aplicar SMOTE para sobremuestreo (si se desea)
smote = SMOTE(random_state=42)
X = df.drop('categoria_jugador', axis=1)  # Características
y = df['categoria_jugador']  # Etiqueta

X_resampled, y_resampled = smote.fit_resample(X, y)

# Crear DataFrame balanceado con SMOTE
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['categoria_jugador'] = y_resampled

# Mostrar las primeras filas del DataFrame balanceado
print("\nPrimeras filas del DataFrame balanceado:")
print(df_resampled.head())

# Guardar en un archivo CSV (si lo deseas)
df_resampled.to_csv('proy_balanceado.csv', index=False)
