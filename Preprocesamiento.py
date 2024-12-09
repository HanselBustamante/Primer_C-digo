import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

df = pd.read_csv('proy.csv')

print("Valores nulos por columna:")
print(df.isnull().sum())

label_encoder = LabelEncoder()

categorical_columns = ['equipo', 'formacion', 'zona_controlada', 'categoria_jugador']

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

scaler = StandardScaler()
df[['posicion_x', 'posicion_y', 'velocidad', 'aceleracion', 'frecuencia_cardiaca', 
    'distancia_al_balon', 'posesion_equipo', 'presion_rival']] = scaler.fit_transform(
    df[['posicion_x', 'posicion_y', 'velocidad', 'aceleracion', 'frecuencia_cardiaca', 
        'distancia_al_balon', 'posesion_equipo', 'presion_rival']])


print("\nDistribución de 'categoria_jugador':")
print(df['categoria_jugador'].value_counts())

df_minority = df[df['categoria_jugador'] == 2]  # 'Reservista' codificado como 2
df_majority = df[df['categoria_jugador'] != 2]

df_majority_undersampled = resample(df_majority, 
                                    replace=False,   
                                    n_samples=len(df_minority), 
                                    random_state=42)  

df_balanced = pd.concat([df_majority_undersampled, df_minority])

smote = SMOTE(random_state=42)
X = df.drop('categoria_jugador', axis=1)  
y = df['categoria_jugador'] 

X_resampled, y_resampled = smote.fit_resample(X, y)


df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['categoria_jugador'] = y_resampled

print("\nPrimeras filas del DataFrame balanceado:")
print(df_resampled.head())

df_resampled.to_csv('proy_balanceado.csv', index=False)
