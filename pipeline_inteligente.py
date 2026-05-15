# -*- coding: utf-8 -*-
"""
EduPredict - Pipeline Inteligente de Detección de Deserción
Este script automatiza el flujo completo: desde la carga de datos hasta la evaluación del modelo.
"""

# ==========================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 2. FUNCIONES DE EXTRACCIÓN
# ==========================================
def extraer_datos(file_path="dataset.csv"):
    """Carga el conjunto de datos desde un archivo CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: No se encontró el archivo {file_path}")
    
    print(f"Cargando datos desde: {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Dataset cargado exitosamente. Registros: {len(df)}")
    return df

# ==========================================
# 3. FUNCIONES DE TRANSFORMACIÓN Y EXPANSIÓN
# ==========================================
def transformar_datos(df):
    """Realiza la limpieza y codificación de variables categóricas."""
    df_transformed = df.copy()
    
    # Codificación de variables categóricas
    le = LabelEncoder()
    if "Internet" in df_transformed.columns:
        df_transformed["Internet"] = le.fit_transform(df_transformed["Internet"])
    if "Trabaja" in df_transformed.columns:
        df_transformed["Trabaja"] = le.fit_transform(df_transformed["Trabaja"])
        
    # Expansión (Ejemplo: Creación de un índice de riesgo académico simple)
    # Ratio de Fallas por Promedio (a mayor ratio, mayor riesgo hipotético)
    df_transformed["Indice_Riesgo"] = df_transformed["Fallas"] / (df_transformed["Promedio"] + 0.1)
    
    print("Transformación y expansión de características completada.")
    return df_transformed

# ==========================================
# 4. FUNCIONES DE VALIDACIÓN Y CALIDAD
# ==========================================
def validar_calidad(df):
    """Verifica valores nulos y consistencia básica del dataset."""
    print("\n--- Informe de Calidad de Datos ---")
    nulos = df.isnull().sum().sum()
    print(f"Valores nulos totales: {nulos}")
    
    if nulos > 0:
        print("Advertencia: Se detectaron valores nulos. Procediendo a eliminarlos.")
        df = df.dropna()
    
    print("Estadísticas descriptivas básicas:")
    print(df.describe().round(2))
    return df

# ==========================================
# 5. FUNCIONES DE MODELADO ML
# ==========================================
def entrenar_evaluar_modelo(df):
    """Entrena un modelo de Random Forest y evalúa su desempeño."""
    X = df.drop("Deserta", axis=1)
    y = df["Deserta"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("\nEntrenando modelo RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5)
    
    print(f"Accuracy en prueba: {acc:.4f}")
    print(f"Accuracy promedio (Validación Cruzada): {np.mean(cv_scores):.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    
    return model, X_test, y_test, y_pred

# ==========================================
# 6. FUNCIONES DE VISUALIZACIÓN
# ==========================================
def generar_visualizaciones(df, model, X_test, y_test, y_pred):
    """Genera gráficos de análisis y resultados del modelo."""
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title("Matriz de Confusión")
    axes[0, 0].set_xlabel("Predicción")
    axes[0, 0].set_ylabel("Real")
    
    # 2. Importancia de Características
    importances = pd.Series(model.feature_importances_, index=X_test.columns).sort_values()
    importances.plot(kind='barh', color='teal', ax=axes[0, 1])
    axes[0, 1].set_title("Importancia de las Características")
    
    # 3. Promedio vs Deserción
    sns.boxplot(x='Deserta', y='Promedio', data=df, palette='Set2', ax=axes[1, 0])
    axes[1, 0].set_title("Distribución de Promedio por Estado de Deserción")
    
    # 4. Fallas vs Deserción
    sns.violinplot(x='Deserta', y='Fallas', data=df, palette='Pastel1', ax=axes[1, 1])
    axes[1, 1].set_title("Distribución de Fallas por Estado de Deserción")
    
    plt.tight_layout()
    plt.savefig('pipeline_results.png')
    print("\nVisualizaciones guardadas como 'pipeline_results.png'")
    plt.show()

# ==========================================
# 7. PIPELINE PRINCIPAL (main)
# ==========================================
def main():
    """Orquestador principal del pipeline inteligente."""
    print("==========================================")
    print(" INICIANDO PIPELINE INTELIGENTE EDUPREDICT")
    print("==========================================\n")
    
    try:
        # Paso 1: Extracción
        df_raw = extraer_datos("dataset.csv")
        
        # Paso 2: Validación y Calidad
        df_clean = validar_calidad(df_raw)
        
        # Paso 3: Transformación y Expansión
        df_final = transformar_datos(df_clean)
        
        # Paso 4: Modelado
        model, X_test, y_test, y_pred = entrenar_evaluar_modelo(df_final)
        
        # Paso 5: Visualización
        generar_visualizaciones(df_final, model, X_test, y_test, y_pred)
        
        print("\nPipeline completado exitosamente.")
        
    except Exception as e:
        print(f"\nERROR CRÍTICO en el pipeline: {e}")

# ==========================================
# 8. EJECUCIÓN AUTOMÁTICA
# ==========================================
if __name__ == "__main__":
    main()
