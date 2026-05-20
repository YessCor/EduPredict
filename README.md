# EduPredict 🎓

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0.0%2B-red.svg)
![Scikit-learn](https://img.shields.io/badge/ScikitLearn-0.24%2B-orange.svg)

**EduPredict** es una plataforma de **predicción de riesgo de deserción estudiantil** usando **Machine Learning**. Permite:
- Visualizar análisis exploratorio del dataset.
- Entrenar y evaluar un modelo (Random Forest).
- Realizar **predicciones en tiempo real** con un simulador.

---

## Objetivo del proyecto
Construir un sistema tipo consultoría de IA que, a partir de variables académicas y socioeducativas, determine si un estudiante está **en riesgo de desertar** para habilitar intervenciones tempranas y mejorar la retención.

**Target (variable objetivo):** `Deserta` (1 = Deserción, 0 = No deserción).

---

## Requisitos e instalación

### 1) Crear entorno (recomendado)
```bash
python -m venv .venv
.4venv\Scripts\activate
```

### 2) Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## Cómo ejecutar

### Ejecutar la aplicación (Streamlit)
```bash
streamlit run dashboard.py
```

La app abre en:
- `http://localhost:8501`

---

## Estructura del proyecto

```
EduPredict/
├─ dashboard.py            # App principal en Streamlit (UI, gráficos, simulador y métricas)
├─ pipeline_inteligente.py # Pipeline en Python (carga, limpieza, transformación, entrenamiento y visualización)
├─ proyecto_3.py          # Notebook exportado (prototipo/experimentos y explicación)
├─ dataset.csv            # Dataset base (1000 registros aprox.)
├─ requirements.txt       # Dependencias
└─ README.md              # Documentación del repositorio
```

---

## Dataset (`dataset.csv`)

El archivo `dataset.csv` contiene columnas:
- `Promedio`: GPA / promedio anterior (numérico, escala 0–5)
- `Fallas`: número de ausencias o fallas (numérico)
- `Horas_Estudio`: horas de estudio semanales (numérico)
- `Internet`: acceso a Internet en casa (`Si` / `No`)
- `Trabaja`: el estudiante trabaja a tiempo parcial (`Si` / `No`)
- `Deserta`: variable objetivo (`1` = Sí deserta, `0` = No)

### Codificación usada
En el código se utiliza `LabelEncoder` para transformar:
- `Internet` (`Si`/`No` → 0/1)
- `Trabaja` (`Si`/`No` → 0/1)

**Importante:** en `dashboard.py` se usa un preprocesamiento ligero solo para `Internet` y `Trabaja`.

---

## Descripción por archivo

### 1) `dashboard.py` (Streamlit)

Es el punto de entrada de la interfaz. Configura la UI (CSS y estilos), carga datos, entrena un modelo y muestra cuatro secciones en la barra lateral:

1. **Inicio**
   - Métricas generales: total de estudiantes, tasa de deserción, variables clave.
   - Tabla con el dataset y etiquetas:
     - `Deserta = 0` → **Estable**
     - `Deserta = 1` → **En Riesgo**

2. **Análisis Exploratorio**
   - Gráficos con Plotly:
     - Dispersión `Promedio` vs `Deserta`
     - Dispersión `Fallas` vs `Deserta`
     - Histograma `Horas_Estudio` por clase

3. **Simulador de Riesgo**
   - Sliders/selectores para construir un perfil de estudiante:
     - Promedio, Fallas, Horas_Estudio
     - Internet (Si/No), Trabaja (Si/No)
   - Construye un `DataFrame` con los campos codificados y calcula:
     - `prob = model_rf.predict_proba(...)`
   - Umbral:
     - riesgo si `prob >= 0.5`
     - muestra probabilidad en % y un indicador visual

4. **Métricas del Modelo**
   - `accuracy_score` en el conjunto de prueba (split 70/30)
   - Validación cruzada `cross_val_score` con `cv=5`
   - Importancia de variables con `model_rf.feature_importances_`

#### Modelo usado en el dashboard
- `RandomForestClassifier(random_state=42)`
- Entrenamiento directo sobre las columnas:
  - `Promedio`, `Fallas`, `Horas_Estudio`, `Internet`, `Trabaja`

---

### 2) `pipeline_inteligente.py` (Pipeline automatizado)

Script en Python (tipo pipeline) que orquesta el flujo:

1. **extraer_datos(file_path='dataset.csv')**
   - Carga el dataset desde CSV.

2. **validar_calidad(df)**
   - Revisa nulos.
   - Si existen nulos, elimina filas con `dropna()`.
   - Muestra estadísticas descriptivas.

3. **transformar_datos(df)**
   - Codifica `Internet` y `Trabaja` con `LabelEncoder`.
   - Crea una feature adicional:
     - `Indice_Riesgo = Fallas / (Promedio + 0.1)`

4. **entrenar_evaluar_modelo(df)**
   - Separa `X` y `y` (target: `Deserta`).
   - Split train/test 70/30 (`random_state=42`).
   - Entrena `RandomForestClassifier(n_estimators=100, random_state=42)`.
   - Reporta:
     - accuracy en prueba
     - accuracy promedio por validación cruzada (cv=5)
     - `classification_report`

5. **generar_visualizaciones(...)**
   - Matriz de confusión
   - Importancia de características
   - Boxplot Promedio por clase
   - Violinplot Fallas por clase
   - Guarda figura en `pipeline_results.png`

6. **main()**
   - Ejecuta todo el pipeline con manejo de excepciones.

---

### 3) `proyecto_3.py` (Notebook exportado / experimentos)

Archivo que contiene el contenido de un notebook exportado desde Colab.

Incluye:
- Descripción del objetivo y variables del dataset.
- Proceso de preprocesamiento con `LabelEncoder`.
- Entrenamiento y evaluación inicial (por ejemplo `DecisionTreeClassifier`).
- Ejemplos de visualizaciones (Promedio vs Deserción, Fallas vs Deserción).
- Validación cruzada.
- Comparación entre pipelines (Decision Tree vs Random Forest) y evaluación.

> Nota: este archivo es principalmente un **prototipo/explicación**; el comportamiento del sistema integrado corre en `dashboard.py` y el pipeline en `pipeline_inteligente.py`.

---

## Dependencias (`requirements.txt`)

El proyecto utiliza principalmente:
- `streamlit`
- `pandas`, `numpy`
- `scikit-learn`
- `plotly`
- (y en el pipeline) librerías para visualización como `matplotlib` y `seaborn`

Para instalar:
```bash
pip install -r requirements.txt
```

---

## Consideraciones técnicas / notas

1. **Codificación de categorías**
   - Tanto `dashboard.py` como `pipeline_inteligente.py` convierten `Internet` y `Trabaja` a numérico.

2. **Feature extra en pipeline**
   - `pipeline_inteligente.py` agrega `Indice_Riesgo`.
   - El dashboard **no usa** esa feature extra (usa el conjunto original sin esa expansión).

3. **Umbral de decisión del simulador**
   - Se considera riesgo cuando `P(Deserta=1) >= 0.5`.

4. **Tamaño de dataset**
   - Con datasets pequeños, pueden aparecer señales de sobreajuste; por eso el proyecto usa validación cruzada.

---

## Cómo usar el proyecto (resumen)
1. Instala dependencias.
2. Ejecuta el dashboard con Streamlit.
3. Usa las secciones:
   - Inicio (estado del dataset)
   - Análisis Exploratorio (gráficos)
   - Simulador de Riesgo (predicción)
   - Métricas del Modelo (accuracy, CV, importancias)

---

## Créditos
- [Yessid Cordero](https://github.com/YessCor)
- [Bleidys Larios](https://github.com/bleidys16)
- [Melany Tesillo](https://github.com/mptse)
- [Emily Monterrosa](https://github.com/emilymontec)

---

## Soporte
Para dudas o problemas, revisa los archivos del proyecto y abre un issue en tu repositorio si lo estás usando con GitHub.

