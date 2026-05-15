# EduPredict 🎓

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0.0+-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24+-orange.svg)

**EduPredict** es un sistema inteligente de predicción de deserción estudiantil que utiliza el Aprendizaje Automático para identificar a los estudiantes en riesgo de abandono escolar de manera temprana y proporcionar recomendaciones de apoyo personalizadas. Esta plataforma permite a las instituciones educativas intervenir de manera proactiva y mejorar las tasas de retención estudiantil.

## ✨ Características

### Panel de Control Integral
- **Real-time Analytics**: Visualizaciones interactivas que muestran las métricas clave de los estudiantes
- **Model Performance Tracking**: Comparación visual de múltiples modelos de ML
- **Data Exploration**: Análisis detallado de los datos de los estudiantes con filtros avanzados

### Núcleo de Aprendizaje Automático
- **Multiple Algorithms**: Compara Regresión Logística, Random Forest, Máquina de Vectores de Soporte, Impulso de Gradiente
- **Cross-Validation**: Asegura la robustez del modelo con validación k-fold
- **Predictive Analytics**: Predice el riesgo de deserción con alta precisión

### Interfaz de Usuario
- **Modern Design**: Diseño moderno, intuitivo y responsivo
- **Easy Navigation**: Navegación simple por barra lateral con secciones claras
- **Quick Predictions**: Predicciones instantáneas con explicaciones detalladas de los resultados

## 🚀 Inicio Rápido

### Requisitos Previos
- Python 3.8+
- pip (Instalador de paquetes de Python)

### Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/YessCor/EduPredict.git
   cd EduPredict
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

### Uso

Iniciar la aplicación:
```bash
streamlit run dashboard.py
```

La aplicación se abrirá automáticamente en su navegador en `http://localhost:8501`.

## 📂 Estructura del Proyecto

```
EduPredict/
├── dashboard.py           # Aplicación principal de Streamlit
├── dataset.csv            # Conjunto de datos de estudiantes (1000 registros)
├── requirements.txt       # Dependencias del proyecto
├── README.md              # Documentación del proyecto
└── .gitignore             # Configuración de Git
```

## 📊 Descripción del Conjunto de Datos (dataset.csv)

El conjunto de datos contiene 1000 registros de estudiantes con las siguientes características:

| Característica | Descripción | Tipo |
|----------------|-------------|------|
| Promedio | GPA anterior (0-5) | Numérico |
| Fallas | Número de ausencias | Numérico |
| Horas_Estudio | Horas de estudio semanales | Numérico |
| Internet | Acceso a Internet en casa | Categórico |
| Trabaja | El estudiante trabaja a tiempo parcial | Categórico |
| Deserta | Estado de deserción (1=Sí, 0=No) | Objetivo |

## 🤖 Modelos de Aprendizaje Automático

El sistema evalúa cuatro modelos de aprendizaje automático:

### 1. Regresión Logística
- Modelo lineal para clasificación binaria
- Rápido e interpretable

### 2. Random Forest
- Ensemble de árboles de decisión
- Alta precisión con importancia de características

### 3. Impulso de Gradiente
- Aprendizaje secuencial en conjunto
- Optimizado para el rendimiento predictivo

### 4. Máquina de Vectores de Soporte (SVM)
- Clasificador de margen máximo
- Efectivo en espacios de alta dimensión

## 📈 Rendimiento del Modelo

El dashboard muestra:
- Puntajes de precisión para cada modelo
- Resultados de validación cruzada
- Análisis de importancia de características
- Curva ROC para cada modelo

## 👥 Miembros

- [Yessid Cordero](https://github.com/YessCor)
- [Bleidys Larios] (https://github.com/bleidys16)
- [Melany Tesillo] (https://github.com/mptse)
- [Emily Monterrosa] (https://github.com/emilymontec)


## 🤝 Contribuciones

Las contribuciones siempre son bienvenidas! Por favor siga los siguientes pasos:

1. Crear una rama feature (`git checkout -b feature/AmazingFeature`)
2. Confirmar los cambios (`git commit -m 'Add some AmazingFeature'`)
3. Push a la rama (`git push origin feature/AmazingFeature`)
4. Abrir un Pull Request

## 📞 Soporte

Para problemas o preguntas, por favor abra un [issue](issues) section.

---

**Desarrollado con ❤️ para la comunidad educativa**