# Modelo de Analisis de Sentimientos

**Ingenería de Datos y Software**
**Universidad San Buenaventura Medellín**  
**Electiva II**
**Mayo 2025**

**Integrantes:**
- Juan Esteban Navia Perez
- Santiago José Gomez Agudelo
- Jordanny Botero Guzman

---

## Introducción

### Problema
Los comentarios y opiniones son una fuente valiosa de información sobre productos, servicios y experiencias. Sin embargo, analizar manualmente grandes volúmenes de texto para determinar el sentimiento predominante es ineficiente y propenso a errores. Existe la necesidad de contar con herramientas automatizadas que puedan clasificar de manera precisa y eficiente el sentimiento expresado en textos.

### Objetivos
1. Desarrollar un modelo de aprendizaje automático capaz de clasificar textos en múltiples categorías de sentimientos.
2. Implementar una API para utilizar el modelo entrenado.
3. Crear una interfaz gráfica de usuario intuitiva para que los usuarios finales puedan analizar textos fácilmente.
4. Integrar el modelo, la API y la GUI en un sistema completo y funcional.

### Resumen de la Solución
Este proyecto desarrolla un sistema completo de análisis de sentimientos que utiliza técnicas de procesamiento de lenguaje natural y aprendizaje automático para clasificar textos en varias categorías emocionales como felicidad, tristeza, amor, enojo, entre otras. El sistema está construido con Django y cuenta con:
- Un modelo de aprendizaje automático basado en Random Forest
- Una API integrada para procesar solicitudes de análisis
- Una interfaz web amigable que permite a los usuarios enviar textos para su análisis
- Retroalimentación personalizada basada en la emoción detectada

---

## Desarrollo del Modelo ML

### Datos
El modelo se entrenó utilizando dos conjuntos de datos principales:
1. **tweet_emotions.csv**: Una colección de tweets etiquetados con diferentes emociones.
2. **emoji_sentiment_data.csv**: Datos adicionales que relacionan emojis con estados emocionales.

Estos conjuntos contienen textos clasificados en 12 categorías de sentimientos:
- Happiness (Felicidad)
- Sadness (Tristeza)
- Love (Amor)
- Anger (Enojo)
- Surprise (Sorpresa)
- Fear (Miedo)
- Neutral (Neutral)
- Enthusiasm (Entusiasmo)
- Worry (Preocupación)
- Fun (Diversión)
- Hate (Odio)
- Boredom (Aburrimiento)

### Preprocesamiento
El preprocesamiento de los datos se realizó a través de varias etapas:
1. **Limpieza de texto**:
   - Eliminación de URLs
   - Eliminación de menciones y hashtags
   - Eliminación de dígitos
   - Conversión a minúsculas
   - Eliminación de espacios en blanco adicionales

2. **Manejo de emojis**:
   - Conversión de emojis a representaciones textuales utilizando la biblioteca `emoji`

3. **Preparación de datos**:
   - Combinación de los conjuntos de datos de tweets y emojis
   - Eliminación de registros con valores nulos o vacíos
   - División de los datos en conjuntos de entrenamiento (80%) y validación (20%)

### Entrenamiento
El modelo fue desarrollado utilizando un pipeline de scikit-learn con los siguientes componentes:
1. **Vectorización**: TF-IDF con un máximo de 10,000 características y n-gramas de 1 a 2 palabras
2. **Escalado**: StandardScaler para normalizar las características
3. **Clasificación**: Algoritmo Random Forest con 100 estimadores y pesos de clase balanceados

El proceso de entrenamiento incluye:
- Codificación de etiquetas de sentimiento
- Ajuste del modelo usando el conjunto de entrenamiento
- Serialización del modelo y codificador para uso futuro

### Métricas de Evaluación
El modelo se evaluó utilizando las siguientes métricas:
- Precisión (Precision)
- Exhaustividad (Recall)
- Puntuación F1 (F1-score)
- Soporte (Support)

Los resultados se generan utilizando scikit-learn's `classification_report` para mostrar el rendimiento por clase.

### Resultados Finales
El modelo entrenado fue capaz de identificar correctamente múltiples sentimientos en los textos de validación, con mejor rendimiento en categorías como "happiness", "love" y "anger", que suelen tener patrones lingüísticos más distintivos. Las categorías más ambiguas como "neutral" presentaron desafíos mayores para el clasificador.

El modelo final se serializó y guardó como `sentiment_model.pkl`, que incluye tanto el pipeline de procesamiento como el codificador de etiquetas.

---

## Desarrollo de la API

### Diseño
La API para el análisis de sentimientos está integrada directamente en la aplicación Django, siguiendo un enfoque RESTful simplificado:

**Endpoints**:
- `/analyze/`: Procesa solicitudes POST con textos para analizar

**Formato JSON**:
- **Entrada**: `{ "comment": "texto a analizar" }`
- **Salida**: `{ "sentiment": "categoría", "feedback": "respuesta personalizada" }`

### Implementación
La implementación se basa en las vistas de Django que manejan tanto solicitudes web como API:
- `SentimentAnalysisView`: Clase basada en FormView que procesa formularios web y devuelve resultados
- `SentimentPredictor`: Clase que encapsula la lógica de carga del modelo y predicción

### Carga y Uso del Modelo
El modelo se carga al inicializar el predictor:
1. Busca el archivo `sentiment_model.pkl` en la ubicación predefinida
2. Deserializa el pipeline del modelo y el codificador de etiquetas
3. Utiliza estos componentes para hacer predicciones sobre nuevos textos
4. Transforma las predicciones numéricas en etiquetas de sentimiento legibles
5. Agrega una respuesta personalizada según la emoción detectada

---

## Desarrollo de la GUI

### Diseño
La interfaz gráfica de usuario es una aplicación web responsiva construida con Django y Bootstrap. Se compone de:
- Página principal que introduce el propósito del sistema
- Formulario de análisis de sentimientos
- Visualización de resultados con código de colores según la emoción detectada

### Flujo de Usuario
1. El usuario ingresa a la página principal
2. Navega al formulario de análisis 
3. Ingresa un texto en el campo de comentario
4. Envía el formulario para análisis
5. Recibe resultados visuales que muestran:
   - El comentario analizado
   - La categoría de sentimiento detectada
   - Una respuesta personalizada basada en la emoción

### Interacción con la API
La GUI interactúa con el backend a través del patrón MVT (Modelo-Vista-Plantilla) de Django:
1. La vista procesa el formulario enviado
2. Llama al predictor que utiliza el modelo cargado
3. Recibe los resultados como un diccionario
4. Renderiza la plantilla con el contexto actualizado

Las plantillas utilizan un sistema de clases CSS para representar visualmente diferentes emociones (positivas, negativas, neutrales) con colores distintos.

---

## Arquitectura del Sistema

### Diagrama de Arquitectura

```
┌─────────────────┐    ┌───────────────────┐    ┌─────────────────┐
│                 │    │                   │    │                 │
│  Usuario (Web)  │◄───┤  Django Frontend  │◄───┤  Django Views   │
│                 │    │  (Templates)      │    │  (Controller)   │
└─────────────────┘    └───────────────────┘    └───────┬─────────┘
                                                        │
                                                        ▼
                               ┌───────────────────────────────────┐
                               │                                   │
                               │        SentimentPredictor         │
                               │                                   │
                               └───────────────┬───────────────────┘
                                               │
                                               ▼
                               ┌───────────────────────────────────┐
                               │                                   │
                               │     Trained ML Model Pipeline     │
                               │     (Random Forest + TF-IDF)      │
                               │                                   │
                               └───────────────────────────────────┘
```

### Flujo de Datos
1. **Entrada**: El usuario ingresa texto a través de la interfaz web
2. **Procesamiento**:
   - El controlador (vistas de Django) recibe la entrada
   - El predictor carga el modelo ML si aún no está cargado
   - El texto se procesa utilizando el mismo pipeline de preprocesamiento usado durante el entrenamiento
   - El modelo genera una predicción de sentimiento
   - Se añade una respuesta personalizada basada en la emoción detectada
3. **Salida**: Los resultados del análisis se presentan al usuario a través de la interfaz gráfica con retroalimentación visual

---

## Instrucciones de Despliegue/Ejecución

### Requisitos
- Python 3.10+ 
- Bibliotecas especificadas en `requirements.txt`

### Instalación
1. Clone el repositorio 
2. Cree un entorno virtual:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Instale las dependencias:
   ```
   pip install -r requirements.txt
   ```

### Ejecución
1. Asegúrese de que el modelo esté entrenado:
   ```
   cd train
   python train.py
   ```

2. Inicie el servidor Django:
   ```
   python manage.py runserver
   ```

3. Acceda a la aplicación en su navegador: `http://127.0.0.1:8000/`

---

## Discusión

### Desafíos Encontrados
1. **Preprocesamiento de texto**: La limpieza efectiva de los datos de redes sociales presentó retos debido a la naturaleza informal del lenguaje, abreviaturas y emojis. Adicionalmente, el dataset utilizado limita su uso debido al lenguaje.
   
2. **Desbalance de clases**: Algunas categorías de sentimientos estaban sobrerrepresentadas, lo que requirió técnicas de balanceo como pesos de clase.

3. **Manejo de emojis**: La interpretación correcta de los emojis fue crucial para mejorar la precisión del modelo.

### Limitaciones
1. **Idioma**: El sistema está optimizado para textos en inglés; el soporte multilingüe requeriría modelos adicionales.

2. **Contexto limitado**: El modelo no considera el contexto completo o el sarcasmo, lo que puede llevar a interpretaciones erróneas.

### Consideraciones Éticas
1. **Sesgos**: Los modelos de ML pueden heredar sesgos presentes en los datos de entrenamiento, lo que podría llevar a interpretaciones injustas de ciertos tipos de contenido.

2. **Privacidad**: El análisis de sentimientos debe realizarse con consentimiento y transparencia sobre cómo se utilizarán los datos.

3. **Interpretación**: Los resultados deben tratarse como estimaciones y no como verdades absolutas sobre el estado emocional de un autor.

### Posibles Mejoras
1. **Escalabilidad**: 
   - Implementación de una cola de mensajes para procesar solicitudes a gran escala
   - Contenerización con Docker para facilitar el despliegue

2. **Monitorización**: 
   - Incorporación de logging para rastrear predicciones y rendimiento del modelo
   - Paneles de control para visualizar métricas de uso y precisión

3. **Mejoras técnicas**:
   - Explorar modelos basados en transformers como BERT para mejorar la precisión
   - Implementar aprendizaje continuo para adaptar el modelo a nuevos datos
   - Desarrollar un API separada con FastAPI para mayor rendimiento
   - Añadir análisis de sentimientos a nivel de aspectos para mayor granularidad

4. **Interfaz de usuario**:
   - Visualizaciones más avanzadas de los resultados
   - Soporte para análisis por lotes de múltiples textos
   - Historial de análisis para usuarios registrados
