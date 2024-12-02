Comparativa entre Modelos Residual y Siamesa

Este proyecto compara el desempeño de dos arquitecturas de redes neuronales: el modelo residual y el modelo siamesa. Se evalúan ambos modelos en un conjunto de datos técnico, con el objetivo de predecir la potencia basada en variables como radiación y temperatura.

Requisitos del Sistema

Para ejecutar el proyecto, necesitarás:

Python 3.8 o superior

TensorFlow 2.9.0 o superior

Pandas

Numpy

Matplotlib

Scikit-learn

Puedes instalar las dependencias ejecutando:

pip install -r requirements.txt

Archivos Principales

comparativa.py: Contiene la implementación de los modelos, el entrenamiento y la evaluación.

lechuzasdataset.csv: Archivo de datos usado para el entrenamiento y la prueba.

Uso

1. Configuración Inicial

Asegúrate de que el archivo lechuzasdataset.csv se encuentra en el mismo directorio que el archivo comparativa.py. Si no tienes el archivo, verifica que la estructura de columnas sea:

Radiación

Temperatura

Temperatura panel

Potencia (variable objetivo)

2. Ejecución del Código

Para ejecutar el script, usa el siguiente comando:

python comparativa.py

3. Salida

El programa entrenará ambos modelos y mostrará:

Gráficos de las pérdidas durante el entrenamiento (entrenamiento y validación).

Métricas de desempeño para cada modelo:

Error Cuadrático Medio (MSE)

Error Absoluto Medio (MAE)

Coeficiente de Determinación (R²)

Ejemplo de Salida

Residual Model Performance:
MSE: 89514.55
MAE: 104.95
R²: 0.9287

Siamese Model Performance:
MSE: 1833059.19
MAE: 767.55
R²: -0.4592

Comparison of Models:
Residual Model Metrics: (89514.55, 104.95, 0.9287)
Siamese Model Metrics: (1833059.19, 767.55, -0.4592)

4. Modificación del Código

Para ajustar los modelos, puedes modificar las funciones:

build_residual_model: Para cambiar la arquitectura del modelo residual.

build_siamese_model: Para ajustar la arquitectura del modelo siamesa.
