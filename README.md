# Medications-CICD

En este repositorio se realiza un entrenamiento de un algoritmo de `machine learning` para intentar predecir el mejor medicamento para una serie de pacientes. 

Se automatiza mediante dos flujos de trabajo en `GitHub Actions` los procesos de integración continua (`CI`) y despliegue continuo (`CD`):

- Cada vez que se hace un `push` en la rama `master`, o una solicitud de un `pull request`, se ejecuta el flujo de trabajo de `CI` en el que aparte de configurar e instalar los requisitos de librerías se ejecuta el archivo `train.py` que realiza el entrenamiento y la evaluación del algoritmo `Random Forest` con los datos de los pacientes, creándose automáticamente un mensaje en el `commit` o en el hilo del `pull request` con las métricas y la matriz de confusión obtenidas en el entrenamiento del flujo de trabajo y actualizando automáticamente la rama `update` de GitHub con estas métricas y con el nuevo modelo obtenido.


- Cuando se completa el flujo de trabajo de `CI`, se ejecuta el flujo de `CD` en el que mediante un `token` creado en nuestra cuenta de `Hugging Face` se suben las carpetas de la aplicación web, las métricas y el modelo entrenado en el espacio de `Hugging Face` creado para el proyecto. De esta forma, se tiene una aplicación web hospedada gratuitamente en este espacio donde se pueden seleccionar los valores de las características de entrada de los pacientes para ver que medicamento se pronostica para cada uno de ellos.


## Estructura

- **.github/workflows**: Aquí se recopilan los flujos de trabajo que contienen los trabajos de integración continua y despliegue continuo.


- **App**: En esta carpeta se encuentra el archivo de la aplicación web del clasificador, el `README.md` con metadatos de la aplicación y el archivo `requirements.txt` que sirve para instalar las librerías necesarias para el funcionamiento de la aplicación web.


- **Data**: Carpeta donde se recopilan los datos de los pacientes.


- **Model**: Aquí se recopila el modelo entrenado en formato `skops`.


- **Results**: En este directorio se guardan las métricas de `test` en un archivo de texto y la matriz de confusión.