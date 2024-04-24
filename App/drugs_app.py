# Importa librerías
import gradio as gr
import skops.io as sio

# Se carga el modelo entrenado sin comprobaciones de seguridad
#pipe = sio.load("../Model/drugs.skops", trusted=True)  # Para ejecutar en local
pipe = sio.load("./Model/drugs.skops", trusted=True)    # Para ejecutar en el espacio de 'Hugging Face'

# Función para predecir el medicamento que necesita el paciente
def predict_drug(age, sex, blood_pressure, cholesterol, na_to_k_ratio):
    """Predice los medicamentos basandose en las características de los pacientes.

    Parámetros:
        age (int): Edad del paciente
        sex (str): Sexo del paciente
        blood_pressure (str): Nivel de presión sanguínea del paciente
        cholesterol (str): Nivel de colesteros del paciente
        na_to_k_ratio (float): Ratio de sodio-potasio en sangre del paciente

    Returns:
        str: Medicamento predicho para el paciente
    """

    # Lista con las características de los pacientes (están en el mismo orden que cuando fueron entrenadas).
    # Importante para que tengan el mismo índice que durante el 'pipeline' de entrenamiento
    features = [age, sex, blood_pressure, cholesterol, na_to_k_ratio]

    # Se realiza el pronóstico para un paciente, obteniendo alguno de los medicamentos
    predicted_drug = pipe.predict([features])[0]

    # Se imprime la frase "Medicamento pronosticado: {medicamento_pronosticado}" y se devuelve fuera de la función
    label = f"Medicamento pronosticado: {predicted_drug}"
    return label

# Se crean botones para seleccionar los valores de las variables de entrada usando 'sliders' para los valores continuos
# (edad y ratio de sodio/potasio) y botones 'Radio' para los valores categóricos (sexo, presión sanguínea y nivel de
# colesterol)
# - Los 'sliders' tienen un rango que va desde el valor mínimo hasta el máximo en los pasos establecidos
# - Los botones 'radio' tienen una serie de opciones de las que solo una puede seleccionarse.
# En el parámetro 'choices' se definen las elecciones a elegir en forma de tuplas '(nombre, valor)', de forma que
# 'nombre' representa el texto que aparece en la interfaz de usuario y 'valor' es el valor que realmente tiene el
# nombre, y que se debe corresponden con las categorías usadas durante el proceso de entrenamiento del modelo.
inputs = [gr.Slider(minimum=15, maximum=74, step=1, label="Edad"),
          gr.Radio(choices=[("Masculino", "M"), ("Femenino", "F")], label="Sexo"),
          gr.Radio(choices=[("Alta", "HIGH"), ("Baja", "LOW"), ("Normal", "NORMAL")], label="Presión sanguínea"),
          gr.Radio(choices=[("Alto", "HIGH"), ("Normal", "NORMAL")], label="Colesterol"),
          gr.Slider(minimum=6.2, maximum=38.2, step=0.1, label="Relación Sodio/Potasio")]

# Se crea una sección para mostrar la etiqueta de clasificación (el medicamento pronosticado)
outputs = [gr.Label(num_top_classes=1, label="Medicamento elegido")]

# Ejemplos de listas que pasar a la función de predicción
examples = [[32, "M", "NORMAL", "NORMAL", 17.2],
            [31, "F", "LOW", "NORMAL", 10],
            [50, "M", "HIGH", "HIGH", 33]]

# Título de la aplicación web, una breve descripción de sus características y funcionalidad, y un pie de página que
# incluye información sobre la app web
title = "Clasificación de medicamentos para pacientes"
description = "Selecciona las variables para identificar correctamente el medicamento pronosticado para el paciente."
article = "Esta aplicación web es una demo de un proceso de CI/CD para algoritmos de machine learning en el que se " \
          "automatiza el entrenamiento, la evaluación y el despliegue de modelos a Hugging Face usando GitHub Actions."

# Se crea una interfaz de usuario con la función definida para predecir el medicamento, las entradas (debe haber el
# mismo número de entradas que parámetros en la función 'fn') y salidas definidas (debe haber el mismo número de
# salidas que valores retornados en 'fn').
# Se pasa también los ejemplos definidos como ejemplos de la interfaz, además del título, la descripción y el pie de
# página de la aplicación web, seleccionando el tema 'Base' para darle la apariencia a la interfaz
# Una vez se crea la interfaz, se lanza una demo en una URL local con 'launch'
gr.Interface(fn=predict_drug, inputs=inputs, outputs=outputs, examples=examples, title=title, description=description,
             article=article, theme=gr.themes.Base()).launch()
