# Librerías
import pandas as pd
import skops.io as sio
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Se cargan los datos y se barajan los registros del 'dataset'
drugs_df = pd.read_csv("Data/drugs.csv")
drugs_df = drugs_df.sample(frac=1)

# Se extraen las 'features' del problema y la variable objetivo
X = drugs_df.drop("Drug", axis=1).values
y = drugs_df['Drug'].values

# Se realiza la repartición entrenamiento (80%) y test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Se recopilan las columnas del 'dataframe' en una lista
cols = drugs_df.columns.values.tolist()

# Se dividen las columnas en numéricas, y en aquellas categóricas a las que se les va realizar 'One Hot Encoding' o
# 'Ordinal Encoding' (se almacenan los índices de las columnas, no los nombres de las mismas)
num_col = [i for i, col in enumerate(cols) if col == 'Age' or col == 'Na_to_K']
ohe_col = [i for i, col in enumerate(cols) if col == 'Sex']
ord_col = [i for i, col in enumerate(cols) if col == 'BP' or col == 'Cholesterol']

# Se realiza el 'pipeline' de las distintas 'features': imputación de valores por la mediana en caso de que haya valores
# perdidos en las columnas numéricas; e imputación con el valor 'missing' en caso de que haya valores perdidos en las
# columnas categóricas.
# Posteriormente se usa 'StandardScaler' para estandarizar las columnas numéricas con media cero y varianza unidad; se
# usa 'OneHotEncoder' para variables categóricas cuyos valores no tienen orden ninguno; y 'OrdinalEncoder' para variables
# categóricas cuyos valores tienen un orden
transform = ColumnTransformer([("num_imputer", SimpleImputer(strategy="median"), num_col),
                               ("ohe_encoding", OneHotEncoder(), ohe_col),
                               ("num_scaler", StandardScaler(), num_col),
                               ("ord_encoding", OrdinalEncoder(), ord_col)])

# Se crea un 'Pipeline' que encadena las transformaciones a las columnas ('transform') y el modelo bagging' con el
# algoritmo 'Random Forest'
pipe = Pipeline(steps=[("preprocessing", transform),
                       ("model", RandomForestClassifier(random_state=42))])

# Entrenamiento del modelo
pipe.fit(X_train, y_train)

# Se realiza la evaluación del modelo: se predicen las muestras de 'test' para calcular posteriormente la 'accuracy' y
# el Valor-F
predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

# Se imprime por pantalla las dos métricas
print("Accuracy:", str(round(accuracy, 2) * 100) + "%", "F1:", round(f1, 2))

# Se calcula y se visualiza la matriz de confusión usando como etiquetas las del clasificador entrenado
cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()

# La matriz de confusión se guarda dentro de la carpeta 'Results'
plt.savefig("./Results/model_results.png", dpi=120)

# Se escriben las métricas de 'accuracy' y Valor-F en un archivo de texto, almacenándolo en la carpeta 'Results'
with open("./Results/metrics.txt", "w") as outfile:
    outfile.write(f"\nAccuracy = {round(accuracy, 2) * 100}%, F1 Score = {round(f1, 2)}")

# Se guarda el modelo dentro de la carpeta 'Model'
sio.dump(pipe, "./Model/drug_pipeline.skops")