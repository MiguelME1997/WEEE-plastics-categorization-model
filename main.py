import pandas as pd
import numpy as np
from numpy import random
from matplotlib import pyplot as plt
import pickle as pkl

# Opciones de display para que pandas muestre todas las columnas
pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 25)
pd.set_option('display.max_columns', 3000)

# Leemos los datos
data = pd.read_excel("Database (extendida_7 variables).xlsx")

val_data_esc = pd.read_excel("Bases de datos (Escocia y España).xlsx")
val_data_esc = val_data_esc[val_data_esc["Type"] != "LDC"] # Normalmente es "LCD"
val_data_esc["SORT AS"] = val_data_esc["SORT AS"].replace({"RECYCLABLE": "Recyclable"})

val_data_esp = pd.read_excel("Bases de datos (Escocia y España).xlsx", sheet_name="Plástico_España")

data = data[data["SORT AS"].notna()]
data = data[data['Plastic type'].notna()]

data = data[data["SIZE"] <= 100]
val_data_esc = val_data_esc[val_data_esc["SIZE"] <= 100]
val_data_esp = val_data_esp[val_data_esp["SIZE"] <= 100]

data_all = pd.concat([data, val_data_esc, val_data_esp], axis="rows")
data_all = data_all.drop("DATE", axis = "columns")

data_all.loc[data_all["Origin"].isna(), "Origin"] = "NaN"
data_all["Origin"] = data_all["Origin"].apply(lambda x: x.lower())
data_all.loc[data_all["Origin"] == "nan", "Origin"] = np.nan
data_all.loc[data_all["Origin"] == "PRC", "Origin"] = "china"

data_all.loc[data_all["YoM"].isna(), "YoM"] = 0
data_all["YoM"] = data_all["YoM"].astype("int32")
data_all.loc[data_all["YoM"] == 0, "YoM"] = np.nan

data_all["Manufacturer"] = data_all["Manufacturer"].apply(lambda x: x.lower())

data_all.info()


# Importamos librerías para codificar
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

oe_date = OrdinalEncoder()

for variable in data_all.iloc[:, 0:6].select_dtypes(exclude=np.number).columns:
    OE = OrdinalEncoder()
    data_all[variable] = OE.fit_transform(data_all[variable].values.reshape(-1, 1))

le = LabelEncoder()
data_all["SORT AS"] = le.fit_transform(data_all["SORT AS"])

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=4)
imputed = imputer.fit_transform(data_all)

data_all = pd.DataFrame(imputed)
data_all.columns = data.iloc[:, 1:].columns

val_data_esc = data_all.iloc[len(data):len(data)+len(val_data_esc), :]
val_data_esp = data_all.iloc[len(data)+len(val_data_esc):, :]
data_train = data_all.iloc[0:len(data), :]

X = data_train.iloc[:, 0:6]
y = data_train["SORT AS"]
X.columns = data_train.iloc[:, 0:len(data_train.columns)-1].columns.values

data_train.info()
counts_df = data_train[["SORT AS"]].value_counts()
counts_df.plot(kind="bar")
plt.show()
"""
Modelo RF.
"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

smote = SMOTE(k_neighbors=4, random_state=123)
# X, y = smote.fit_resample(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,
                                                    random_state=1255, stratify=y)


"""
Ggridsearch
"""

Best_params = {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 150}

try:
    with open("model_RF_final.pkl", "rb") as f:
        modeloRF = pkl.load(f)

except IOError:
    modeloRF = RandomForestClassifier(max_depth=15, min_samples_leaf=4, min_samples_split=3, n_estimators=150,
                                  random_state=123)
    modeloRF.fit(X_train, y_train)

    with open("model_RF_final.pkl", "wb") as f:
        pkl.dump(modeloRF, f)

y_pred = modeloRF.predict(X_test)

from sklearn.metrics import classification_report
report = classification_report(y_true=y_test, y_pred=y_pred)
print(report)

"""Confusion matrix TEST"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

matrix = confusion_matrix(y_true=y_test, y_pred = y_pred)
disp = ConfusionMatrixDisplay(matrix, display_labels=modeloRF.classes_)
disp.plot()
plt.title("Confusion matrix TEST")
plt.show()


"""
Validación cruzada
"""
from sklearn.model_selection import cross_val_score
CV = cross_val_score(modeloRF, X_train, y_train, cv = 5)
print(CV)

"""
Validación
"""

print("--- VALIDACIÓN ESCOCIA ---")

X_val = val_data_esc.iloc[:, 0:6]
y_val = val_data_esc["SORT AS"]

y_val_pred = modeloRF.predict(X_val)

print(classification_report(y_true=y_val, y_pred=y_val_pred))

"""Confusion matrix VAL ESC"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

matrix = confusion_matrix(y_true=y_val, y_pred = y_val_pred)
disp = ConfusionMatrixDisplay(matrix, display_labels=modeloRF.classes_)
disp.plot()
plt.title("Confusion matrix Validation Scotland")
plt.show()


print("--- VALIDACIÓN ESPAÑA ---")

X_val = val_data_esp.iloc[:, 0:6]
y_val = val_data_esp["SORT AS"]

y_val_pred = modeloRF.predict(X_val)

print(classification_report(y_true=y_val, y_pred=y_val_pred))

"""Confusion matrix VAL ESP"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

matrix = confusion_matrix(y_true=y_val, y_pred = y_val_pred)
disp = ConfusionMatrixDisplay(matrix, display_labels=modeloRF.classes_)
disp.plot()
plt.title("Confusion matrix Validation Spain")

plt.show()

"""
Validación con todo 
"""

print("--- VALIDACIÓN DOS ---")

val_data_all = pd.concat([val_data_esc, val_data_esp], axis= "rows")

X_val = val_data_all.iloc[:, 0:6]
y_val = val_data_all["SORT AS"]

y_val_pred = modeloRF.predict(X_val)

print(classification_report(y_true=y_val, y_pred=y_val_pred))

"""Confusion matrix VAL DOS"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

matrix = confusion_matrix(y_true=y_val, y_pred = y_val_pred)
disp = ConfusionMatrixDisplay(matrix, display_labels=modeloRF.classes_)
disp.plot()
plt.title("Confusion matrix Validation Combined")

plt.show()



"""
Importancia de variables
"""

importances = modeloRF.feature_importances_

forest_importances = pd.Series(importances, index=X_train.columns.values)

std = np.std([tree.feature_importances_ for tree in modeloRF.estimators_], axis=0)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
# plt.show()