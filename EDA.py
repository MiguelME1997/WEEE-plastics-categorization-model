import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns


def cramers_v(var1, varObj):
    if not var1.dtypes.name == 'category':
        # bins = min(5,var1.value_counts().count())
        var1 = pd.cut(var1, bins=5)
    if not varObj.dtypes.name == 'category':  # np.issubdtype(varObj, np.number):
        # bins = min(5,varObj.value_counts().count())
        varObj = pd.cut(varObj, bins=5)

    data = pd.crosstab(var1, varObj).values
    vCramer = stats.contingency.association(data, method='cramer')
    return vCramer

# Opciones de display para que pandas muestre todas las columnas
pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 25)
pd.set_option('display.max_columns', 3000)

# Leemos los datos
data = pd.read_excel("Database (extendida_7 variables).xlsx")
# data.info()
# print(data[data["SORT AS"] == "POPs"].count())
counts_df = data[["SORT AS"]].value_counts()
counts_df.plot(kind="bar")
# plt.show()


data.loc[data["Origin"].isna(), "Origin"] = "NaN"
data["Origin"] = data.Origin.str.lower()
data.loc[data["Origin"] == "nan", "Origin"] = np.nan
data.loc[data["Origin"] == "PRC", "Origin"] = "china"

counts_df = data[["Type"]].value_counts()
counts_df.plot(kind="bar")
# plt.show()

"""
type_counts = data.groupby("SORT AS")["Type"].value_counts().unstack()
print(type_counts)
sns.barplot(data=type_counts)
# plt.show()
"""
plt.close()
sns.countplot(data=data, x="Type", hue="SORT AS")
# plt.show()

# print(data[["SIZE"]].describe())

# for column in data.iloc[:,:7]:
#     cramer = cramers_v(column, data["SORT AS"])
#     print(cramer)

from statsmodels.graphics.mosaicplot import mosaic

# plt.clf()
mosaic(data, ["Type", "SORT AS"], gap = 0.1, title="Type vs Sort")
# plt.show()

sns.boxplot(x="SIZE", y = "SORT AS", data = data, palette="viridis")
# plt.show()

data = data[data["SORT AS"].notna()]
data = data[data['Plastic type'].notna()]

# print(data.select_dtypes(exclude=np.number).describe())
# print(data)

from string import ascii_letters

sns.set_theme(style="white")


# Generate a large random dataset
rs = np.random.RandomState(33)
data = data[data["SORT AS"].notna()]
data = data[data['Plastic type'].notna()]

plt.show()
plt.close()


for column in ["Plastic type", "Manufacturer", "Origin", "Type"]:

    df_counts = pd.DataFrame(data[data["SORT AS"] == "Recyclable"][[column]]
                                       .value_counts()).sort_values(by=["count"], ascending=False)
    data_count = data[[column]].value_counts()

    df_counts["count_pct"] = df_counts["count"]/data_count
    df_counts["count_total"] = data_count
    df_counts[column] = df_counts.index
    df_counts = df_counts.head(10)
    sns.barplot(data=df_counts, y="count_pct", x=column, order=df_counts.sort_values("count_pct", ascending=False)[column])
    plt.show()


for column in ["Plastic type", "Manufacturer", "Origin", "Type"]:

    df_counts = pd.DataFrame(data[data["SORT AS"] != "Recyclable"][[column]]
                                       .value_counts()).sort_values(by=["count"], ascending=False)
    data_count = data[[column]].value_counts()

    df_counts["count_pct"] = df_counts["count"]/data_count
    df_counts["count_total"] = data_count
    df_counts[column] = df_counts.index
    df_counts = df_counts.head(10)
    sns.barplot(data=df_counts, y="count_pct", x=column, order=df_counts.sort_values("count_pct", ascending=False)[column])
    plt.show()
