# Module 1: Present-Day Habitability
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')


file_path = os.path.join('data', 'phl_exoplanet_catalog_2019.csv')
data = pd.read_csv(file_path)


print(data.info())
print(data.shape)
print(data.isnull().sum())
print(data.describe())


plt.figure(figsize=(9, 6))
data.P_HABITABLE.value_counts(normalize=True, ascending=False).plot(
    kind='bar', color=['navy', 'orange', 'green'], alpha=0.8, rot=0
)
plt.title('Habitability Indicator: No (0) / Conservatively Yes (1) / Optimistically Yes (2)')
plt.show()


no = data[data.P_HABITABLE == 0]
yes_cons = data[data.P_HABITABLE == 1]
yes_opti = data[data.P_HABITABLE == 2]

yes_cons_oversampled = resample(yes_cons, replace=True, n_samples=len(no), random_state=12345)
yes_opti_oversampled = resample(yes_opti, replace=True, n_samples=len(no), random_state=12345)
oversampled = pd.concat([no, yes_cons_oversampled, yes_opti_oversampled])


plt.figure(figsize=(9, 6))
oversampled.P_HABITABLE.value_counts(normalize=True, ascending=False).plot(
    kind='bar', color=['navy', 'orange', 'green'], alpha=0.8, rot=0
)
plt.title('Habitability Indicator After Oversampling (Balanced Dataset)')
plt.show()


by_p_detec = (
    oversampled.groupby('P_DETECTION')
    .filter(lambda x: len(x) > 5)
    .groupby(['P_DETECTION', 'P_YEAR'])
    .size()
    .unstack()
)
plt.figure(figsize=(12, 12))
sns.heatmap(by_p_detec, square=True, cbar_kws={'fraction': 0.01}, cmap='OrRd', linewidth=1)
plt.show()


sns.catplot(y="P_YEAR", hue="P_HABITABLE", kind="count", palette="pastel", edgecolor=".6", data=oversampled, aspect=1.5)
sns.catplot(x="P_TYPE_TEMP", y="P_ESI", hue="P_HABITABLE", col="P_TYPE", col_wrap=3, aspect=0.8, kind="boxen", data=oversampled)
plt.show()