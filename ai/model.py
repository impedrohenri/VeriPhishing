import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance



df = pd.read_csv("./ai/dataset.csv")
df = pd.DataFrame(df)

y = df.target
x = df.drop(columns=['target'], inplace=False)


x_treino, x_teste, y_treino, y_teste = train_test_split(x, y , test_size= 0.20, shuffle=True, stratify=y, random_state=52)


XGBoost_model = XGBClassifier()
XGBoost_model.fit(x_treino, y_treino)

y_pred = XGBoost_model.predict(x_teste)


# ------------ Random Forrest ------------- 

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y , test_size= 0.20, shuffle=True, stratify=y, random_state=52)

RF_model = XGBClassifier()
RF_model.fit(x_treino, y_treino)

y_pred = RF_model.predict(x_teste)

print(classification_report(y_teste, y_pred))

