# Importacion de librerias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import plotly.express as px
#import folium
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# Se importa la base de datos
url='https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv'
df=pd.read_csv(url)
# Se definen los features y variable target
X=df.drop(columns='Outcome')
y=df['Outcome']
# Se divide la muestra
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=1607)
df_train=pd.concat([X_train,y_train], axis=1)
# Se transforman las variables según lo descubierto en el EDA
df_train['Age_rang']=pd.cut(df_train['Age'],bins=[20,30,40,50,60,70])
df_train['Age_rang']=df_train['Age_rang'].astype('category')
for i in ['BMI','BloodPressure','Glucose']:
    df_train=df_train.replace({i:0}, df_train[i].mean()) 
df_train['Insulin'].mask(df_train['Insulin']>400,df_train['Insulin'].mean(),inplace=True)
# Se aplica la transformacion al set de prueba que se utillizara luego para testear el modelo
X_test['Age_rang']=pd.cut(X_test['Age'],bins=[20,30,40,50,60,70])
X_test['Age_rang']=X_test['Age_rang'].astype('category')
# Ordinal Encode
df_train['Age_rang_cod']=df_train['Age_rang'].cat.codes
X_test['Age_rang_cod']=X_test['Age_rang'].cat.codes

X_train=df_train.drop(columns=['Outcome','Age','Age_rang'])
X_test=X_test.drop(columns=['Age','Age_rang'])
y_train=df_train['Outcome']

# Se carga el modelo tuneado
filename = '/workspace/CART/models/finalized_model.sav'
best_tree_model = pickle.load(open(filename, 'rb'))
best_tree_model.fit(X_train,y_train) # se entrena el modelo
y_pred_best=best_classif.predict(X_test) # predicción
best_tree_model.score(X_test, Y_test) # ajuste del modelo en la muestra de prueba
