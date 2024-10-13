#Imports

import pandas as pd
import numpy as np
import joblib
import shap
import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
# CSV-Datei wird geladen
df = pd.read_csv('Versicherungsdatensatz.csv')

# Überprüfen, ob das DataFrame korrekt geladen wurde
print(df)
# Datensatz auf Verarbeitung vorbereiten
le = LabelEncoder()
df['Geschlecht'] = le.fit_transform(df['Geschlecht'])  # Männlich = 1, Weiblich = 0
df['Raucher'] = le.fit_transform(df['Raucher'])        # Ja = 1, Nein = 0
df['Gesundheitszustand'] = le.fit_transform(df['Gesundheitszustand'])  # Beispiel: Durchschnittlich = 2, Gut = 1, Sehr gut = 0
df = pd.get_dummies(df, columns=['Familienstand', 'Beruf', 'Bildungsabschluss'], drop_first=True)
#Zielvariable für eine Regression festlegen (Versicherungsprämie vorhersagen)
X = df.drop('Versicherungsprämie', axis=1)  # Alle anderen Spalten sind die Features
y = df['Versicherungsprämie']  # Zielvariable
# Aufteilen der Daten in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# DataFrame als CSV speichern
X_train.to_csv('X_train.csv', index=False)  # 'X_train.csv' wird im gleichen Ordner gespeichert

print("CSV-Datei erfolgreich gespeichert.")
# Modellinitialisierung mit RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Trainiere das Modell mit den Trainingsdaten
model.fit(X_train, y_train)

# Vorhersagen für die Testdaten treffen
y_pred = model.predict(X_test)

# Modellbewertung durch Berechnung des Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Extrahieren der Feature Importances aus dem Modell
feature_importances = pd.Series(model.feature_importances_, index=X.columns)

# Sortieren der Feature Importances in absteigender Reihenfolge und Ausgabe
print(feature_importances.sort_values(ascending=False))
# Feature Importance plotten (optional)
import matplotlib.pyplot as plt

feature_importances.sort_values(ascending=False).plot(kind='bar', figsize=(10, 6))
plt.title('Feature Importance')
plt.ylabel('Wichtigkeit')
plt.xlabel('Features')
plt.show()
# Definition des Parameter-Rasters für die Grid-Suche
param_grid = {
    'n_estimators': [50, 100, 200],    # Anzahl der Bäume im Wald
    'max_depth': [None, 10, 20, 30],   # Maximale Tiefe der Bäume
    'min_samples_split': [2, 5, 10],    # Mindestanzahl von Stichproben für einen Split
    'min_samples_leaf': [1, 2, 4],     # Mindestanzahl von Blättern in einem Knoten
    'bootstrap': [True, False]         # Verwendung von Bootstrap-Stichproben
}
# GridSearchCV initialisieren
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# GridSearchCV trainieren
grid_search.fit(X_train, y_train)
# Beste Parameter anzeigen
print("Beste Parameter: ", grid_search.best_params_)

# Bestes Modell und Score
print("Bestes Score: ", grid_search.best_score_)
# Speichern des Modells nach dem GridSearchCV
joblib.dump(grid_search.best_estimator_, 'random_forest_model.pkl')
# Erstelle den Explainer
explainer = shap.Explainer(grid_search.best_estimator_)

# Berechne die SHAP-Werte
shap_values = explainer(X_test)

# Zusammenfassungsdiagramm erstellen
shap.summary_plot(shap_values, X_test)

# Abhängigkeitstabelle für ein bestimmtes Feature erstellen
shap.dependence_plot('Einkommen', shap_values.values, X_test)

# Erstelle den LIME-Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns,
    class_names=['Nicht Raucher', 'Raucher'],  # Falls Klassifikation
    mode='regression'  # oder 'classification' je nach Modelltyp

# Wählt eine Instanz aus den Testdaten aus
instance = X_test.iloc[0]

# Erzeuge eine Erklärung für diese Instanz
explanation = explainer.explain_instance(instance.values, model.predict)

# Visualisiere die Erklärung
explanation.show_in_notebook(show_table=True, show_all=False)
explanation.as_list()  # Ausgabe als Liste


