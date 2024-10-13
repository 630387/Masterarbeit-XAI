#Imports

import joblib
import pandas as pd
import lime
import lime.lime_tabular
import shap
import numpy as np
# Lade das gespeicherte Modell
loaded_model = joblib.load('random_forest_model.pkl')

# Einzelner Datensatz mit denselben Features wie X_train
single_data = pd.DataFrame({
    'Alter': [45],
    'Geschlecht': ['M'],
    'Familienstand': ['Verheiratet'],
    'Kinder': [2],
    'Beruf': ['Ingenieur'],
    'Einkommen': [80000],
    'Frühere Schadenshistorie': [1],
    'Gesundheitszustand': ['Gut'],
    'Raucher': ['Nein'],
    'Versicherungssumme': [300000],
    'Laufzeit_in_Jahren': [20]
})

# One-Hot-Encoding der kategorischen Variablen
single_data_encoded = pd.get_dummies(single_data)

# Gleiche Spaltenstruktur wie im Training sicherstellen
single_data_encoded = single_data_encoded.reindex(columns=['ID', 'Alter', 'Geschlecht', 'Kinder', 'Einkommen',
       'Frühere Schadenshistorie', 'Gesundheitszustand', 'Raucher',
       'Versicherungssumme', 'Laufzeit_in_Jahren', 'Familienstand_Single',
       'Familienstand_Verheiratet', 'Beruf_Arzt', 'Beruf_Biologe',
       'Beruf_Buchhalter', 'Beruf_Chemiker', 'Beruf_Feuerwehrmann',
       'Beruf_Ingenieur', 'Beruf_Jurist', 'Beruf_Künstler', 'Beruf_Lehrer',
       'Beruf_Manager', 'Beruf_Marketingexperte', 'Beruf_Musiker',
       'Beruf_Pilot', 'Beruf_Polizist', 'Beruf_Psychologe',
       'Beruf_Schriftsteller', 'Beruf_Softwareentwickler', 'Beruf_Techniker',
       'Beruf_Verkäufer', 'Bildungsabschluss_Kein Abschluss',
       'Bildungsabschluss_MBA', 'Bildungsabschluss_Master',
       'Bildungsabschluss_PhD'], fill_value=0)

# Initialisiere den SHAP-Explainer (für Random Forest wird TreeExplainer verwendet)
explainer = shap.TreeExplainer(loaded_model)

# Berechnung der SHAP-Werte für den einzelnen Datensatz
shap_values = explainer.shap_values(single_data_encoded)

# Visualisierung der SHAP-Werte (Beitrag jedes Features zur Vorhersage)
shap.initjs()  # Initialisiere Javascript für die Darstellung (nur für Jupyter)
shap.force_plot(explainer.expected_value, shap_values[0], single_data_encoded.iloc[0])

# CSV-Datei laden
X_train = pd.read_csv('X_train.csv')

# Überprüfen, ob das DataFrame geladen wurde
print("X_train geladen")
print(X_train.head(5))

# Initialisiere den LIME-Erklärer mit den Trainingsdaten
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),  # Trainingsdaten als numpy array
    feature_names=X_train.columns,    # Feature-Namen
    class_names=['Versicherungsprämie'],  # Zielvariable (optional)
    mode='regression'  # Bei Regression
)

# Erklärt die Vorhersage des Modells für den einzelnen Datensatz
exp = explainer.explain_instance(
    data_row=single_data_encoded.iloc[0],  # Der einzelne Datensatz
    predict_fn=loaded_model.predict        # Modell-Vorhersagefunktion
)

# Zeigt die Erklärung
exp.show_in_notebook(show_all=False)  # Falls du in einem Notebook arbeitest
exp.as_list()  # Ausgabe als Liste (falls nicht in einem Notebook)

import openai

# Deinen OpenAI API-Schlüssel eingeben
openai.api_key = 'sk-u08zmV-PsFr7f6lOXugzX33gxFG5VvpFRc6neBCUqKT3BlbkFJ-clHqXzrbLVbRt2_td_bDqDNbIiKqL8tcuakJIf0IA'

# LIME-Daten als String
lime_data = exp.as_list()

# Anfrage an die ChatGPT API senden
response = openai.ChatCompletion.create(
    model="gpt-4",  # GPT-4-Modell verwenden
    messages=[
        {"role": "system", "content": "Du bist ein Versicherungsvertreter, der einem Kunden die Auswirkungen seiner Daten auf seine Versicherungsprämie erklärt. Verwende eine freundliche, verständliche und professionelle Sprache."},
        {"role": "user", "content": f"Hier sind die Daten aus der LIME-Analyse: {lime_data}. Erkläre bitte dem Kunden, welche Faktoren sich am meisten auf seine Versicherungsprämie auswirkt und gib ihm eine kurze Empfehlung was er tun könnte um diese zu senken"}
    ],
    max_tokens=400  # Maximale Token-Anzahl der Antwort
)

# Ausgabe der Antwort von ChatGPT
print(response['choices'][0]['message']['content'])



