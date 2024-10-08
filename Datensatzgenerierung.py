#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error


# In[2]:


# Anzahl der Datensätze
num_entries = 1000

# Eindeutige Identifikationsnummer für jeden Versicherungsnehmer
ID = np.arange(1, num_entries + 1)

# Alter des Versicherungsnehmers in Jahren (18-80 Jahre)
Alter = np.random.randint(18, 81, size=num_entries)

# Geschlecht des Versicherungsnehmers (M = männlich, W = weiblich)
Geschlecht = np.random.choice(['M', 'W'], size=num_entries)

# Familienstand (Verheiratet, Single, Geschieden)
Familienstand = np.random.choice(['Verheiratet', 'Single', 'Geschieden'], size=num_entries)

# Anzahl der Kinder (0-4 Kinder)
Kinder = np.random.randint(0, 5, size=num_entries)

# Beruf des Versicherungsnehmers (Beispielberufe)
Berufe = ['Ingenieur', 'Lehrer', 'Verkäufer', 'Manager', 'Techniker', 'Arzt', 'Künstler']
Beruf = np.random.choice(Berufe, size=num_entries)

# Höchster Bildungsabschluss (z.B. Bachelor, Master, PhD, MBA, kein Abschluss)
Bildungsabschluss = np.random.choice(['Kein Abschluss', 'Bachelor', 'Master', 'PhD', 'MBA'], size=num_entries)

# Einkommen in Euro (zwischen 20.000 und 150.000 Euro jährlich)
Einkommen = np.random.randint(20000, 150001, size=num_entries)

# Frühere Schadenshistorie (Anzahl der früheren Versicherungsschäden 0-5)
Frühere_Schadenshistorie = np.random.randint(0, 6, size=num_entries)

# Gesundheitszustand (z.B. Gut, Sehr gut, Durchschnittlich)
Gesundheitszustand = np.random.choice(['Gut', 'Sehr gut', 'Durchschnittlich'], size=num_entries)

# Raucherstatus (Ja, Nein)
Raucher = np.random.choice(['Ja', 'Nein'], size=num_entries)

# Versicherungssumme in Euro (zwischen 50.000 und 1.000.000 Euro, korreliert mit Einkommen)
Versicherungssumme = Einkommen * np.random.uniform(2.5, 4.5, size=num_entries)

# Laufzeit der Versicherung in Jahren (1-30 Jahre)
Laufzeit_in_Jahren = np.random.randint(1, 31, size=num_entries)

# Versicherungsprämie in Euro (korreliert mit mehreren Variablen)
# Faktoren: Einkommen, Gesundheitszustand, Kinder, Laufzeit, Versicherungssumme
base_prämie = 300 + (0.002 * Versicherungssumme) + (0.03 * Einkommen) + (100 * Kinder) + (50 * Laufzeit_in_Jahren)

# Anpassung durch Gesundheitszustand
gesundheitszustand_factor = np.array([1.2 if x == 'Durchschnittlich' else 0.9 for x in Gesundheitszustand])
base_prämie *= gesundheitszustand_factor

# Anpassung durch Raucherstatus
raucher_factor = np.array([1.3 if x == 'Ja' else 1.0 for x in Raucher])
Versicherungsprämie = base_prämie * raucher_factor

# Erstellen des DataFrame
df = pd.DataFrame({
    'ID': ID,
    'Alter': Alter,
    'Geschlecht': Geschlecht,
    'Familienstand': Familienstand,
    'Kinder': Kinder,
    'Beruf': Beruf,
    'Bildungsabschluss': Bildungsabschluss,
    'Einkommen': Einkommen,
    'Frühere Schadenshistorie': Frühere_Schadenshistorie,
    'Gesundheitszustand': Gesundheitszustand,
    'Raucher': Raucher,
    'Versicherungssumme': Versicherungssumme.round(2),
    'Laufzeit_in_Jahren': Laufzeit_in_Jahren,
    'Versicherungsprämie': Versicherungsprämie.round(2)
})

# Zeigt die ersten 5 Datensätze
print(df.head())


# In[3]:


# DataFrame als CSV speichern
df.to_csv('Versicherungsdatensatz.csv', index=False)  # 'Versicherungsdatensatz.csv' wird im gleichen Ordner gespeichert

print("CSV-Datei erfolgreich gespeichert.")

