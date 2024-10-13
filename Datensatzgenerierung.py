# Import der notwendigen Bibliotheken
import numpy as np
import pandas as pd

# Anzahl der Datensätze
num_entries = 1000

# Generierung einer eindeutigen Identifikationsnummer für jeden Versicherungsnehmer
ID = np.arange(1, num_entries + 1)

# Generierung des Alters der Versicherungsnehmer (18-80 Jahre)
Alter = np.random.randint(18, 81, size=num_entries)

# Zuweisung des Geschlechts der Versicherungsnehmer (M = männlich, W = weiblich)
Geschlecht = np.random.choice(['M', 'W'], size=num_entries)

# Zuweisung des Familienstands (Verheiratet, Single, Geschieden)
Familienstand = np.random.choice(['Verheiratet', 'Single', 'Geschieden'], size=num_entries)

# Realistische Verteilung der Kinderanzahl basierend auf Alter und Familienstand
Kinder = np.zeros(num_entries, dtype=int)

for i in range(num_entries):
    if Familienstand[i] == 'Verheiratet':
        if Alter[i] < 25:
            Kinder[i] = np.random.choice([0, 1], p=[0.8, 0.2])  # Junge verheiratete Menschen haben seltener Kinder
        elif 25 <= Alter[i] <= 40:
            Kinder[i] = np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.3, 0.3, 0.2, 0.1])  # Höhere Kinderanzahl im mittleren Alter
        else:
            Kinder[i] = np.random.choice([0, 1, 2, 3], p=[0.1, 0.3, 0.4, 0.2])  # Ältere Versicherungsnehmer haben tendenziell weniger Kinder
    elif Familienstand[i] == 'Single':
        if Alter[i] < 25:
            Kinder[i] = 0  # Sehr wenige Singles unter 25 haben Kinder
        elif 25 <= Alter[i] <= 40:
            Kinder[i] = np.random.choice([0, 1], p=[0.7, 0.3])  # Single im mittleren Alter haben häufiger ein Kind
        else:
            Kinder[i] = np.random.choice([0, 1], p=[0.8, 0.2])  # Ältere Singles haben meist keine Kinder
    elif Familienstand[i] == 'Geschieden':
        if Alter[i] < 25:
            Kinder[i] = np.random.choice([0, 1], p=[0.9, 0.1])  # Junge Geschiedene haben selten Kinder
        elif 25 <= Alter[i] <= 40:
            Kinder[i] = np.random.choice([0, 1, 2, 3], p=[0.2, 0.4, 0.3, 0.1])  # Geschiedene im mittleren Alter haben mehr Kinder
        else:
            Kinder[i] = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])  # Ältere Geschiedene haben weniger Kinder

# Beruf des Versicherungsnehmers aus einer erweiterten Liste von Berufen
Berufe = ['Ingenieur', 'Lehrer', 'Verkäufer', 'Manager', 'Techniker', 'Arzt', 'Künstler', 'Polizist', 
          'Feuerwehrmann', 'Pilot', 'Architekt', 'Softwareentwickler', 'Marketingexperte', 
          'Buchhalter', 'Schriftsteller', 'Biologe', 'Chemiker', 'Musiker', 'Jurist', 'Psychologe']
Beruf = np.random.choice(Berufe, size=num_entries)

# Höchster Bildungsabschluss der Versicherungsnehmer
Bildungsabschluss = np.random.choice(['Kein Abschluss', 'Bachelor', 'Master', 'PhD', 'MBA'], size=num_entries)

# Einkommensverteilung nach Beruf und Bildungsabschluss
einkommensverteilung = {
    'Ingenieur': (50000, 120000),
    'Lehrer': (30000, 60000),
    'Verkäufer': (20000, 50000),
    'Manager': (70000, 150000),
    'Techniker': (40000, 80000),
    'Arzt': (80000, 150000),
    'Künstler': (20000, 60000),
    'Polizist': (30000, 70000),
    'Feuerwehrmann': (30000, 70000),
    'Pilot': (60000, 130000),
    'Architekt': (50000, 100000),
    'Softwareentwickler': (60000, 120000),
    'Marketingexperte': (40000, 90000),
    'Buchhalter': (35000, 80000),
    'Schriftsteller': (25000, 60000),
    'Biologe': (35000, 80000),
    'Chemiker': (45000, 90000),
    'Musiker': (20000, 50000),
    'Jurist': (60000, 150000),
    'Psychologe': (40000, 90000)
}

# Anpassung des Einkommens basierend auf dem Bildungsabschluss
bildungs_einkommen_factor = {
    'Kein Abschluss': 0.8,
    'Bachelor': 1.0,
    'Master': 1.2,
    'PhD': 1.5,
    'MBA': 1.3
}
Einkommen = np.array([int(np.random.randint(einkommensverteilung[beruf][0], einkommensverteilung[beruf][1] + 1) * 
                         bildungs_einkommen_factor[abschluss]) 
                      for beruf, abschluss in zip(Beruf, Bildungsabschluss)])

# Zuweisung der Schadenshistorie (Anzahl der bisherigen Versicherungsschäden) basierend auf Beruf und Alter
berufs_risiko = {
    'Ingenieur': 1,
    'Lehrer': 1,
    'Verkäufer': 1,
    'Manager': 1,
    'Techniker': 2,
    'Arzt': 1,
    'Künstler': 1,
    'Polizist': 3,  # Höheres Risiko aufgrund des Berufs
    'Feuerwehrmann': 3,  # Höheres Risiko aufgrund des Berufs
    'Pilot': 2,  
    'Architekt': 1,
    'Softwareentwickler': 1,
    'Marketingexperte': 1,
    'Buchhalter': 1,
    'Schriftsteller': 1,
    'Biologe': 1,
    'Chemiker': 2,
    'Musiker': 1,
    'Jurist': 1,
    'Psychologe': 1
}

Frühere_Schadenshistorie = np.array([np.random.randint(0, berufs_risiko[beruf] + 1) + (Alter[i] // 20)
                                    for i, beruf in enumerate(Beruf)])

# Zuweisung des Gesundheitszustands (realistische Verteilung von sehr gut bis sehr schlecht)
Gesundheitszustand = np.random.choice(
    ['Sehr gut', 'Gut', 'Durchschnittlich', 'Krank', 'Chronisch Krank', 'Schwer Krank', 'Sehr schlecht'],
    size=num_entries,
    p=[0.2, 0.3, 0.3, 0.1, 0.05, 0.03, 0.02]  # Wahrscheinlichkeitsverteilung nach Gesundheitszustand
)

# Zuweisung des Raucherstatus basierend auf Alter (jüngere und ältere Personen rauchen seltener)
Raucher = np.array(['Nein'] * num_entries)
for i in range(num_entries):
    if 25 <= Alter[i] <= 45:
        Raucher[i] = np.random.choice(['Ja', 'Nein'], p=[0.35, 0.65])
    elif 18 <= Alter[i] < 25:
        Raucher[i] = np.random.choice(['Ja', 'Nein'], p=[0.15, 0.85])
    elif 46 <= Alter[i] <= 60:
        Raucher[i] = np.random.choice(['Ja', 'Nein'], p=[0.25, 0.75])
    else:
        Raucher[i] = np.random.choice(['Ja', 'Nein'], p=[0.1, 0.9])

# Berechnung der Versicherungssumme basierend auf dem Einkommen (zwischen 50.000 und 1.000.000 Euro)
Versicherungssumme = Einkommen * np.random.uniform(2.5, 4.5, size=num_entries)

# Berechnung der Laufzeit der Versicherung (1-30 Jahre) basierend auf dem Alter
Laufzeit_in_Jahren = np.zeros(num_entries, dtype=int)
for i in range(num_entries):
    if Alter[i] <= 30:
        Laufzeit_in_Jahren[i] = np.random.randint(10, 31)  # Jüngere Menschen bevorzugen längere Laufzeiten
    elif 30 < Alter[i] <= 60:
        Laufzeit_in_Jahren[i] = np.random.randint(5, 21)  # Mittleres Alter bevorzugt mittlere Laufzeiten
    else:
        Laufzeit_in_Jahren[i] = np.random.randint(1, 11)  # Ältere Menschen bevorzugen kürzere Laufzeiten

# Berechnung des beruflichen Risikofaktors (Berufe mit höherem Risiko führen zu höheren Prämien)
berufs_risiko_factor = {
    'Ingenieur': 1.0,
    'Lehrer': 1.0,
    'Verkäufer': 1.1,
    'Manager': 1.2,
    'Techniker': 1.2,
    'Arzt': 1.1,
    'Künstler': 1.0,
    'Polizist': 1.5,  # Erhöhtes Risiko für Polizisten
    'Feuerwehrmann': 1.5,  # Erhöhtes Risiko für Feuerwehrleute
    'Pilot': 1.3,
    'Architekt': 1.1,
    'Softwareentwickler': 1.0,
    'Marketingexperte': 1.1,
    'Buchhalter': 1.0,
    'Schriftsteller': 1.0,
    'Biologe': 1.0,
    'Chemiker': 1.2,
    'Musiker': 1.0,
    'Jurist': 1.1,
    'Psychologe': 1.0
}

# Altersfaktor für die Prämienberechnung (ältere Personen zahlen mehr)
alter_factor = np.array([1.0 if age < 30 else 1.1 if age <= 50 else 1.3 for age in Alter])

# Schadenshistorie-Faktor (mehr Schäden führen zu höheren Prämien)
schadenshistorie_factor = 1 + (Frühere_Schadenshistorie * 0.05)

# Grundprämie basierend auf Versicherungssumme, Einkommen, Kinderanzahl und Laufzeit
base_prämie = 300 + (0.002 * Versicherungssumme) + (0.03 * Einkommen) + (100 * Kinder) + (50 * Laufzeit_in_Jahren)

# Anpassung durch Gesundheitszustand (höhere Prämien für schlechtere Gesundheitszustände)
gesundheitszustand_factor = np.array([1.5 if x == 'Sehr schlecht' else 1.3 if x == 'Schwer Krank' else 1.2 if x == 'Chronisch Krank' else 1.1 if x == 'Krank' else 0.9 if x == 'Sehr gut' else 1.0 for x in Gesundheitszustand])

# Anpassung der Prämie für Raucherstatus (Raucher zahlen mehr)
raucher_factor = np.array([1.3 if x == 'Ja' else 1.0 for x in Raucher])

# Anwendung des beruflichen Risikofaktors
berufs_risiko = np.array([berufs_risiko_factor[beruf] for beruf in Beruf])

# Berechnung der endgültigen Versicherungsprämie unter Berücksichtigung aller Faktoren
Versicherungsprämie = base_prämie * gesundheitszustand_factor * raucher_factor * berufs_risiko * schadenshistorie_factor * alter_factor

# Runden der Prämie auf zwei Nachkommastellen
Versicherungsprämie = Versicherungsprämie.round(2)

# Erstellen des finalen DataFrames mit allen Variablen
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
    'Versicherungsprämie': Versicherungsprämie
})

# Zeigt die ersten 5 Datensätze
print(df.head())
print("CSV-Datei erfolgreich gespeichert.")
