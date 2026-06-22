# Roadmap — Construction du Gold Dataset InduSense 4.0

> **À qui s'adresse ce document ?**
> À des étudiants en Machine Learning qui veulent comprendre, pas à pas, comment des données brutes industrielles se transforment en un tableau prêt à être entraîné. Chaque étape explique le *quoi*, le *pourquoi* et les *pièges* à éviter.

---

## Vue d'ensemble du pipeline

```
telemetry.csv          releves_incidents.csv      machine.sql
(télémétrie brute)     (incidents signalés)       (référentiel machines)
       │                        │                        │
       ▼                        ▼                        ▼
  ┌─────────────────────────────────────────────────────────┐
  │                     BRONZE                             │
  │   Chargement brut — validation de format, typage       │
  └─────────────────────────────┬───────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────┐
  │                     SILVER                             │
  │   Nettoyage — déduplication, normalisation, imputation │
  └─────────────────────────────┬───────────────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────┐
  │                      GOLD                              │
  │   Agrégation + Feature Engineering + Labels + *Split*    │
  └─────────────────────────────┬───────────────────────────┘
                                │
                                ▼
                     100 colonnes × ~9 000 lignes/machine
                     Prêt pour sklearn / LightGBM / XGBoost
```

Le Gold Dataset est la couche finale : **une ligne = une machine à une heure donnée**, avec toutes les informations dont un modèle a besoin pour prédire si une panne va survenir dans les prochaines 6h, 12h, 24h ou 48h.

---

## Point de départ : les données brutes

### 1.1 `telemetry.csv` — Ce que captent les capteurs

Chaque ligne est une **mesure brute** d'une machine à un instant donné.

```
machine_id | timestamp            | temperature_c | pressure_bar | voltage_mean_v | rotation_mean_rpm | pieces_produced
MACH-01    | 2025-06-01 00:00:00  | 46.35         | 198.20       | 227.57         | 1541.79           | 4
MACH-01    | 2025-06-01 00:00:00  | 46.33         | 198.21       | 227.57         | 1541.76           | 4   ← doublon !
MACH-01    | 2025-06-01 01:00:00  | 48.76         | 198.30       | 227.48         | 1537.86           | 4
```

**Problèmes connus dans ce fichier :**
- Doublons (double-transmission capteur, ~1 % des lignes)
- Valeurs manquantes (panne capteur, blocs de 4-12h consécutives, ~2 % des lignes)
- Températures parfois en Fahrenheit (capteurs mal configurés)

### 1.2 `releves_incidents.csv` — Ce que signalent les opérateurs

Chaque ligne est un **incident déclaré** par un opérateur sur une machine.

```
incident_id | date       | time  | machine_id | severity | type_surchauffe | type_vibration | ...
INC-000001  | 2025-06-01 | 05:42 | MACH-06    | 4        | 1               | 0              | ...
INC-000002  | 2025-06-01 | 21:08 | MACH-15    | 3        | 0               | 0              | ...
```

**Ce fichier est discret** : il ne contient que les moments où quelqu'un a appuyé sur le bouton. Il n'y a **pas** de ligne "pas de panne à 03h00". Le modèle devra inférer les périodes normales à partir de l'absence d'incidents.

---

## Étape 1 — Silver : nettoyer avant de construire

Avant tout feature engineering, les données doivent être fiables.

### 1.1 Déduplication

```python
# Clé métier : (machine, heure). Si deux mesures existent pour le même créneau,
# on garde la première et on marque l'autre.
telemetry_clean, dedup_log = deduplicate_sensor_records(telemetry_raw, ...)
```

**Pourquoi ici et pas après ?** Si on ne déduplique pas, le calcul de moyenne horaire (étape 2) sera biaisé par les doublons qui gonflent certaines heures.

### 1.2 Normalisation des unités

Certains capteurs envoient des Fahrenheit. La fonction détecte les fenêtres aberrantes (température > 80°C suspect) et convertit :

```
°F → °C :  temperature_c = (temperature_f - 32) × 5/9
```

**Piège classique :** ne pas corriger les unités fait apparaître des "anomalies" fantômes qui polluent les features de tendance.

### 1.3 Imputation des valeurs manquantes

Trois stratégies sont benchmarkées sur chaque signal. La meilleure est sélectionnée automatiquement :

| Stratégie | Principe | Quand c'est bon |
|---|---|---|
| **Médiane** | Remplace par la médiane globale du signal | Signal sans tendance temporelle |
| **Interpolation** | Valeur intermédiaire entre avant/après | Pannes courtes (< 6h) |
| **Itératif** | Régression sur les autres signaux | Signaux corrélés entre eux |

**Règle impérative :** la stratégie est **fittée uniquement sur le train set**. On ne peut pas utiliser des valeurs futures pour imputer le passé — c'est du *data leakage*.

---

## Étape 2 — Agrégation horaire

La télémétrie contient plusieurs mesures par heure. On ramène tout à **exactement une ligne par machine par heure**, qui devient notre unité de base.

```python
hourly = tel.groupby(['machine_id_std', 'window_start']).agg(
    temp_mean_1h  = ('temperature_c', 'mean'),
    temp_max_1h   = ('temperature_c', 'max'),
    pressure_mean_1h = ('pressure_bar', 'mean'),
    ...
)
```

**Résultat :** chaque ligne Gold représente une fenêtre `[window_start, window_start + 1h)`.

```
machine_id_std | window_start        | temp_mean_1h | pressure_mean_1h | ...
MACH-01        | 2025-06-01 00:00:00 | 46.34        | 198.21           | ...
MACH-01        | 2025-06-01 01:00:00 | 48.76        | 198.30           | ...
MACH-01        | 2025-06-01 02:00:00 | 51.35        | 199.55           | ...   ← température monte !
```

**Pourquoi l'heure ?** C'est le plus petit grain qui réduit le bruit des capteurs tout en conservant les dynamiques de dégradation (qui se jouent sur des heures, pas des minutes).

---

## Étape 3 — Features de mémoire : les fenêtres glissantes

Un modèle ne voit qu'une ligne à la fois. Il faut donc **encoder la mémoire du passé récent** dans chaque ligne. C'est le cœur du feature engineering pour les séries temporelles.

### Principe

Pour chaque signal et chaque horizon (6h, 12h, 24h), on calcule la moyenne, le maximum et l'écart-type des heures précédentes **de la même machine**.

```
         t-6   t-5   t-4   t-3   t-2   t-1    t
temp :  46.3  46.8  47.2  48.1  49.5  51.2  53.7
                                              ▲
                              fenêtre 6h ────┘
temp_mean_6h[t] = mean(46.3, 46.8, 47.2, 48.1, 49.5, 51.2, 53.7) = 49.0
temp_max_6h[t]  = 53.7
temp_std_6h[t]  = 2.5   ← la variabilité augmente, signe d'instabilité !
```

```python
# Implémentation : rolling par groupe machine pour éviter la contamination
hourly['temp_mean_6h'] = hourly.groupby('machine_id_std')['temp_mean_1h'].transform(
    lambda s: s.rolling(6, min_periods=1).mean()
)
```

### Ce que produisent les 3 horizons

| Fenêtre | Capture | Colonnes générées (×4 signaux) |
|---|---|---|
| **6h** | Pics et surchauffes soudaines | `temp_mean_6h`, `temp_max_6h`, `temp_std_6h`, idem pressure/voltage/rotation |
| **12h** | Dérives de mi-journée | idem sur 12h |
| **24h** | Cycle journalier complet | idem sur 24h — baseline de référence |

> **Pourquoi garder les 3 ?** Un modèle arbre (LightGBM) découvrira lui-même lequel est le plus pertinent. Lui présenter les 3 lui donne le choix sans qu'on l'impose.

**⚠️ Piège : ne jamais calculer le rolling cross-machine.** Si MACH-01 et MACH-02 sont dans le même DataFrame sans `groupby('machine_id_std')`, les dernières heures de MACH-01 "contaminent" la fenêtre de MACH-02. Résultat : des features incohérentes et un modèle qui apprend du bruit.

---

## Étape 4 — Features de tendance : détecter la direction

Les fenêtres glissantes donnent un niveau moyen. Elles ne disent pas si la température *monte* ou *descend*. Les features de tendance encodent la **direction et la vitesse** du changement.

### Trend sur 6h (delta absolu)

```python
hourly['temp_trend_6h'] = hourly.groupby('machine_id_std')['temp_mean_1h'].transform(
    lambda s: s - s.shift(6)   # valeur actuelle - valeur il y a 6h
)
```

```
t-6 : 46.3°C     t : 53.7°C     temp_trend_6h = +7.4°C  ← montée marquée
```

### Deltas court-terme (1h et 3h)

Complémentaires du trend 6h, ils capturent les **accélérations** soudaines qui précèdent les pannes aiguës.

```python
hourly['temp_delta_1h'] = hourly.groupby('machine_id_std')['temp_mean_1h'].transform(
    lambda s: s - s.shift(1)   # accélération sur 1h
)
hourly['temp_delta_3h'] = hourly.groupby('machine_id_std')['temp_mean_1h'].transform(
    lambda s: s - s.shift(3)   # tendance sur 3h
)
```

**Interprétation concrète :**

```
Exemple pré-panne (surchauffe sévérité 4) :
  t-3h : temp_mean_1h = 47.1°C
  t-2h : temp_mean_1h = 49.8°C
  t-1h : temp_mean_1h = 52.4°C
  t    : temp_mean_1h = 55.9°C  ← panne déclarée

  temp_delta_1h = +3.5°C   (accélération forte sur 1h)
  temp_delta_3h = +8.8°C   (rampe sur 3h)
  temp_trend_6h = +12.1°C  (contexte sur 6h)
```

Le modèle apprend que `delta_1h > 3°C` combiné à `delta_3h > 8°C` est un signal fort — ce qu'un seul niveau moyen n'aurait pas permis.

---

## Étape 5 — Features d'anomalie : le z-score

Un niveau absolu (53°C) ne dit rien sans contexte. MACH-03 (vieille machine, criticité HIGH) tourne naturellement à 53°C. MACH-07 (récente, criticité MEDIUM) tourne à 45°C. Pour le même modèle, 53°C signifie "normal" pour l'une et "critique" pour l'autre.

### Z-score glissant 24h

Mesure l'écart de la valeur actuelle **par rapport à sa propre moyenne des dernières 24h**.

```
temp_zscore_24h = (temp_mean_1h - temp_mean_24h) / temp_std_24h
```

```
Exemple :
  temp_mean_1h  = 53.7°C    (valeur actuelle)
  temp_mean_24h = 48.2°C    (moyenne des 24h précédentes)
  temp_std_24h  =  2.1°C    (variabilité habituelle)

  temp_zscore_24h = (53.7 - 48.2) / 2.1 = +2.6 σ  ← anomalie significative
```

### Z-score par machine (sans leakage)

Mesure l'écart de la valeur **par rapport à la baseline propre de cette machine**, calculée sur le set d'entraînement uniquement.

```python
# On identifie les lignes d'entraînement (70 % du temps chronologiquement)
_q70 = hourly['window_start'].quantile(0.7)
_is_train = hourly['window_start'] < _q70

# On calcule mean/std de la température sur MACH-01 en train uniquement
train_stats = hourly[_is_train].groupby('machine_id_std')['temp_mean_1h'].agg(['mean','std'])
# → MACH-01 : mean=47.3°C, std=2.8°C
# → MACH-03 : mean=53.1°C, std=3.2°C

# On applique ces stats à TOUTES les lignes (train + val + test)
hourly['temp_zscore_machine'] = (hourly['temp_mean_1h'] - train_mean) / train_std
```

**Pourquoi cette distinction est critique :**

```
❌ Si on calcule la baseline sur TOUTES les données :
   On utilise des informations futures (val/test) pour normaliser le passé.
   Le modèle voit une réalité que le système de production ne verra jamais.
   C'est du data leakage → performances artificiellement gonflées en test.

✅ Si on calcule la baseline uniquement sur le train :
   Le modèle apprend à interpréter les écarts tels qu'il les verrait en prod.
   Les performances mesurées sur val/test sont réalistes.
```

---

## Étape 6 — Features contextuelles : incidents et maintenance

Les capteurs ne racontent qu'une partie de l'histoire. Le contexte opérationnel (incidents récents, maintenance) est tout aussi prédictif.

### 6.1 Incidents passés (lookback)

On compte les incidents survenus dans les fenêtres passées et on cherche la sévérité maximale.

```python
# Jointure des incidents sur l'heure correspondante, puis rolling
hourly['incident_count_prev_24h'] = hourly.groupby('machine_id_std')['incident_count_1h'].transform(
    lambda s: s.rolling(24, min_periods=1).sum()
)
hourly['incident_max_severity_prev_24h'] = ...  # idem avec max()
hourly['incident_count_prev_7d'] = ...          # rolling sur 168h
hourly['hours_since_last_incident'] = ...       # calculé machine par machine
```

**Intuition :** une machine qui a eu 3 incidents de sévérité 3 dans les 7 derniers jours est en train de se dégrader, même si ses capteurs semblent stables à l'instant t.

### 6.2 Types d'incidents (one-hot rolling)

Les 9 types d'incidents (`type_surchauffe`, `type_vibration`, etc.) sont agrégés sur 24h pour créer un historique de la nature des problèmes.

```
type_surchauffe_count_prev_24h = 2   ← 2 épisodes de surchauffe en 24h
type_vibration_count_prev_24h  = 0
```

### 6.3 Maintenance (lookback)

```python
# Pour chaque ligne, on cherche la dernière intervention dans le passé
hourly['days_since_last_maintenance'] = ...   # en jours
hourly['maintenance_count_prev_30d']  = ...   # nombre d'interventions sur 30j
```

**Pourquoi c'est prédictif ?** Une machine qui n'a pas été maintenue depuis 45 jours mais dont la planification recommande une maintenance mensuelle accumule un risque. À l'inverse, une machine maintenue hier a une probabilité de panne beaucoup plus faible.

---

## Étape 7 — Labels : ce que le modèle doit prédire

Le Gold Dataset est supervisé : chaque ligne porte une **étiquette de vérité terrain** qui dit si une panne a eu lieu dans les X heures suivantes.

### Construction par lookahead inversé

On ne peut pas utiliser un rolling "classique" (qui regarde en arrière). Il faut regarder **en avant** dans le temps.

```python
# Astuce : on inverse la série, on fait un rolling en arrière (= en avant dans le temps),
# puis on ré-inverse.
future = hourly.groupby('machine_id_std')['incident_count_1h'].transform(
    lambda s: s[::-1].rolling(6, min_periods=1).sum()[::-1]
)
hourly['label_failure_next_6h'] = future > 0
```

**Illustration :**

```
Heure  | incident_count_1h | label_6h | label_12h | label_24h | label_48h
  t    |         0         |    0     |     0     |     0     |     1
  t+1  |         0         |    0     |     0     |     1     |     1
  t+2  |         0         |    0     |     1     |     1     |     1
  t+3  |         0         |    1     |     1     |     1     |     1
  t+4  |         0         |    1     |     1     |     1     |     1
  t+5  |         0         |    1     |     1     |     1     |     1
  t+6  |         1         |    1     |     1     |     1     |     1   ← panne !
  t+7  |         0         |    0     |     0     |     0     |     0
```

### Pourquoi 4 horizons ?

| Horizon | Utilité industrielle | Difficulté du modèle |
|---|---|---|
| **6h** | Arrêt d'urgence, alerte immédiate | Très difficile — peu de signal précurseur |
| **12h** | Organisation d'une équipe de maintenance | Difficile |
| **24h** | Commande de pièces, planification | Raisonnable |
| **48h** | Planification d'arrêt programmé | Plus facile — signal dégradation visible |

**Choisir l'horizon, c'est choisir un compromis.** Un modèle à 48h sera plus précis mais moins urgent. Un modèle à 6h sera plus urgent mais plus sujet aux faux positifs.

---

## Étape 8 — Split temporel : l'ordre chronologique est sacré

### Pourquoi on n'utilise PAS `train_test_split` aléatoire

```
❌ Split aléatoire (sklearn par défaut) :

  Train : t=1, t=3, t=5, t=8, t=10, t=12 ...
  Test  : t=2, t=4, t=6, t=7, t=9, t=11 ...

  Problème : le modèle voit des données du FUTUR pendant l'entraînement
  (t=12 dans le train alors que t=7 est dans le test).
  Résultat : performances irréalistes.
```

```
✅ Split temporel (utilisé ici) :

  |←────── 70 % ──────→|←── 15 % ──→|←── 15 % ──→|
  |        train        | validation  |    test     |
  2024-01              2025-07      2025-12      2026-06

  La validation simule "3 mois après l'entraînement".
  Le test simule "6 mois après l'entraînement".
  C'est ce que le système verra réellement en production.
```

### Implémentation par quantiles

```python
hourly['split_set'] = np.select(
    [
        hourly['window_start'] < hourly['window_start'].quantile(0.70),
        hourly['window_start'] < hourly['window_start'].quantile(0.85),
    ],
    ['train', 'validation'],
    default='test',
)
```

**Le seuil 0.70 est le même que celui utilisé pour le z-score par machine.** Ce n'est pas un hasard : les deux doivent être cohérents pour que les stats de normalisation correspondent exactement aux données vues pendant l'entraînement.

---

## La ligne Gold complète

Voici ce que contient **une seule ligne** du Gold Dataset, pour MACH-01 à t=2025-11-15 03:00:

```
Identifiants
  machine_id_std    : MACH-01
  window_start      : 2025-11-15 03:00:00
  window_end        : 2025-11-15 04:00:00
  split_set         : validation

Agrégation horaire brute (non utilisée en entraînement, exclue des features)
  temp_mean_1h      : 54.2°C
  pressure_mean_1h  : 196.8 bar

Features de mémoire — température
  temp_mean_6h      : 51.3°C     ← niveau moyen sur 6h
  temp_max_6h       : 54.2°C     ← pic sur 6h
  temp_std_6h       : 1.9°C      ← instabilité croissante
  temp_mean_12h     : 49.1°C
  temp_mean_24h     : 48.0°C     ← baseline journalière

Features de tendance
  temp_trend_6h     : +7.8°C     ← montée marquée sur 6h
  temp_delta_1h     : +2.4°C     ← accélération sur 1h
  temp_delta_3h     : +5.9°C     ← rampe sur 3h

Features d'anomalie
  temp_zscore_24h    : +2.3σ     ← écart par rapport aux dernières 24h
  temp_zscore_machine: +2.8σ     ← écart par rapport à la baseline de MACH-01

Features contextuelles — incidents
  incident_count_prev_24h        : 2
  incident_max_severity_prev_24h : 3
  incident_count_prev_7d         : 5
  hours_since_last_incident      : 18.0h
  type_surchauffe_count_prev_24h : 2

Features contextuelles — maintenance
  days_since_last_maintenance    : 38.5 jours  ← maintenance en retard
  maintenance_count_prev_30d     : 0

Labels (vérité terrain)
  label_failure_next_6h  : True   ← panne dans les 6h  ✓
  label_failure_next_12h : True
  label_failure_next_24h : True
  label_failure_next_48h : True
```

**Ce qu'un modèle doit apprendre :** quand `temp_zscore_machine > 2` ET `temp_delta_3h > 5` ET `incident_count_prev_24h >= 2` ET `days_since_last_maintenance > 30`, la probabilité d'une panne dans les 6h est élevée. Aucune de ces features seule n'est suffisante — c'est leur combinaison qui est prédictive.

---

## Tableau récapitulatif des 100 colonnes Gold

| Famille | Colonnes | Nb | Rôle pour le modèle |
|---|---|---|---|
| **Identifiants** | `machine_id_std`, `window_start`, `window_end`, `split_set` | 4 | Non utilisés en entraînement |
| **Agrégats 1h** | `temp_mean_1h`, `temp_max_1h`, ... (×4 signaux) | 9 | Non utilisés (trop bruités) |
| **Rolling 6h** | `temp_mean_6h`, `temp_max_6h`, `temp_std_6h`, ... (×4) | 12 | Pics et surchauffes soudaines |
| **Rolling 12h** | idem (×4) | 12 | Dérives de mi-journée |
| **Rolling 24h** | idem (×4) | 12 | Cycle journalier, baseline |
| **Tendances** | `*_trend_6h` (×4), `*_delta_1h`, `*_delta_3h` (×4) | 12 | Direction et vitesse du changement |
| **Anomalies** | `temp_zscore_24h`, `temp_zscore_machine`, `pressure_zscore_machine` | 3 | Déviation par rapport à la normale |
| **Production** | `pieces_produced_sum_24h`, `capacity_utilization_pct` | 2 | Charge de la machine |
| **Incidents passés** | count/sévérité sur 24h et 7j, `hours_since_last_incident` | 4 | Historique de santé |
| **Types incidents** | `type_*` (9 types) + rolling 24h (×9) | 18 | Nature des problèmes |
| **Maintenance** | `days_since_last_maintenance`, `maintenance_count_prev_30d` | 2 | Risque de retard de maintenance |
| **Labels** | `label_failure_next_6h/12h/24h/48h` + `future_incident_count_*` | 8 | Ce qu'on cherche à prédire |

---

## Les règles à ne jamais violer

### Règle 1 — Pas de données futures dans les features

Toute feature qui utilise des informations postérieures à `window_start` est du leakage.

```
✅ temp_mean_6h → rolling sur les 6h PRÉCÉDENTES
❌ temp_mean_6h → rolling sur les 3h précédentes + les 3h suivantes
```

### Règle 2 — Toujours grouper par machine

Les fenêtres glissantes, les z-scores et les lookback d'incidents doivent être calculés **intra-machine**. Deux machines différentes n'ont aucune raison de partager un contexte temporel.

### Règle 3 — Fitter les statistiques uniquement sur le train

Moyenne, écart-type, médiane d'imputation : tout ce qui "apprend" quelque chose sur les données doit être calculé exclusivement sur le train set, puis appliqué tel quel à val et test.

### Règle 4 — Respecter l'ordre chronologique du split

Ne jamais mélanger les splits. La validation est dans le futur par rapport au train. Le test est dans le futur par rapport à la validation. C'est ce qui rend les métriques fiables.

---

## Accès au Gold Dataset en pratique

Le Gold Dataset est persisté en base PostgreSQL via SQLAlchemy. Pour le recharger dans un notebook sans relancer tout le pipeline d'ingestion :

```python
from indusense.db.gold_loader import load_gold_from_db

gold = load_gold_from_db()   # DataFrame identique à celui produit par le pipeline
print(f"{gold.shape[0]:,} lignes × {gold.shape[1]} colonnes")
print(gold['split_set'].value_counts())
```

Le CSV `datas/gold_dataset_YYYYMMDDHHMMSS.csv` sert de traçabilité et peut être rechargé via `pd.read_csv()`, mais la source canonique pour l'entraînement est la base de données.

---

*Document généré à partir du pipeline `src/indusense/processing/ingestion.py` — fonction `build_gold_from_telemetry()`.*
