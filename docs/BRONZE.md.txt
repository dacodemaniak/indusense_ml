# Couche Bronze — Du CSV brut à la base de données

> **À qui s'adresse ce document ?**
> Aux étudiants qui vont construire le Bronze Dataset à partir des fichiers CSV sources. On y explique *quoi faire*, *comment le faire* et surtout *pourquoi* on le fait ainsi. Chaque section correspond à une ou plusieurs cellules dans le notebook ou dans le dossier src.

---

## Vue d'ensemble : qu'est-ce que le Bronze ?

```
telemetry.csv          releves_incidents.csv
(capteurs bruts)       (incidents opérateurs)
       │                        │
       ▼                        ▼
  ┌─────────────────────────────────────────────────────┐
  │                     BRONZE                         │
  │  • Tout est stocké tel quel (aucune modification)   │
  │  • Chaque ligne est validée et taguée parse_ok      │
  │  • Les rejets sont conservés avec leur motif        │
  │  • Un batch d'ingestion trace chaque chargement     │
  └─────────────────────────────────────────────────────┘
```

**La règle d'or du Bronze : on ne jette rien.** Une ligne qui ne passe pas la validation n'est pas supprimée — elle est insérée avec `parse_ok = False` et un `rejected_reason` explicite. Cela permet de tracer les problèmes de qualité à la source et de les corriger sans relancer tout le pipeline.

---

## Partie 1 — Les données sources

Avant de charger quoi que ce soit, regardons les fichiers sources. Relire le début du document `gold_roadmap.md` pour bien les comprendre.

### 1.1 `telemetry.csv` — mesures capteurs

```python
# Cellule 1 — Explorer telemetry.csv
import pandas as pd

tel = pd.read_csv("datas/telemetry.csv")
print(tel.shape)
tel.head(4)
```

```
(~50 000 lignes, 7 colonnes)

machine_id | timestamp            | temperature_c | pressure_bar | voltage_mean_v | rotation_mean_rpm | pieces_produced
MACH-01    | 2025-06-01 00:00:00  | 46.348        | 198.203      | 227.568        | 1541.787          | 4
MACH-01    | 2025-06-01 00:00:00  | 46.332        | 198.206      | 227.570        | 1541.760          | 4   ← doublon !
MACH-01    | 2025-06-01 01:00:00  | 48.762        | 198.295      | 227.480        | 1537.860          | 4
```

**Ce qu'on voit :**
- Les deux premières lignes ont le même `machine_id` et le même `timestamp` → doublon capteur.
- Tout est numérique et semble propre… mais on ne sait pas encore si certaines températures sont en Fahrenheit.
- La colonne `pieces_produced` est un entier.

### 1.2 `releves_incidents.csv` — déclarations opérateurs

```python
# Cellule 2 — Explorer releves_incidents.csv
inc = pd.read_csv("datas/releves_incidents.csv")
print(inc.shape)
inc.head(3)
```

```
(~2 000 lignes, 19 colonnes)

incident_id | date       | time  | machine_id | severity | type_surchauffe | type_vibration | ...
INC-000001  | 2025-06-01 | 05:42 | MACH-06    | 4        | 1               | 0              | ...
INC-000002  | 2025-06-01 | 21:08 | MACH-15    | 3        | 0               | 0              | ...
```

**Ce qu'on voit :**
- La date et l'heure sont dans **deux colonnes séparées** — il faudra les fusionner.
- `severity` doit être un entier entre 1 et 5.
- Les 9 colonnes `type_*` sont des flags 0/1 décrivant la nature du problème.

---

## Partie 2 — Le schéma de base de données

### 2.1 Pourquoi SQLAlchemy ?

SQLAlchemy est un ORM (*Object-Relational Mapper*) : il traduit des classes Python en tables SQL. Au lieu d'écrire `CREATE TABLE bronze_telemetry_raw (...)` à la main, on déclare une classe Python et l'ORM génère le SQL.

Le fichier [src/indusense/db/models.py](../src/indusense/db/models.py) contient tous les modèles du projet. Voici la structure Bronze qui nous intéresse :

```python
# Extrait de src/indusense/db/models.py

class BronzeTelemetryRaw(TimestampMixin, Base):
    __tablename__ = "bronze_telemetry_raw"

    telemetry_raw_id: Mapped[int]          # clé primaire auto-incrémentée
    ingestion_batch_id: Mapped[uuid.UUID]  # lien vers le batch d'ingestion
    row_number: Mapped[int]                # numéro de ligne dans le CSV source
    machine_id_raw: Mapped[str | None]     # valeur brute, telle quelle
    timestamp_raw: Mapped[str | None]      # idem
    temperature_raw: Mapped[str | None]    # tout est stocké en STRING
    pressure_raw: Mapped[str | None]
    voltage_raw: Mapped[str | None]
    rotation_raw: Mapped[str | None]
    pieces_raw: Mapped[str | None]
    parse_ok: Mapped[bool]                 # False si la validation a échoué
    rejected_reason: Mapped[str | None]    # message d'erreur si parse_ok = False
```

> **Pourquoi tout stocker en `str` (STRING) ?**
> Parce que le Bronze est un miroir fidèle du CSV. Si le CSV contient `"46.348"`, on stocke `"46.348"`. La conversion en `float`, le contrôle de cohérence, la correction d'unités — tout ça appartient au Silver. Le Bronze ne transforme pas ; il archive.

De même pour les incidents (`BronzeIncidentRaw`) :

```python
class BronzeIncidentRaw(TimestampMixin, Base):
    __tablename__ = "bronze_incident_raw"

    incident_raw_id: Mapped[int]
    ingestion_batch_id: Mapped[uuid.UUID]
    row_number: Mapped[int]
    incident_code_raw: Mapped[str | None]
    machine_id_raw: Mapped[str | None]
    occurred_at_raw: Mapped[str | None]    # date + time fusionnés en une string
    severity_raw: Mapped[str | None]
    shift_raw: Mapped[str | None]
    comment_raw: Mapped[str | None]
    type_surchauffe: Mapped[int]           # exception : les flags 0/1 sont déjà typés
    # ... (8 autres colonnes type_*)
    parse_ok: Mapped[bool]
    rejected_reason: Mapped[str | None]
```

### 2.2 Pourquoi Alembic ?

Alembic est l'outil de **migration de schéma** : il versionne les modifications de la base de données, exactement comme `git` versionne le code.

```
alembic/versions/
├── 20260507_0001_init_bronze_silver_gold.py   ← création initiale de toutes les tables
├── 20260512_0002_add_gold_rolling_windows.py  ← ajout de colonnes au Gold
├── 20260513_0003_add_incident_types_and_weather.py
├── 20260611_0004_add_telemetry_and_maintenance.py
└── 20260617_0005_add_delta_and_zscore_features.py
```

Chaque fichier est une migration : il contient une fonction `upgrade()` (appliquer le changement) et une fonction `downgrade()` (l'annuler). Alembic garde en base une table `alembic_version` qui indique quelle migration a été appliquée en dernier.

```python
# Cellule 3 — Appliquer toutes les migrations (créer les tables)
# À exécuter UNE SEULE FOIS (ou après un reset de la base)
import subprocess
result = subprocess.run(["alembic", "upgrade", "head"], capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
```

**Ce que fait `upgrade head` :**
1. Lit `alembic_version` pour voir où on en est.
2. Applique dans l'ordre toutes les migrations manquantes jusqu'à `head` (la plus récente).
3. Crée toutes les tables (`ingestion_batch`, `bronze_telemetry_raw`, `bronze_incident_raw`, etc.).

---

## Partie 3 — Connexion à la base

### 3.1 La session SQLAlchemy

```python
# Cellule 4 — Ouvrir une connexion
from indusense.db.session import SessionLocal

session = SessionLocal()
print("Connexion établie :", session.bind)
```

`SessionLocal` est une fabrique de sessions configurée dans [src/indusense/db/session.py](../src/indusense/db/session.py). Une session est l'équivalent d'une transaction : tout ce qu'on écrit dedans est en attente jusqu'au `session.commit()`.

```
SessionLocal()
    │
    ├── autoflush=False   → les INSERT ne partent pas automatiquement
    ├── autocommit=False  → on contrôle manuellement le commit
    └── expire_on_commit=False → les objets restent accessibles après commit
```

### 3.2 Tester la connexion

```python
# Cellule 5 — Vérifier que la base est accessible et les tables existent
import sqlalchemy as sa
from indusense.db.session import create_postgres_engine

engine = create_postgres_engine()
with engine.connect() as conn:
    tables = sa.inspect(engine).get_table_names()
    print("Tables créées :", tables)
```

Vous devez voir apparaître : `ingestion_batch`, `bronze_telemetry_raw`, `bronze_incident_raw`, `machine`, `silver_sensor_reading`, `gold_machine_hourly_feature`, etc.

---

## Partie 4 — La validation avec Pydantic

Avant d'insérer une ligne en base, on la **valide** avec Pydantic. Pydantic est une bibliothèque de validation de données : on déclare ce qu'on attend, et elle lève une `ValidationError` si la donnée ne correspond pas.

Le fichier [src/indusense/schemas/ingestion.py](../src/indusense/schemas/ingestion.py) définit les schémas de validation Bronze.

### 4.1 Validation d'une ligne de télémétrie

```python
# Cellule 6 — Comprendre la validation Pydantic
from indusense.schemas.ingestion import TemperatureInput

# ✅ Ligne valide
row_ok = TemperatureInput(
    machine_id_raw="MACH-01",
    timestamp_raw="2025-06-01 00:00:00",
    temperature_raw="46.348",
)
print("machine_code :", row_ok.machine_code)   # → MACH-01
print("observed_at  :", row_ok.observed_at)    # → 2025-06-01 00:00:00
print("temperature  :", row_ok.temperature)    # → Decimal('46.348')
```

```python
# ❌ Ligne invalide — machine_id illisible
from pydantic import ValidationError

try:
    TemperatureInput(
        machine_id_raw="???",
        timestamp_raw="2025-06-01 00:00:00",
        temperature_raw="46.348",
    )
except ValidationError as e:
    print("Rejeté :", e)
```

**Ce que fait `TemperatureInput` :**
1. Vérifie que `machine_id_raw` contient des chiffres (`MACH-01` → extrait `01` → normalise en `MACH-01`).
2. Parse `timestamp_raw` en `datetime` Python.
3. Convertit `temperature_raw` en `Decimal` (précision exacte, pas de flottant).
4. Si l'une de ces étapes échoue → `ValidationError`.

### 4.2 Validation d'un incident

```python
# Cellule 7 — Validation d'un incident
from indusense.schemas.ingestion import IncidentInput

# ✅ Incident valide
inc_ok = IncidentInput(
    incident_code_raw="INC-000001",
    machine_id_raw="MACH-06",
    date_raw="2025-06-01",
    time_raw="05:42",
    severity_raw="4",
    shift_raw="nuit",
    comment_raw="chauffe anormale",
    type_surchauffe=1,
)
print("occurred_at :", inc_ok.occurred_at)   # → 2025-06-01 05:42:00
print("severity    :", inc_ok.severity)      # → 4

# ❌ Sévérité hors plage
try:
    IncidentInput(
        incident_code_raw="INC-000099",
        machine_id_raw="MACH-01",
        date_raw="2025-06-01",
        time_raw="08:00",
        severity_raw="9",           # ← invalide : doit être entre 1 et 5
    )
except ValidationError as e:
    print("Rejeté :", e)
```

---

## Partie 5 — L'IngestionBatch : traçabilité complète

Chaque chargement crée un **batch d'ingestion** dans la table `ingestion_batch`. C'est le journal de bord du pipeline.

```python
# Extrait de src/indusense/pipeline/bronze.py

def _create_batch(session, source_name, source_file):
    batch = IngestionBatch(
        ingestion_batch_id=uuid.uuid4(),   # identifiant unique (UUID)
        source_name=source_name,           # ex: "telemetry"
        source_file=source_file,           # ex: "telemetry.csv"
        started_at=datetime.now(tz=utc),
        status=IngestionStatus.RUNNING,    # en cours
    )
    session.add(batch)
    session.flush()    # envoie l'INSERT sans committer
    return batch

def _finish_batch(session, batch, rows_read, rows_loaded, rows_rejected):
    batch.finished_at = datetime.now(tz=utc)
    batch.rows_read     = rows_read
    batch.rows_loaded   = rows_loaded
    batch.rows_rejected = rows_rejected
    # PARTIAL si des lignes ont été rejetées, COMPLETED sinon
    batch.status = COMPLETED if rows_rejected == 0 else PARTIAL
    session.flush()
```

À la fin d'un batch, on peut lire directement depuis la base :

```python
# Cellule 8 — Inspecter les batches d'ingestion
from indusense.db.models import IngestionBatch
from indusense.db.session import SessionLocal

with SessionLocal() as session:
    batches = session.query(IngestionBatch).order_by(IngestionBatch.started_at.desc()).limit(5).all()
    for b in batches:
        print(f"{b.source_name:12s} | {b.status.value:10s} | "
              f"{b.rows_loaded:>6} chargées | {b.rows_rejected:>4} rejetées")
```

```
telemetry    | completed  |  49873 chargées |    0 rejetées
incidents    | partial    |   1987 chargées |   13 rejetées
```

---

## Partie 6 — Charger les fichiers CSV dans le Bronze

Les fonctions d'ingestion sont dans [src/indusense/pipeline/bronze.py](../src/indusense/pipeline/bronze.py).

### 6.1 Charger la télémétrie

```python
# Cellule 9 — Ingestion de telemetry.csv
from pathlib import Path
from indusense.pipeline.bronze import load_telemetry_file
from indusense.db.session import SessionLocal

CSV_PATH = Path("datas/telemetry.csv")

with SessionLocal() as session:
    batch = load_telemetry_file(session, CSV_PATH)
    session.commit()

print(f"Batch ID  : {batch.ingestion_batch_id}")
print(f"Status    : {batch.status.value}")
print(f"Lues      : {batch.rows_read}")
print(f"Chargées  : {batch.rows_loaded}")
print(f"Rejetées  : {batch.rows_rejected}")
```

**Ce que fait `load_telemetry_file` ligne par ligne :**

```python
# Principe simplifié (extrait pédagogique)

df = pd.read_csv(path, dtype=str)   # 1. Tout lire en string — aucune conversion implicite

rows = []
for row_num, row in enumerate(df.itertuples(), start=1):
    raw = {
        "ingestion_batch_id": batch.ingestion_batch_id,
        "row_number": row_num,
        "machine_id_raw": row.machine_id,
        "timestamp_raw": row.timestamp,
        "temperature_raw": row.temperature_c,
        # ...
        "parse_ok": False,          # 2. Pessimiste par défaut
        "rejected_reason": None,
    }
    try:
        TelemetryInput(...)         # 3. Validation Pydantic
        raw["parse_ok"] = True      # 4. Succès → on marque la ligne valide
    except ValidationError as exc:
        raw["rejected_reason"] = str(exc)[:512]   # 5. Échec → on garde le motif

    rows.append(raw)

session.execute(sa.insert(BronzeTelemetryRaw), rows)   # 6. INSERT en masse (bulk)
```

> **Pourquoi `dtype=str` dans `pd.read_csv` ?**
> Par défaut, pandas convertit `"46.348"` en `float64`, ce qui peut introduire des erreurs d'arrondi ou interpréter `""` comme `NaN`. En lisant tout en `str`, on préserve exactement ce qui était dans le CSV et on délègue l'interprétation à Pydantic.

### 6.2 Charger les incidents

```python
# Cellule 10 — Ingestion de releves_incidents.csv
from indusense.pipeline.bronze import load_incidents_file

CSV_INC = Path("datas/releves_incidents.csv")

with SessionLocal() as session:
    batch_inc = load_incidents_file(session, CSV_INC)
    session.commit()

print(f"Incidents chargés  : {batch_inc.rows_loaded}")
print(f"Incidents rejetés  : {batch_inc.rows_rejected}")
```

---

## Partie 7 — Inspecter le Bronze

### 7.1 Vérifier les lignes valides

```python
# Cellule 11 — Compter les lignes valides / rejetées dans bronze_telemetry_raw
from indusense.db.models import BronzeTelemetryRaw
from sqlalchemy import func

with SessionLocal() as session:
    total     = session.query(func.count(BronzeTelemetryRaw.telemetry_raw_id)).scalar()
    valid     = session.query(func.count(BronzeTelemetryRaw.telemetry_raw_id))\
                       .filter(BronzeTelemetryRaw.parse_ok == True).scalar()
    rejected  = total - valid

    print(f"Total    : {total:>8,}")
    print(f"Valides  : {valid:>8,}  ({100*valid/total:.1f} %)")
    print(f"Rejetées : {rejected:>8,}  ({100*rejected/total:.1f} %)")
```

### 7.2 Inspecter les rejets

```python
# Cellule 12 — Afficher les 10 premières lignes rejetées
from indusense.db.models import BronzeIncidentRaw

with SessionLocal() as session:
    rejected_rows = (
        session.query(BronzeIncidentRaw)
        .filter(BronzeIncidentRaw.parse_ok == False)
        .limit(10)
        .all()
    )
    for r in rejected_rows:
        print(f"Ligne {r.row_number:>5} | machine={r.machine_id_raw!r:12s} "
              f"| severity={r.severity_raw!r} | motif: {r.rejected_reason[:80]}")
```

```
Ligne    42 | machine='MACH-06'    | severity='6'  | motif: incident severity must be between 1 and 5
Ligne   187 | machine=''           | severity='3'  | motif: machine_id does not contain digits
```

Ces lignes **existent en base** avec `parse_ok = False`. Elles ne seront pas traitées par le Silver, mais elles permettent de remonter les anomalies à l'équipe de qualité.

### 7.3 Vérifier avec SQL brut (alternative)

```python
# Cellule 13 — Requête SQL directe pour vérifier les types d'erreurs
from indusense.db.session import create_postgres_engine
import pandas as pd

engine = create_postgres_engine()
query = """
SELECT
    LEFT(rejected_reason, 60)  AS motif,
    COUNT(*)                   AS nb
FROM bronze_incident_raw
WHERE parse_ok = FALSE
GROUP BY 1
ORDER BY nb DESC
LIMIT 10;
"""
df_rejets = pd.read_sql(query, engine)
print(df_rejets.to_string(index=False))
```

---

## Partie 8 — Résumé : ce que contient le Bronze

À ce stade, votre base contient :

| Table | Contenu | Colonnes clés |
|---|---|---|
| `ingestion_batch` | Journal de chaque chargement | `status`, `rows_loaded`, `rows_rejected` |
| `bronze_telemetry_raw` | Toutes les lignes de `telemetry.csv` | `machine_id_raw`, `timestamp_raw`, `temperature_raw`, `parse_ok` |
| `bronze_incident_raw` | Toutes les lignes de `releves_incidents.csv` | `machine_id_raw`, `occurred_at_raw`, `severity_raw`, `parse_ok` |

**Invariants garantis :**
- Toutes les lignes sources sont présentes (aucune suppression).
- Chaque ligne porte `parse_ok` et `rejected_reason`.
- Chaque ligne est rattachée à un batch via `ingestion_batch_id`.
- Les données sont stockées en `str` — aucune transformation n'a eu lieu.

---

## Partie 9 — Ce qui vous attend dans le Silver

Le Bronze est une archive fidèle. Le Silver va nettoyer :

| Problème Bronze | Traitement Silver |
|---|---|
| Doublons (même machine, même timestamp) | Déduplication : on garde la première occurrence |
| Températures potentiellement en Fahrenheit | Détection et conversion `°F → °C` |
| Valeurs manquantes (blocs de 4-12h) | Imputation : médiane, interpolation ou régression |
| `machine_id_raw` non normalisé (`"machine-1"`, `"M01"`, etc.) | Normalisation en `MACH-01` |
| Date et heure séparées dans les incidents | Fusion en `occurred_at` (datetime avec timezone) |

La prochaine étape est donc : **lire le Silver depuis le Bronze validé** (`parse_ok = True`), appliquer ces corrections, et persister dans les tables `silver_telemetry_reading` et `silver_incident`.

```python
# Aperçu de ce que fera le Silver (ne pas exécuter maintenant)
from indusense.pipeline.silver import build_silver_from_bronze

with SessionLocal() as session:
    build_silver_from_bronze(session)
    session.commit()
```

---

*Les fonctions d'ingestion complètes sont dans [`src/indusense/pipeline/bronze.py`](../src/indusense/pipeline/bronze.py). Les modèles SQLAlchemy sont dans [`src/indusense/db/models.py`](../src/indusense/db/models.py). Les schémas Pydantic sont dans [`src/indusense/schemas/ingestion.py`](../src/indusense/schemas/ingestion.py).*
