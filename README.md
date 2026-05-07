# InduSense

## Modele cible pour l'ingestion PostgreSQL

L'exploration de `datas` met en evidence trois contraintes structurantes pour la suite du projet :
- les trois sources ont des separateurs et des formats heterogenes ;
- les identifiants machine doivent etre normalises avant toute jointure ;
- la fenetre commune exploitable pour le croisement capteurs/incidents est centree sur la periode `2025-08-26` a `2026-02-25`, avec `15` machines communes apres harmonisation.

L'objectif de la prochaine etape n'est donc pas de charger directement les CSV dans une table finale, mais de preparer un modele relationnel qui permette :
- l'atterrissage brut des fichiers ;
- la tracabilite des traitements et des rejets ;
- la construction de tables Silver normalisees ;
- la production d'un Gold dataset horaire, pret pour l'entrainement et protege contre le data leakage.

## Operations a realiser avant l'implementation Alembic / SQLAlchemy

1. Creer une couche `Bronze` qui conserve les lignes brutes, la provenance des fichiers et le statut d'ingestion.
2. Normaliser les `machine_id`, les timestamps et les types de donnees avant de peupler la couche `Silver`.
3. Pseudonymiser les informations operateur des incidents avant de sortir des tables brutes.
4. Isoler les doublons, lignes invalides, timestamps mal formes et valeurs suspectes dans une table de qualite de donnees.
5. Unifier les capteurs temperature et pression dans un fait Silver horodate par machine.
6. Construire un Gold dataset par fenetre temporelle glissante, centre sur `machine x heure`, avec variables explicatives et label cible.
7. Conserver la notion de `split_set` dans le Gold pour verrouiller les futurs jeux `train`, `validation` et `test` temporels.

## Proposition Mermaid

```mermaid
erDiagram
    INGESTION_BATCH ||--o{ BRONZE_TEMPERATURE_RAW : charge
    INGESTION_BATCH ||--o{ BRONZE_PRESSURE_RAW : charge
    INGESTION_BATCH ||--o{ BRONZE_INCIDENT_RAW : charge
    INGESTION_BATCH ||--o{ DATA_QUALITY_ISSUE : produit

    MACHINE ||--o{ SILVER_SENSOR_READING : emet
    MACHINE ||--o{ SILVER_INCIDENT : subit
    OPERATOR ||--o{ SILVER_INCIDENT : declare
    MACHINE ||--o{ GOLD_MACHINE_HOURLY_FEATURE : alimente

    INGESTION_BATCH {
        uuid ingestion_batch_id PK
        string source_name
        string source_file
        timestamptz started_at
        timestamptz finished_at
        int rows_read
        int rows_loaded
        int rows_rejected
        string status
    }

    MACHINE {
        bigint machine_id PK
        string machine_code UK
        date commissioning_date
        int max_daily_capacity
        string model
        string production_line
        string location
        string criticality
        boolean is_active
    }

    OPERATOR {
        bigint operator_id PK
        string operator_key UK
        string badge_hash
        boolean is_active
    }

    BRONZE_TEMPERATURE_RAW {
        bigint temperature_raw_id PK
        uuid ingestion_batch_id FK
        int row_number
        string machine_id_raw
        string timestamp_raw
        string temperature_raw
        boolean parse_ok
    }

    BRONZE_PRESSURE_RAW {
        bigint pressure_raw_id PK
        uuid ingestion_batch_id FK
        int row_number
        string machine_id_raw
        string timestamp_raw
        string pressure_raw
        boolean parse_ok
    }

    BRONZE_INCIDENT_RAW {
        bigint incident_raw_id PK
        uuid ingestion_batch_id FK
        int row_number
        string incident_code_raw
        string machine_id_raw
        string operator_name_raw
        string operator_badge_raw
        string occurred_at_raw
        string severity_raw
        boolean parse_ok
    }

    DATA_QUALITY_ISSUE {
        bigint dq_issue_id PK
        uuid ingestion_batch_id FK
        string dataset_name
        string rule_code
        string severity
        string entity_key
        text details
    }

    SILVER_SENSOR_READING {
        bigint sensor_reading_id PK
        bigint machine_id FK
        timestamptz observed_at
        string sensor_type
        numeric sensor_value
        string unit
        boolean is_missing
        boolean is_duplicate
        boolean is_outlier
        uuid ingestion_batch_id FK
    }

    SILVER_INCIDENT {
        bigint incident_id PK
        string incident_code UK
        bigint machine_id FK
        bigint operator_id FK
        timestamptz occurred_at
        smallint severity
        string shift
        text comment
        boolean is_label_event
        uuid ingestion_batch_id FK
    }

    GOLD_MACHINE_HOURLY_FEATURE {
        bigint feature_row_id PK
        bigint machine_id FK
        timestamptz window_start
        timestamptz window_end
        numeric temp_mean_24h
        numeric temp_max_24h
        numeric temp_std_24h
        numeric pressure_mean_24h
        numeric pressure_std_24h
        int incident_count_prev_24h
        int incident_max_severity_prev_24h
        boolean label_failure_next_24h
        string split_set
    }
```

## Lecture proposee du modele

- `Bronze` preserve la verite source, y compris les identifiants heterogenes, les formats d'horodatage et les lignes rejetables.
- `Silver` porte les faits nettoyes et normalises, avec les drapeaux utiles a l'analyse qualite et a l'explicabilite.
- `Gold` represente une table de features temporelles, a granularite stable, adaptee a un apprentissage supervise de prediction de panne.

## Decision de modelisation a retravailler ensemble

- conserver ou non une cle technique `machine_id` en plus de `machine_code` ;
- choisir la granularite cible du Gold : heure, quart d'heure, jour ;
- definir l'horizon du label cible : panne dans `6h`, `12h`, `24h` ou `48h` ;
- fixer les regles exactes de dedoublonnage et de gestion des valeurs manquantes avant le passage en Silver.
