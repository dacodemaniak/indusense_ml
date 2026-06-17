#!/usr/bin/env python3
"""
InduSense — Dataset regeneration (H1 → H9).

Run from project root:
    python scripts/regenerate_dataset.py

Transformations:
  H1  Degradation ramps in telemetry before sev >= 3 incidents
  H2  Machine-specific normal operating baselines (distinct fingerprints)
  H3  Realistic missing values: sensor failure NaN blocs (2-3 % total)
  H4  Controlled duplicates: double-transmission noise (0.5 %)
  H5  Severity rebalancing: more sev 3-5, less sev 1
  H6  Pre-failure clusters: 2 precursor incidents before each sev 4-5
  H7  Realistic timestamps (not always :00) + coherent comments
  H8  Machine age × criticality correlated with incident frequency
  H9  Reactive maintenance in SQL after sev >= 3 incidents
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
RNG  = np.random.default_rng(SEED)
DATA_DIR   = Path("datas")
BACKUP_DIR = DATA_DIR / "_originals"

# ── Machine reference ─────────────────────────────────────────────────────────
COMMISSIONED = {
    'MACH-01': 2021, 'MACH-02': 2024, 'MACH-03': 2019, 'MACH-04': 2023,
    'MACH-05': 2024, 'MACH-06': 2022, 'MACH-07': 2025, 'MACH-08': 2023,
    'MACH-09': 2023, 'MACH-10': 2021, 'MACH-11': 2022, 'MACH-12': 2024,
    'MACH-13': 2019, 'MACH-14': 2021, 'MACH-15': 2022,
}
CRITICALITY = {
    'MACH-01': 'MEDIUM', 'MACH-02': 'LOW',    'MACH-03': 'HIGH',  'MACH-04': 'LOW',
    'MACH-05': 'HIGH',   'MACH-06': 'LOW',    'MACH-07': 'MEDIUM','MACH-08': 'HIGH',
    'MACH-09': 'MEDIUM', 'MACH-10': 'LOW',    'MACH-11': 'MEDIUM','MACH-12': 'MEDIUM',
    'MACH-13': 'MEDIUM', 'MACH-14': 'LOW',    'MACH-15': 'MEDIUM',
}
AGE       = {m: 2026 - y for m, y in COMMISSIONED.items()}
CRIT_RANK = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}

TYPE_COLS = [
    'type_surchauffe', 'type_baisse_pression', 'type_vibration',
    'type_bruit_mecanique', 'type_surconsommation', 'type_blocage_mecanique',
    'type_alarme_capteur', 'type_arret_urgence', 'type_defaut_qualite',
]

SIGNAL_BOUNDS = {
    'temperature_c':      (32.0,  80.0),
    'pressure_bar':      (160.0, 220.0),
    'voltage_mean_v':    (220.0, 242.0),
    'rotation_mean_rpm': (1100.0,1900.0),
    'pieces_produced':   (  0.0, 120.0),
}

# (signal, max_drift_or_mode, lead_hours) — stacks if multiple types active
TYPE_EFFECTS: dict[str, list] = {
    'type_surchauffe':       [('temperature_c',      +9.0,  24), ('rotation_mean_rpm',  +60.0, 12)],
    'type_baisse_pression':  [('pressure_bar',       -12.0, 18), ('temperature_c',       +2.5,  8)],
    'type_vibration':        [('rotation_mean_rpm', 'noise', 14)],
    'type_bruit_mecanique':  [('rotation_mean_rpm', 'spikes', 8)],
    'type_surconsommation':  [('voltage_mean_v',     +5.0,  18), ('temperature_c',       +3.0, 12)],
    'type_blocage_mecanique':[('rotation_mean_rpm', -220.0,  4), ('pressure_bar',        +5.0,  2)],
    'type_alarme_capteur':   [('voltage_mean_v',     +3.0,   8)],
    'type_arret_urgence':    [
        ('temperature_c',     +8.0, 12), ('pressure_bar',     -9.0,  6),
        ('rotation_mean_rpm',-280.0,  3), ('voltage_mean_v',   +4.0,  8),
    ],
    'type_defaut_qualite':   [('pieces_produced',  -30.0, 16), ('rotation_mean_rpm', -80.0,  8)],
}

SEV_SCALE = {1: 0.15, 2: 0.30, 3: 0.60, 4: 0.85, 5: 1.0}

TYPE_COMMENTS: dict[str, list[str]] = {
    'type_surchauffe':       [
        'température anormalement élevée',        'surchauffe capteur thermique',
        'montée thermique progressive',            'alerte température haute',
    ],
    'type_baisse_pression':  [
        'baisse pression hydraulique',             'pression en dessous seuil critique',
        'micro-fuite détectée — chute pression',   'circuit hydraulique en défaut',
    ],
    'type_vibration':        [
        'vibrations excessives détectées',         'oscillations hors tolérance',
        'balourd rotor confirmé',                  'vibration anormale roulement',
    ],
    'type_bruit_mecanique':  [
        'bruit mécanique anormal',                 'cliquetis roulement axe',
        'frottement mécanique cyclique',           'choc sur transmission',
    ],
    'type_surconsommation':  [
        'surconsommation électrique constatée',    'pic courant anormal moteur',
        'consommation hors plage nominale',        'échauffement bobinage moteur',
    ],
    'type_blocage_mecanique':[
        'blocage axe principal',                   'résistance mécanique anormale',
        'coincement pièce mobile',                 'enrayement transmission',
    ],
    'type_alarme_capteur':   [
        'alarme capteur active',                   'signal capteur hors plage',
        'perte intermittente signal mesure',       'défaut capteur confirmé',
    ],
    'type_arret_urgence':    [
        "arrêt d'urgence déclenché par opérateur", 'stop urgence automatique sécurité',
        "coupure urgence — surchauffe critique",   "arrêt d'urgence — vibration critique",
    ],
    'type_defaut_qualite':   [
        'défaut qualité pièces constaté',          'non-conformité dimensionnelle',
        'chute taux de production anormale',       'pièces hors tolérance — contrôle qualité',
    ],
}

MAINTENANCE_INFO: dict[str, tuple[str, str]] = {
    'type_surchauffe':       ('capteur température',    'Remplacement capteur thermique + purge circuit'),
    'type_baisse_pression':  ('joint hydraulique',       'Remplacement joint + test étanchéité circuit'),
    'type_vibration':        ('roulement axe principal', 'Remplacement roulement + rééquilibrage rotor'),
    'type_bruit_mecanique':  ('roulement axe principal', 'Inspection + remplacement roulement billes'),
    'type_surconsommation':  ('variateur vitesse',       'Diagnostic variateur + recalibration moteur'),
    'type_blocage_mecanique':('transmission',             'Déblocage + inspection complète transmission'),
    'type_alarme_capteur':   ('capteur pression',        'Remplacement capteur + recalibration zéro'),
    'type_arret_urgence':    ('système sécurité',        'Inspection complète suite arrêt urgence'),
    'type_defaut_qualite':   ('filtre hydraulique',      'Nettoyage + remplacement filtres + contrôle qualité'),
}

OPERATORS = [
    ('Lucas Bernard', 'OP1002'), ('Hugo Thomas', 'OP1004'),
    ('Chloé Robert', 'OP1005'), ('Marie Dupont', 'OP1001'),
    ('Pierre Martin', 'OP1003'), ('Sophie Leclerc', 'OP1006'),
    ('Antoine Moreau', 'OP1007'), ('Julie Petit', 'OP1008'),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ramp(lead_h_series: pd.Series, max_lead_h: float) -> pd.Series:
    """Quadratic factor: 0 at max_lead_h, 1 at 0h before incident."""
    x = (1.0 - lead_h_series.clip(upper=max_lead_h) / max_lead_h).clip(lower=0.0)
    return x ** 2


def _dominant_type(row: pd.Series) -> str:
    for t in TYPE_COLS:
        if row.get(t, 0) == 1:
            return t
    return 'type_alarme_capteur'


def _shift_comment(row: pd.Series) -> str:
    t = _dominant_type(row)
    opts = TYPE_COMMENTS[t]
    return opts[int(RNG.integers(len(opts)))]


# ── Step 0 — Backup ───────────────────────────────────────────────────────────

def backup_originals() -> None:
    BACKUP_DIR.mkdir(exist_ok=True)
    for fname in ('telemetry.csv', 'releves_incidents.csv', 'machine.sql'):
        src, dst = DATA_DIR / fname, BACKUP_DIR / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f'  backup → {dst}')
        elif dst.exists():
            print(f'  already backed up: {dst}')


# ── Step 1 — Load ─────────────────────────────────────────────────────────────

def load_telemetry() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / 'telemetry.csv')
    df['event_ts'] = pd.to_datetime(df['timestamp'])
    df['machine_id_std'] = df['machine_id'].str.strip()
    return df.sort_values(['machine_id_std', 'event_ts']).reset_index(drop=True)


def load_incidents() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / 'releves_incidents.csv')
    df['event_ts'] = pd.to_datetime(
        df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce'
    )
    df['machine_id_std'] = df['machine_id'].str.strip()
    return df.sort_values('event_ts').reset_index(drop=True)


# ── Step 2 — Incident redesign ────────────────────────────────────────────────

def h5_rebalance_severity(inc: pd.DataFrame) -> pd.DataFrame:
    """H5 — Upgrade low-severity incidents on old/critical machines."""
    df = inc.copy()
    df['_w'] = df['machine_id_std'].map(
        lambda m: AGE.get(m, 3) * CRIT_RANK.get(CRITICALITY.get(m, 'LOW'), 1)
    )
    sev1_sorted = df[df['severity'] == 1].sort_values('_w', ascending=False).index

    # sev 1 → sev 5 (top 18)
    n5 = min(18, len(sev1_sorted))
    df.loc[sev1_sorted[:n5], 'severity'] = 5

    # sev 1 → sev 4 (next 72)
    remaining1 = df[df['severity'] == 1].sort_values('_w', ascending=False).index
    n4 = min(72, len(remaining1))
    df.loc[remaining1[:n4], 'severity'] = 4

    # sev 1 → sev 3 (next 130)
    remaining1b = df[df['severity'] == 1].sort_values('_w', ascending=False).index
    n3 = min(130, len(remaining1b))
    df.loc[remaining1b[:n3], 'severity'] = 3

    # sev 2 → sev 3 (top 60 by weight)
    sev2_sorted = df[df['severity'] == 2].sort_values('_w', ascending=False).index
    n32 = min(60, len(sev2_sorted))
    df.loc[sev2_sorted[:n32], 'severity'] = 3

    return df.drop(columns=['_w'])


def h8_redistribute_by_age(inc: pd.DataFrame) -> pd.DataFrame:
    """H8 — Keep high-sev incidents; thin low-sev on young machines."""
    df = inc.copy()
    machines = list(COMMISSIONED.keys())
    weights = {m: AGE[m] * CRIT_RANK[CRITICALITY[m]] for m in machines}
    total_w = sum(weights.values())
    target  = {m: max(25, round(len(df) * weights[m] / total_w)) for m in machines}

    parts = []
    for m in machines:
        sub = df[df['machine_id_std'] == m].copy()
        tgt = target[m]
        if len(sub) <= tgt:
            parts.append(sub)
        else:
            high = sub[sub['severity'] >= 3]
            low  = sub[sub['severity'] < 3]
            keep_low = max(0, tgt - len(high))
            if keep_low < len(low):
                low = low.sample(keep_low, random_state=SEED)
            parts.append(pd.concat([high, low]))

    return pd.concat(parts).sort_values('event_ts').reset_index(drop=True)


def _new_incident(machine: str, ts: pd.Timestamp, severity: int,
                   inc_type: str, iid: int) -> dict:
    op, badge = OPERATORS[int(RNG.integers(len(OPERATORS)))]
    h = ts.hour
    shift = 'matin' if 6 <= h < 14 else ('apres-midi' if 14 <= h < 22 else 'nuit')
    opts = TYPE_COMMENTS.get(inc_type, ['incident détecté'])
    row: dict = {
        'incident_id':    f'INC-{iid:06d}',
        'date':           ts.strftime('%Y-%m-%d'),
        'time':           ts.strftime('%H:%M'),
        'operator_name':  op,
        'machine_id':     machine,
        'severity':       severity,
        'operator_badge': badge,
        'comment':        opts[int(RNG.integers(len(opts)))],
        'shift':          shift,
        'machine_id_std': machine,
        'event_ts':       ts,
    }
    for tc in TYPE_COLS:
        row[tc] = 1 if tc == inc_type else 0
    return row


def h6_add_clusters(inc: pd.DataFrame) -> pd.DataFrame:
    """H6 — Add 2 precursor incidents 48h and 24h before each sev 4-5 event."""
    sev45 = inc[inc['severity'] >= 4]
    new_rows: list[dict] = []
    next_id = 90001
    tel_start = inc['event_ts'].min()

    for _, row in sev45.iterrows():
        machine  = row['machine_id_std']
        ts       = row['event_ts']
        inc_type = _dominant_type(row)

        for delta_h, pre_sev in [(48, 2), (24, 3)]:
            pre_ts = ts - pd.Timedelta(hours=delta_h)
            if pre_ts >= tel_start:
                new_rows.append(_new_incident(machine, pre_ts, pre_sev, inc_type, next_id))
                next_id += 1

    if new_rows:
        inc = pd.concat([inc, pd.DataFrame(new_rows)], ignore_index=True)

    return inc.sort_values('event_ts').reset_index(drop=True)


def h7_fix_timestamps_comments(inc: pd.DataFrame) -> pd.DataFrame:
    """H7 — Realistic minutes + coherent comments for empty/generic ones."""
    df = inc.copy()

    # Randomise minutes that are exactly :00
    round_mask = df['time'].str.endswith(':00')
    n = round_mask.sum()
    if n > 0:
        df.loc[round_mask, 'event_ts'] += pd.to_timedelta(
            RNG.integers(1, 58, size=n), unit='min'
        )
    df['time'] = df['event_ts'].dt.strftime('%H:%M')
    df['date'] = df['event_ts'].dt.strftime('%Y-%m-%d')

    # Fix empty / generic comments
    empty = df['comment'].isna() | (df['comment'].astype(str).str.strip() == '')
    df.loc[empty, 'comment'] = df[empty].apply(_shift_comment, axis=1)

    return df


def renumber_ids(inc: pd.DataFrame) -> pd.DataFrame:
    df = inc.sort_values('event_ts').reset_index(drop=True)
    df['incident_id'] = [f'INC-{i+1:06d}' for i in range(len(df))]
    return df


# ── Step 3 — Telemetry transformation ────────────────────────────────────────

def h2_machine_baselines(tel: pd.DataFrame) -> pd.DataFrame:
    """H2 — Shift per-machine baselines to create distinct machine fingerprints."""
    df = tel.copy()

    for m in df['machine_id_std'].unique():
        mask = df['machine_id_std'] == m
        age  = AGE.get(m, 3)
        crit = CRIT_RANK.get(CRITICALITY.get(m, 'LOW'), 1)

        # Temperature: older + higher criticality machines run hotter
        tgt_temp = 43.0 + age * 0.85 + (crit - 1) * 2.2
        df.loc[mask, 'temperature_c'] += tgt_temp - df.loc[mask, 'temperature_c'].mean()

        # Pressure: gentle age-related drop (wear on seals)
        tgt_pres = 200.5 - age * 0.18
        df.loc[mask, 'pressure_bar'] += tgt_pres - df.loc[mask, 'pressure_bar'].mean()

        # Rotation: HIGH machines run faster
        tgt_rot = 1565.0 + (crit - 1) * 28.0
        df.loc[mask, 'rotation_mean_rpm'] += tgt_rot - df.loc[mask, 'rotation_mean_rpm'].mean()

    for col, (lo, hi) in SIGNAL_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    return df


def h1_degradation_ramps(tel: pd.DataFrame, inc: pd.DataFrame) -> pd.DataFrame:
    """H1 — Vectorised pre-failure signal drift for every sev >= 3 incident."""
    df = tel.copy()
    # pieces_produced is int64 — upcast to float for drift arithmetic
    df['pieces_produced'] = df['pieces_produced'].astype(float)
    sev3p = inc[inc['severity'] >= 3].copy()
    total = len(sev3p)

    for i, (_, irow) in enumerate(sev3p.iterrows(), 1):
        if i % 50 == 0 or i == total:
            print(f'    H1 ramps {i}/{total}…', end='\r')

        machine  = irow['machine_id_std']
        inc_ts   = irow['event_ts']
        scale    = SEV_SCALE.get(int(irow['severity']), 0.6)

        # Collect all effects for active incident types
        effects: list = []
        for tc, eff_list in TYPE_EFFECTS.items():
            if irow.get(tc, 0) == 1:
                effects.extend(eff_list)
        if not effects:
            effects = [('temperature_c', +3.0, 12)]

        for signal, drift_or_mode, lead_h in effects:
            window_start = inc_ts - pd.Timedelta(hours=lead_h)
            mask = (
                (df['machine_id_std'] == machine) &
                (df['event_ts'] >= window_start) &
                (df['event_ts'] < inc_ts)
            )
            sub = df.loc[mask, ['event_ts', signal]].copy()
            if sub.empty:
                continue

            lead_h_series = (inc_ts - sub['event_ts']).dt.total_seconds() / 3600
            factors       = _ramp(lead_h_series, lead_h)

            if drift_or_mode == 'noise':
                m_mean    = df.loc[df['machine_id_std'] == machine, signal].mean()
                noise_amp = 1.0 + scale * factors * 3.0
                df.loc[mask, signal] = m_mean + (sub[signal].values - m_mean) * noise_amp.values

            elif drift_or_mode == 'spikes':
                spike_hit = RNG.random(size=len(sub)) < (factors * 0.35 * scale).values
                magnitudes = (
                    RNG.choice([-1, 1], size=len(sub)) *
                    RNG.uniform(80, 240, size=len(sub)) * scale
                )
                df.loc[mask, signal] = sub[signal].values + spike_hit * magnitudes

            else:
                df.loc[mask, signal] = sub[signal].values + drift_or_mode * scale * factors.values

    print()  # newline after progress
    for col, (lo, hi) in SIGNAL_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi).round(3)

    # pieces_produced stays integer
    df['pieces_produced'] = df['pieces_produced'].round().clip(0, 120).astype(int)

    return df


def h3_missing_values(tel: pd.DataFrame) -> pd.DataFrame:
    """H3 — Sensor failure NaN blocs (4-8h, 2-3 events per machine)."""
    df = tel.copy()
    sensors = ['temperature_c', 'pressure_bar', 'rotation_mean_rpm']

    for m in df['machine_id_std'].unique():
        m_idx = df.index[df['machine_id_std'] == m].tolist()
        if len(m_idx) < 400:
            continue
        # 18-25 events per machine → ~2 % of rows carry at least one NaN
        n_events    = int(RNG.integers(18, 26))
        sel_sensors = RNG.choice(sensors, size=n_events, replace=True)

        for sensor in sel_sensors:
            start = int(RNG.integers(100, len(m_idx) - 100))
            dur   = int(RNG.integers(4, 13))   # 4-12 h per failure event
            df.loc[m_idx[start: start + dur], sensor] = np.nan

    return df


def h4_duplicates(tel: pd.DataFrame) -> pd.DataFrame:
    """H4 — 0.5 % duplicate rows with micro-noise (double-transmission sim)."""
    n_dup  = max(1, round(len(tel) * 0.005))
    dup_ix = RNG.choice(tel.index, size=n_dup, replace=False)
    dups   = tel.loc[dup_ix].copy()

    for col in ['temperature_c', 'pressure_bar', 'voltage_mean_v', 'rotation_mean_rpm']:
        dups[col] = (dups[col] + RNG.normal(0, 0.02, size=len(dups))).round(3)

    return (
        pd.concat([tel, dups], ignore_index=True)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )


# ── Step 4 — SQL update ───────────────────────────────────────────────────────

def h9_reactive_maintenance(sql_text: str, inc: pd.DataFrame) -> str:
    """H9 — Append reactive maintenance records after every sev >= 3 incident."""
    existing_ids = re.findall(r'\((\d+),\s*\'MACH-', sql_text)
    next_id      = (max(int(i) for i in existing_ids) + 1) if existing_ids else 200

    sev3p = inc[inc['severity'] >= 3].copy()
    new_vals: list[str] = []

    for _, row in sev3p.iterrows():
        machine  = row['machine_id_std']
        inc_ts   = row['event_ts']
        severity = int(row['severity'])
        inc_id   = row.get('incident_id', '')

        delay_h  = int(RNG.integers(10, 38))
        maint_ts = (inc_ts + pd.Timedelta(hours=delay_h)).strftime('%Y-%m-%d %H:%M:%S+00')
        duration = round(float(RNG.uniform(1.5, 3.0) * (severity / 3.0)), 2)

        component, desc = MAINTENANCE_INFO.get(_dominant_type(row), ('composant', 'Intervention corrective'))
        rel_inc = f"'{inc_id}'" if inc_id and str(inc_id) != 'nan' else 'NULL'

        new_vals.append(
            f"({next_id}, '{machine}', '{maint_ts}', 'reactive', "
            f"'intervention_corrective', '{component}', '{desc}', "
            f"{rel_inc}, {duration})"
        )
        next_id += 1

    if not new_vals:
        return sql_text

    # Insert new VALUES rows before the ON CONFLICT clause
    pattern = r"(INSERT INTO maintenance\b.*?VALUES\s*)((?:\([^)]+\),?\s*)+?)(\s*ON CONFLICT)"
    def replacer(m: re.Match) -> str:
        existing = m.group(2).rstrip().rstrip(',')
        added    = ',\n'.join(new_vals)
        return f"{m.group(1)}{existing},\n{added}{m.group(3)}"

    return re.sub(pattern, replacer, sql_text, flags=re.DOTALL)


# ── Step 5 — Save ─────────────────────────────────────────────────────────────

def save_outputs(tel: pd.DataFrame, inc: pd.DataFrame, sql_text: str) -> None:
    TEL_COLS = ['machine_id', 'timestamp', 'temperature_c', 'pressure_bar',
                'voltage_mean_v', 'rotation_mean_rpm', 'pieces_produced']
    INC_COLS = [
        'incident_id', 'date', 'time', 'operator_name', 'machine_id',
        'severity', 'operator_badge', 'comment', 'shift',
    ] + TYPE_COLS

    tel_out = (
        tel[TEL_COLS]
        .sort_values(['machine_id', 'timestamp'])
        .reset_index(drop=True)
    )
    tel_out.to_csv(DATA_DIR / 'telemetry.csv', index=False)
    print(f'  telemetry.csv         → {len(tel_out):,} rows')

    inc_out = inc[[c for c in INC_COLS if c in inc.columns]].reset_index(drop=True)
    inc_out.to_csv(DATA_DIR / 'releves_incidents.csv', index=False)
    print(f'  releves_incidents.csv → {len(inc_out):,} rows')

    (DATA_DIR / 'machine.sql').write_text(sql_text, encoding='utf-8')
    print(f'  machine.sql           → updated')


# ── Step 6 — Validation ───────────────────────────────────────────────────────

def validate(tel: pd.DataFrame, inc: pd.DataFrame) -> None:
    sep = '─' * 55

    print(f'\n{sep}')
    print('VALIDATION')
    print(sep)

    # Severity distribution
    sv = inc['severity'].value_counts().sort_index()
    print(f'\nSévérité distribution (total {len(inc):,}):')
    for k, v in sv.items():
        bar = '█' * (v // 10)
        print(f'  sév {k}: {v:4d}  {bar}')

    # Machine age vs incident count
    print('\nIncidents par machine (top 8):')
    counts = inc.groupby('machine_id_std').size().rename('n')
    ages_s = pd.Series(AGE, name='age')
    crit_s = pd.Series({m: CRIT_RANK[c] for m, c in CRITICALITY.items()}, name='crit')
    tbl = pd.concat([counts, ages_s, crit_s], axis=1).sort_values('n', ascending=False)
    print(tbl.head(8).to_string())

    # Pearson r: age × incident_count
    corr = tbl['n'].corr(tbl['age'])
    print(f'\n  Corrélation incidents ↔ âge machine : r = {corr:.3f}')

    # Per-machine temperature baseline spread
    means = tel.groupby('machine_id_std')['temperature_c'].mean().round(2)
    print(f'\nTemp baseline par machine — min={means.min():.1f}°C  max={means.max():.1f}°C  '
          f'spread={means.max() - means.min():.1f}°C')
    print(means.sort_values().to_string())

    # Pre-failure signal check: mean temperature in 6h before sev>=3 surchauffe incidents
    tel_ts = tel.copy()
    if 'event_ts' not in tel_ts.columns:
        tel_ts['event_ts'] = pd.to_datetime(tel_ts['timestamp'])
        tel_ts['machine_id_std'] = tel_ts['machine_id'].str.strip()

    surchauffe = inc[(inc['type_surchauffe'] == 1) & (inc['severity'] >= 3)]
    deltas = []
    for _, row in surchauffe.head(40).iterrows():
        m, ts = row['machine_id_std'], row['event_ts']
        before = tel_ts[
            (tel_ts['machine_id_std'] == m) &
            (tel_ts['event_ts'] >= ts - pd.Timedelta(hours=6)) &
            (tel_ts['event_ts'] < ts)
        ]['temperature_c']
        baseline = tel_ts[tel_ts['machine_id_std'] == m]['temperature_c'].mean()
        if len(before) > 0:
            deltas.append(before.mean() - baseline)
    if deltas:
        print(f'\nDelta temp moyen (6h avant surchauffe sév≥3 vs baseline machine): '
              f'+{np.mean(deltas):.2f}°C  '
              f'(attendu >> 0, signal pédagogique H1)')

    # Missing values
    n_nan = tel[['temperature_c', 'pressure_bar', 'rotation_mean_rpm']].isna().sum().sum()
    pct   = n_nan / (len(tel) * 3) * 100
    print(f'\nValeurs manquantes : {n_nan:,}  ({pct:.2f}%)  [cible ~2-3%]')

    # Duplicates
    n_dup = tel.duplicated(subset=['machine_id', 'timestamp']).sum()
    pct_d = n_dup / len(tel) * 100
    print(f'Doublons (même machine+timestamp) : {n_dup:,}  ({pct_d:.2f}%)  [cible ~0.5%]')

    print(f'\n{sep}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print('InduSense — Dataset Regeneration H1→H9')
    print('=' * 55)

    print('\n[0] Backup des originaux…')
    backup_originals()

    print('\n[1] Chargement…')
    tel = load_telemetry()
    inc = load_incidents()
    sql = (DATA_DIR / 'machine.sql').read_text(encoding='utf-8')
    print(f'  télémétrie : {len(tel):,} lignes')
    print(f'  incidents  : {len(inc):,} lignes')
    print(f'  sév dist   : {inc["severity"].value_counts().sort_index().to_dict()}')

    print('\n[2] Redesign incidents…')
    inc = h5_rebalance_severity(inc)
    print(f'  H5 sév dist : {inc["severity"].value_counts().sort_index().to_dict()}')
    inc = h8_redistribute_by_age(inc)
    print(f'  H8 total    : {len(inc):,}')
    inc = h6_add_clusters(inc)
    print(f'  H6 total    : {len(inc):,} (+clusters)')
    inc = h7_fix_timestamps_comments(inc)
    print(f'  H7 timestamps & commentaires OK')
    inc = renumber_ids(inc)

    print('\n[3] Transformation télémétrie…')
    tel = h2_machine_baselines(tel)
    print(f'  H2 baselines ajustées')
    tel = h1_degradation_ramps(tel, inc)
    print(f'  H1 rampes appliquées')
    tel = h3_missing_values(tel)
    n_nan = tel[['temperature_c', 'pressure_bar', 'rotation_mean_rpm']].isna().sum().sum()
    print(f'  H3 {n_nan:,} NaN injectés')
    tel = h4_duplicates(tel)
    print(f'  H4 doublons → {len(tel):,} lignes total')

    print('\n[4] Mise à jour SQL (H9)…')
    sql = h9_reactive_maintenance(sql, inc)

    print('\n[5] Sauvegarde…')
    save_outputs(tel, inc, sql)

    validate(tel, inc)
    print('Terminé.')


if __name__ == '__main__':
    main()
