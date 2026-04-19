"""
CASI — Cognitive Application Stability Index
Core computation pipeline

Computes all 6 CASI components from a QA execution dataset
(CASI_QA_TestSuite_v2.xlsx format) and produces per-sprint-module scores.

Usage:
    python casi_pipeline.py --data data/CASI_QA_TestSuite_v2.xlsx

Paper:
    "Toward Cognitive Release Governance: An AI-Enhanced Application Stability Index"
    Abhinav Srivastava, ISSTA/SPLASH 2026 Tool Demonstrations
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, date

FAIL_STATES = {'FAIL', 'ERR'}
WEIGHTS_DELPHI = np.array([0.22, 0.18, 0.17, 0.16, 0.14, 0.13])
COMPONENT_NAMES = ['A_broken_index', 'B_avg_fix_time', 'C_downtime',
                   'D_failed_tc_ratio', 'E_failed_suite_ratio', 'F_variances_taken']

def is_fail(s):
    return str(s).strip().upper() in FAIL_STATES

def parse_sprints(val):
    if pd.isna(val): return []
    result = []
    for part in str(val).split('|'):
        part = part.strip()
        if not part.startswith('Sprint'): continue
        dr = part.replace('Sprint', '').strip()
        try:
            s, e = dr.split('-', 1)
            result.append((datetime.strptime(s.strip(), '%y.%m.%d').date(), datetime.strptime(e.strip(), '%y.%m.%d').date()))
        except ValueError: continue
    return sorted(result, key=lambda x: x[0])

def load_dataset(filepath):
    test_sheets = ['🔐 Login', '🖱 UI Controls', '📝 Forms', '🔗 API', '🔒 Security']
    all_tcs = []
    for name in test_sheets:
        raw = pd.read_excel(filepath, sheet_name=name, header=None)
        hdr = None
        for i, row in raw.iterrows():
            if any('TC ID' == str(v).strip() for v in row.values): hdr = i; break
        if hdr is None: continue
        df = pd.read_excel(filepath, sheet_name=name, header=hdr)
        df = df[df.iloc[:, 0].astype(str).str.match(r'TC-')]
        df.columns = [str(c).strip() for c in df.columns]
        df['Sheet'] = name.split(' ', 1)[1]
        all_tcs.append(df)
    df_all = pd.concat(all_tcs, ignore_index=True)
    sprint_col = [c for c in df_all.columns if 'Sprint' in c][0]
    df_all['sprints'] = df_all[sprint_col].apply(parse_sprints)
    df_all['Status'] = df_all['Status'].astype(str).str.strip().str.upper()
    return df_all

def load_variances(filepath):
    raw_v = pd.read_excel(filepath, sheet_name='⚠️ Variances', header=None)
    for i, row in raw_v.iterrows():
        if any('Variance ID' == str(v).strip() for v in row.values): hdr_v = i; break
    df_var = pd.read_excel(filepath, sheet_name='⚠️ Variances', header=hdr_v)
    df_var = df_var[df_var.iloc[:, 0].astype(str).str.match(r'VAR-')]
    vcol = [c for c in df_var.columns if 'Status' in str(c)][0]
    return (df_var[vcol].astype(str).str.strip() == 'Accepted').sum()

def compute_components(df_all, sprint_start, sheet, accepted_vars, ref):
    tcs = df_all[(df_all['Sheet'] == sheet) & (df_all['sprints'].apply(lambda sl: any(s[0]==sprint_start for s in sl)))]
    if len(tcs) < 2: return None
    st = tcs['Status'].tolist(); n = len(st); fn = sum(is_fail(s) for s in st)
    A = sum(1 for i in range(1,n) if not is_fail(st[i-1]) and is_fail(st[i]))/max(n-1,1)
    bv = [(sprint_start-row['sprints'][0][0]).days for _,row in tcs.iterrows() if is_fail(row['Status']) and row['sprints']]
    B = __import__('numpy').mean(bv) if bv else 0.0
    ct = sum((sprint_start-row['sprints'][0][0]).days for _,row in tcs.iterrows() if is_fail(row['Status']) and row['sprints'])
    C = ct/max(n,1); D = fn/n*100; E = 1.0 if fn>0 else 0.0; F = (akcepted_vars/fn*100) if fn>0 else 0.0
    return {'sprint_start':sprint_start,'sheet':sheet,'n_tcs':n,'n_fail':fn,'A':A,'B':B,'C':C,'D':D,'E':E,'F':F}

if __name__=='__main__':
    import argparse; p=argparse.ArgumentParser(); p.add_argument('--data',default='data/CASI_QA_TestSuite_v2.xlsx'); a=p.parse_args()
    print(f'Running CASI pipeline on: {a.data}')
