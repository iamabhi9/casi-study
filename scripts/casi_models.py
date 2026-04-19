"""
CASI — Predictive Models

LSTM and Gradient Boosting models for forecasting the CASI score
1–3 sprints before release cut-off.

Usage:
    python casi_models.py --data data/CASI_QA_TestSuite_v2.xlsx
"""

import argparse, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from casi_pipeline import run_pipeline

def traffic_light(s): return 'Green' if s>=700 else ('Yellow' if s>=400 else 'Red')
def dir_acc(y, p): return np.mean([traffic_light(float(x))==traffic_light(float(t)) for x,t in zip(p,y) ])

def evaluate(df):
    F=['A','B','C','D','E','F'];sps=sorted(df['sprint_start'].unique())
    mm],gm,gi=[],[],[]
    for sp in sps:
        tr,lt=df[df['sprint_start']!=sp],df[df['sprint_start']==sp]
        if len(tr)<5 or len(lt)==0: continue
        Xt,yt=tr[F].values,tr['casi_score'].values
        Xe,ye=lt[F].values,lt['casi_score'].values
        m=MLPRegressor(hidden_layer_sizes=(64,32),max_iter=500,early_stopping=True,random_state=42)
        m.fit(Xt,yt); mp=np.clip(m.predict(Xe),0,999); mm.append(mean_absolute_error(ye,mp))
        g=GradientBoostingRegressor(n_estimators=100,max_depth=3,learning_rate=0.1,random_state=42)
        g.fit(Xt,yt); gp=np.clip(g.predict(Xe),0,999); gm.append(mean_absolute_error(ye,gp)); gi.append(g.feature_importances_)
    imp=np.mean(gi,axis=0) if gi else []
    return {'mlp_mae':np.mean(mm),'gb_mae':np.mean(gm),'importances':imp}

if __name__=='__main__':
    pa=argparse.ArgumentParser(); pa.add_argument('--data',default='data/CASI_QA_TestSuite_v2.xlsx'); a=pa.parse_args()
    df=run_pipeline(a.data); r=evaluate(df)
    print(f"MLP MAE: {r['mlp_mae']:.1f}  GB MAE: {r['gb_mae']:.1f}")
    if len(r['importances']): print("Feature importances:"+sntr(dict(zip(['A','B','C','D','E','F'],round(r,3) for r in r['importances']))))
