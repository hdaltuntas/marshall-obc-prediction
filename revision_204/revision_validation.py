"""Reproducible, retrospective validation of the supplied Marshall dataset.

Run with Python 3.11+, numpy, pandas, scipy, scikit-learn 1.8.0, openpyxl.
All model inputs are explicitly pre-test variables. Source/date identifiers
are used only to define validation splits and output provenance.
"""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn, scipy
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
    LeaveOneGroupOut, TimeSeriesSplit, RandomizedSearchCV)

SEED=42
GRAD=[f'passing_{s}mm' for s in ['25_4','19_1','12_7','9_5','4_75','2_0','0_425','0_177','0_075']]
BINDER=['bitumen_specific_gravity','penetration','softening_point']
QUALITY=['los_angeles_loss','mgso4_loss','flakiness_index']
FEATURES=['layer_type',*GRAD,*BINDER,*QUALITY]
TARGET='optimum_bitumen_ratio'
FIXED=dict(n_estimators=150,max_depth=8,min_samples_leaf=2,max_features='sqrt')
GRID={'model__n_estimators':[150,300,450], 'model__max_depth':[None,8,12],
      'model__min_samples_leaf':[1,2,3], 'model__max_features':['sqrt',.6,1.0]}

def pipeline(cols, params=None):
    nums=[c for c in cols if c!='layer_type']
    prep=ColumnTransformer([('num',StandardScaler(),nums),
        ('cat',OneHotEncoder(drop='first',handle_unknown='ignore'),['layer_type'])])
    return Pipeline([('prep',prep),('model',ExtraTreesRegressor(
        random_state=SEED,n_jobs=1,**(params or {})))])

def metrics(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float);e=y-p
    sst=float(np.sum((y-y.mean())**2));sse=float(np.sum(e**2))
    return {'n':int(len(y)),'r2':1-sse/sst if len(y)>1 and sst>1e-14 else None,
        'mae':float(np.mean(np.abs(e))),'rmse':float(np.sqrt(np.mean(e**2))),
        'mean_error':float(np.mean(e)),
        'within_010':float(np.mean(np.abs(e)<=.10+1e-12)),
        'within_025':float(np.mean(np.abs(e)<=.25+1e-12)),
        'within_050':float(np.mean(np.abs(e)<=.50+1e-12))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--jobs',type=int,default=4);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    source=Path(args.input)
    raw=pd.read_excel(source,sheet_name='Sheet1')
    meta=pd.read_excel(source,sheet_name='Metadata')
    assert raw.file_name.is_unique and meta.file_name.is_unique
    assert set(raw.file_name)==set(meta.file_name)
    data=raw.merge(meta,on='file_name',validate='one_to_one',sort=False)
    data['design_date']=pd.to_datetime(data.design_date,errors='raise')
    data['quarry_id_raw']=data.quarry_id
    data['quarry_id']=data.quarry_id.str.strip()
    assert data[FEATURES+[TARGET,'quarry_id','design_date']].notna().all().all()
    assert not raw.duplicated([c for c in raw if c!='file_name']).any()
    assert np.all(np.diff(raw[['passing_37_5mm',*GRAD]],axis=1)<=1e-10)
    assert ((raw[GRAD]>=0)&(raw[GRAD]<=100)).all().all()
    assert raw.passing_37_5mm.eq(100).all()
    X=data[FEATURES];y=data[TARGET];layers=data.layer_type;groups=data.quarry_id
    config={
      'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'n':len(data),
      'n_quarries':groups.nunique(),'layer_counts':layers.value_counts().to_dict(),
      'random_state':SEED,'features':FEATURES,'target':TARGET,'target_unit':'percentage points by mass of dry aggregate; manuscript convention',
      'algorithm':'ExtraTreesRegressor; family retained from original manuscript',
      'outer_folds':5,'inner_folds':4,'search_iterations':25,'search_scoring':'neg_mean_absolute_error',
      'search_grid':GRID,'fixed_sensitivity_parameters':FIXED,
      'temporal_cutoff':'2022-01-01','temporal_inner':'4 forward splits of distinct training years; equal years kept together',
      'notes':[
       'Grouped outer and inner splits use quarry identifiers and layer stratification.',
       'Headline CV metrics pool the out-of-fold predictions; they are not means of fold R2.',
       'Layer means are estimated only from the corresponding training partition.',
       'No fold-independence confidence intervals or p-values are calculated.',
       'Ablation and leave-one-quarry-out analyses reuse previously reported hyperparameters and are sensitivity analyses.',
       'This is retrospective internal validation, not independent-laboratory external validation.',
       'The temporal cutoff was chosen from calendar-year counts before calculating this run results.',
       'No new all-algorithm model-family selection is performed.',
       'Only trailing/leading quarry whitespace is normalized; no material measurements are edited.',
       'Project, batch and laboratory fields are unfilled; quarry grouping does not prove project independence.'
      ],'versions':{'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__}}
    (out/'protocol.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    raw.to_csv(out/'modeling_data_as_received.csv',index=False)
    meta.to_csv(out/'metadata_as_received.csv',index=False,date_format='%Y-%m-%d')
    data.to_csv(out/'analysis_data.csv',index=False,date_format='%Y-%m-%d')
    preds=[];folds=[];summary=[];trials=[];start=time.time()

    def save_predictions(protocol,fold,train,test,p,parameters):
        means=data.iloc[train].groupby('layer_type')[TARGET].mean()
        baseline=data.iloc[test].layer_type.map(means).to_numpy()
        assert np.isfinite(baseline).all()
        for j,idx in enumerate(test):
            row=data.iloc[idx]
            preds.append({'protocol':protocol,'fold':int(fold),'file_name':row.file_name,
             'layer_type':row.layer_type,'quarry_id':row.quarry_id,'design_date':row.design_date.strftime('%Y-%m-%d'),
             'observed_obc':float(y.iloc[idx]),'predicted_obc':float(p[j]),'baseline_obc':float(baseline[j]),
             'error':float(y.iloc[idx]-p[j]),'absolute_error':float(abs(y.iloc[idx]-p[j])),
             'baseline_absolute_error':float(abs(y.iloc[idx]-baseline[j])),
             'quarry_seen_in_training':bool(row.quarry_id in set(groups.iloc[train]))})
        folds.append({'protocol':protocol,'fold':int(fold),'n_train':len(train),'n_test':len(test),
         'train_quarries':int(groups.iloc[train].nunique()),'test_quarries':int(groups.iloc[test].nunique()),
         'quarry_overlap':len(set(groups.iloc[train])&set(groups.iloc[test])),
         'train_last_date':data.iloc[train].design_date.max().strftime('%Y-%m-%d'),
         'test_first_date':data.iloc[test].design_date.min().strftime('%Y-%m-%d'),
         **metrics(y.iloc[test],p),'baseline_mae':metrics(y.iloc[test],baseline)['mae'],
         'best_parameters':json.dumps(parameters,ensure_ascii=False,sort_keys=True)})

    group_outer=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(X,layers,groups))
    random_outer=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED).split(X,layers))
    for protocol,outer in [('quarry_nested',group_outer),('record_nested',random_outer)]:
        for f,(tr,te) in enumerate(outer,1):
            if protocol=='quarry_nested':
                assert not set(groups.iloc[tr])&set(groups.iloc[te])
                inner=list(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=SEED).split(X.iloc[tr],layers.iloc[tr],groups.iloc[tr]))
                for itr,ite in inner:assert not set(groups.iloc[tr].iloc[itr])&set(groups.iloc[tr].iloc[ite])
            else:
                inner=list(StratifiedKFold(n_splits=4,shuffle=True,random_state=SEED).split(X.iloc[tr],layers.iloc[tr]))
            search=RandomizedSearchCV(pipeline(FEATURES),GRID,n_iter=25,scoring='neg_mean_absolute_error',cv=inner,
               n_jobs=args.jobs,refit=True,random_state=SEED,error_score='raise',return_train_score=False)
            search.fit(X.iloc[tr],y.iloc[tr]);p=search.predict(X.iloc[te])
            save_predictions(protocol,f,tr,te,p,search.best_params_)
            for i,params in enumerate(search.cv_results_['params']):
                trials.append({'protocol':protocol,'outer_fold':f,'inner_mean_mae':float(-search.cv_results_['mean_test_score'][i]),
                  'parameters':json.dumps(params,sort_keys=True)})
            pd.DataFrame(preds).to_csv(out/'predictions_checkpoint.csv',index=False)
            print(json.dumps({'done':protocol,'fold':f,'elapsed_s':round(time.time()-start,1)},ensure_ascii=False),flush=True)

    tr=np.flatnonzero(data.design_date<pd.Timestamp('2022-01-01'))
    te=np.flatnonzero(data.design_date>=pd.Timestamp('2022-01-01'))
    yrs=data.iloc[tr].design_date.dt.year.to_numpy();unique_years=np.sort(np.unique(yrs));inner=[]
    for a,b in TimeSeriesSplit(n_splits=4).split(unique_years):
        itr=np.flatnonzero(np.isin(yrs,unique_years[a]));ite=np.flatnonzero(np.isin(yrs,unique_years[b]))
        assert yrs[itr].max()<yrs[ite].min();inner.append((itr,ite))
    search=RandomizedSearchCV(pipeline(FEATURES),GRID,n_iter=25,scoring='neg_mean_absolute_error',cv=inner,
       n_jobs=args.jobs,refit=True,random_state=SEED,error_score='raise')
    search.fit(X.iloc[tr],y.iloc[tr]);p=search.predict(X.iloc[te])
    assert data.iloc[tr].design_date.max()<data.iloc[te].design_date.min()
    save_predictions('temporal_2022',1,tr,te,p,search.best_params_)
    for i,params in enumerate(search.cv_results_['params']):
        trials.append({'protocol':'temporal_2022','outer_fold':1,'inner_mean_mae':float(-search.cv_results_['mean_test_score'][i]),'parameters':json.dumps(params,sort_keys=True)})
    print('Temporal validation complete',flush=True)

    for variant,cols in [('layer_gradation',['layer_type',*GRAD]),('layer_gradation_binder',['layer_type',*GRAD,*BINDER]),('full',FEATURES)]:
        for f,(tr,te) in enumerate(group_outer,1):
            p=pipeline(cols,FIXED).fit(X.iloc[tr][cols],y.iloc[tr]).predict(X.iloc[te][cols])
            save_predictions('ablation_'+variant,f,tr,te,p,FIXED)
    print('Fixed-parameter grouped ablations complete',flush=True)
    for f,(tr,te) in enumerate(LeaveOneGroupOut().split(X,y,groups),1):
        assert not set(groups.iloc[tr])&set(groups.iloc[te])
        p=pipeline(FEATURES,FIXED).fit(X.iloc[tr],y.iloc[tr]).predict(X.iloc[te])
        save_predictions('leave_one_quarry_fixed',f,tr,te,p,FIXED)
    print('Leave-one-quarry-out sensitivity complete',flush=True)

    pr=pd.DataFrame(preds);fr=pd.DataFrame(folds)
    for prot,g in pr.groupby('protocol',sort=False):
        assert not g.file_name.duplicated().any()
        if prot!='temporal_2022':assert len(g)==len(data)
        for layer in ['all',*sorted(layers.unique())]:
            z=g if layer=='all' else g[g.layer_type==layer]
            for model,col in [('Extra Trees','predicted_obc'),('Layer mean','baseline_obc')]:
                summary.append({'protocol':prot,'layer':layer,'model':model,**metrics(z.observed_obc,z[col])})
    quarryrows=[]
    for (prot,quarry),g in pr.groupby(['protocol','quarry_id'],sort=True):
        e=metrics(g.observed_obc,g.predicted_obc);b=metrics(g.observed_obc,g.baseline_obc)
        quarryrows.append({'protocol':prot,'quarry_id':quarry,**e,'baseline_mae':b['mae'],
            'baseline_rmse':b['rmse'],'mae_difference_et_minus_baseline':e['mae']-b['mae']})
    numeric=GRAD+BINDER+QUALITY;correlations=[]
    centered_y=y-data.groupby('layer_type')[TARGET].transform('mean')
    for c in numeric:
        centered_x=data[c]-data.groupby('layer_type')[c].transform('mean')
        item={'feature':c,'pooled_r':data[c].corr(y),'layer_adjusted_r':centered_x.corr(centered_y)}
        for layer,g in data.groupby('layer_type'):
            item[layer]=g[c].corr(g[TARGET]) if g[c].nunique()>1 else np.nan
        correlations.append(item)
    pd.DataFrame(summary).to_csv(out/'metrics.csv',index=False)
    pr.to_csv(out/'predictions.csv',index=False);fr.to_csv(out/'folds.csv',index=False)
    pd.DataFrame(quarryrows).to_csv(out/'quarry_metrics.csv',index=False)
    pd.DataFrame(correlations).to_csv(out/'correlations.csv',index=False)
    pd.DataFrame(trials).to_csv(out/'hyperparameter_search.csv',index=False)
    # Date/count tables support audit of the predefined temporal cutoff.
    pd.crosstab(data.design_date.dt.year,layers).to_csv(out/'year_layer_counts.csv')
    qq=pd.crosstab(groups,layers);qq['total']=qq.sum(axis=1);qq.to_csv(out/'quarry_layer_counts.csv')
    print('FINAL_SUMMARY',flush=True)
    print(pd.DataFrame(summary).loc[lambda z:(z.layer=='all')&z.protocol.isin(['quarry_nested','record_nested','temporal_2022'])].to_string(index=False),flush=True)
    print('DONE',round(time.time()-start,1),'seconds',flush=True)

if __name__=='__main__':main()
