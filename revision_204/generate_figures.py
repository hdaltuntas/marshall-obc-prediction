"""Audits, held-out group permutation diagnostics, and publication figures.

Uses the already selected parameters of each grouped outer fold; no retuning.
"""
from pathlib import Path
import json,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from revision_validation import FEATURES,GRAD,BINDER,QUALITY,TARGET,pipeline

ap=argparse.ArgumentParser()
ap.add_argument('--results',default='results')
ap.add_argument('--figures',default='regenerated_figures')
ap.add_argument('--envelopes',default='gradation_envelopes.json')
ap.add_argument('--reuse-diagnostics',action='store_true',help='Reuse saved permutation outputs; models are not refitted')
args=ap.parse_args()
R=Path(args.results);F=Path(args.figures);F.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(R/'analysis_data_normalized.csv');p=pd.read_csv(R/'predictions.csv');m=pd.read_csv(R/'metrics.csv');folds=pd.read_csv(R/'folds.csv')
layers=['ac_base_course','binder_course','wearing_course'];labels=['AC base','Binder','Wearing'];colors=['#4477AA','#EE7733','#228833']
# Check every recorded sieve against each possible subtype in the supplied table.
envelopes=json.loads(Path(args.envelopes).read_text())
audit=[]
for _,r in d.iterrows():
    names=list(envelopes)[:2] if r.layer_type==layers[0] else ([list(envelopes)[2]] if r.layer_type==layers[1] else list(envelopes)[3:])
    compatible=[];fail=[]
    for name in names:
        bad=[f'{k}: {r[k]:g} outside {lo:g}–{hi:g}' for k,(lo,hi) in envelopes[name].items() if not lo-1e-8<=r[k]<=hi+1e-8]
        if not bad:compatible.append(name)
        fail.append(name+': '+ '; '.join(bad))
    audit.append({'file_name':r.file_name,'layer_type':r.layer_type,'compatible_envelope':' / '.join(compatible),'status':'compatible_with_supplied_table' if compatible else 'author_verification_required','detail':'' if compatible else ' | '.join(fail)})
audit=pd.DataFrame(audit);audit.to_csv(R/'gradation_audit_204.csv',index=False)
print('GRADATION FLAGS',audit[audit.status!='compatible_with_supplied_table'].to_dict('records'),flush=True)
# Recover grouped outer models from saved training/test assignments and parameters.
if args.reuse_diagnostics:
    stats=pd.read_csv(R/'group_permutation_summary.csv')
else:
    imp=[];gp=p[p.protocol=='quarry_nested'];rng=np.random.default_rng(20261014)
    groups={'Layer type':['layer_type'],'Gradation':GRAD,'Binder properties':BINDER,'Aggregate quality':QUALITY}
    for f in range(1,6):
        test_ids=gp[gp.fold==f].file_name.tolist();test=d[d.file_name.isin(test_ids)].copy();train=d[~d.file_name.isin(test_ids)].copy()
        z=folds[(folds.protocol=='quarry_nested')&(folds.fold==f)].iloc[0]
        params={k.replace('model__',''):v for k,v in json.loads(z.best_parameters).items()}
        model=pipeline(FEATURES,params).fit(train[FEATURES],train[TARGET])
        original=model.predict(test[FEATURES]);saved=gp.set_index('file_name').loc[test.file_name,'predicted_obc'].to_numpy()
        assert np.allclose(original,saved,atol=1e-12)
        base=np.mean(np.abs(test[TARGET].to_numpy()-original))
        for mode in ['Across layers','Within layer']:
            for name,cols in groups.items():
                if mode=='Within layer' and name=='Layer type':continue
                for repeat in range(30):
                    x=test[FEATURES].reset_index(drop=True).copy();perm=np.arange(len(x))
                    if mode=='Across layers':perm=rng.permutation(perm)
                    else:
                        for layer in layers:
                            idx=np.flatnonzero(x.layer_type.eq(layer));perm[idx]=rng.permutation(idx)
                    x.loc[:,cols]=x.loc[perm,cols].to_numpy()
                    error=np.mean(np.abs(test[TARGET].to_numpy()-model.predict(x)))
                    imp.append({'fold':f,'mode':mode,'group':name,'repeat':repeat,'n':len(test),'increase_mae':error-base})
        print('Permutation fold',f,'done',flush=True)
    imp=pd.DataFrame(imp);imp.to_csv(R/'group_permutation_repeats.csv',index=False)
    agg=imp.groupby(['mode','group','repeat']).apply(lambda z:np.average(z.increase_mae,weights=z.n),include_groups=False).rename('increase_mae').reset_index()
    stats=agg.groupby(['mode','group']).increase_mae.agg(['mean','std','min','max']).reset_index();stats.to_csv(R/'group_permutation_summary.csv',index=False)
    print(stats.to_string(index=False),flush=True)
desc=d[[TARGET,*GRAD,*BINDER,*QUALITY]].describe().T
desc.to_csv(R/'descriptive_204.csv')
q=pd.read_csv(R/'quarry_metrics.csv').query('protocol=="quarry_nested"');delta=q.mae_difference_et_minus_baseline
summary={'obc_mean':float(d[TARGET].mean()),'obc_sd':float(d[TARGET].std()),'quarry_delta_median':float(delta.median()),'quarry_delta_q25':float(delta.quantile(.25)),'quarry_delta_q75':float(delta.quantile(.75)),'quarries_et_better':int((delta<0).sum()),'gradation_flags':audit[audit.status!='compatible_with_supplied_table'].file_name.tolist()}
(R/'analysis_summary.json').write_text(json.dumps(summary,indent=2))
# Plain, colorblind-friendly, vector and 300-dpi figure exports.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
def save(fig,name):
    fig.savefig(F/(name+'.png'),dpi=300,bbox_inches='tight',facecolor='white');fig.savefig(F/(name+'.pdf'),bbox_inches='tight',facecolor='white');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(7,3.1),layout='constrained')
for i,layer in enumerate(layers):
    vals=d.loc[d.layer_type.eq(layer),TARGET].to_numpy();xx=np.random.default_rng(7+i).uniform(i-.16,i+.16,len(vals))
    axs[0].scatter(xx,vals,s=10,alpha=.45,color=colors[i],linewidths=0)
    axs[0].plot([i-.25,i+.25],[vals.mean()]*2,color='black',lw=1.5)
axs[0].set(xticks=range(3),xticklabels=[f'{l}\n(n={sum(d.layer_type==k)})' for l,k in zip(labels,layers)],ylabel='Experimental OBC (%)',title='(a) OBC by pavement layer')
ct=pd.crosstab(pd.to_datetime(d.design_date).dt.year,d.layer_type).reindex(columns=layers)
bottom=np.zeros(len(ct))
for k,lab,c in zip(layers,labels,colors):axs[1].bar(ct.index,ct[k],bottom=bottom,color=c,label=lab,width=.8);bottom+=ct[k]
axs[1].axvline(2021.5,color='black',ls='--',lw=1);axs[1].set(title='(b) Design years',ylabel='Records',xticks=[2011,2014,2017,2020,2023,2026]);axs[1].legend(frameon=False)
save(fig,'Figure_1_population')

fig,axs=plt.subplots(1,2,figsize=(7,3.4),layout='constrained',sharex=True,sharey=True)
for ax,prot,title in zip(axs,['quarry_nested','temporal_2022'],['(a) Quarry grouped CV (n=204)','(b) Temporal test (n=42)']):
    z=p[p.protocol==prot]
    for layer,lab,c in zip(layers,labels,colors):
        zz=z[z.layer_type==layer];ax.scatter(zz.observed_obc,zz.predicted_obc,s=22,color=c,alpha=.65,label=lab,edgecolor='white',lw=.3)
    ax.plot([3.6,5.2],[3.6,5.2],'k--',lw=1);ax.set(xlim=(3.6,5.2),ylim=(3.6,5.2),xlabel='Experimental OBC (%)',ylabel='Predicted OBC (%)',title=title);ax.set_aspect('equal')
axs[0].legend(frameon=False,loc='upper left');save(fig,'Figure_2_predictions')

fig,axs=plt.subplots(1,2,figsize=(7,3.2),layout='constrained')
for ax,metric,label in zip(axs,['mae','rmse'],['(a) MAE (percentage points)','(b) RMSE (percentage points)']):
    for j,(model,c) in enumerate([('Extra Trees','#4477AA'),('Layer mean','#CC6677')]):
        vals=[float(m[(m.protocol=='quarry_nested')&(m.layer==k)&(m.model==model)][metric].iloc[0]) for k in layers]
        ax.bar(np.arange(3)+(j-.5)*.32,vals,width=.32,color=c,label=model)
    ax.set(xticks=range(3),xticklabels=labels,ylabel=label,ylim=(0,.14));ax.grid(axis='y',alpha=.2)
axs[0].legend(frameon=False);save(fig,'Figure_3_layer_errors')

corr=pd.read_csv(R/'correlations.csv');cn=[f'{s.replace("_",".")} mm' for s in ['25_4','19_1','12_7','9_5','4_75','2_0','0_425','0_177','0_075']]+['Bitumen specific gravity','Penetration','Softening point','LA loss','MgSO4 loss','Flakiness index']
fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained');yy=np.arange(len(corr))
ax.hlines(yy,corr.pooled_r,corr.layer_adjusted_r,color='#BBBBBB',lw=1.2);ax.scatter(corr.pooled_r,yy,c='#4477AA',label='Pooled',s=26);ax.scatter(corr.layer_adjusted_r,yy,c='#EE7733',label='Layer adjusted',s=26)
ax.axvline(0,c='black',lw=.6);ax.set(yticks=yy,yticklabels=cn,xlim=(-.4,1),xlabel='Pearson correlation with OBC');ax.invert_yaxis();ax.legend(frameon=False,loc='lower right');save(fig,'Figure_4_correlations')

fig,axs=plt.subplots(1,2,figsize=(7,3.2),layout='constrained')
for ax,prot,title in zip(axs,['quarry_nested','temporal_2022'],['(a) Quarry grouped CV','(b) Temporal test']):
    z=p[p.protocol==prot]
    for col,label,c in [('absolute_error','Extra Trees','#4477AA'),('baseline_absolute_error','Layer mean','#CC6677')]:
        x=np.sort(z[col]);ax.step(np.r_[0,x],np.r_[0,np.arange(1,len(x)+1)/len(x)],where='post',color=c,label=label)
    for x in [.10,.25,.50]:ax.axvline(x,color='#888888',lw=.6,ls=':')
    ax.set(xlim=(0,.55),ylim=(0,1.02),xlabel='Absolute error (percentage points)',ylabel='Fraction of designs',title=title)
axs[0].legend(frameon=False,loc='lower right');save(fig,'Figure_5_tolerances')

fig,axs=plt.subplots(1,2,figsize=(7,3.1),layout='constrained',gridspec_kw={'width_ratios':[1,1]})
axs[0].scatter(np.arange(1,len(delta)+1),np.sort(delta),s=18,c='#4477AA');axs[0].axhline(0,c='black',lw=.7);axs[0].set(xlabel='Quarries ranked by MAE difference',ylabel='ET MAE − layer mean MAE (pp)',title='(a) Paired quarry differences')
st=stats[stats['mode']=='Within layer'].set_index('group').loc[['Gradation','Binder properties','Aggregate quality']]
axs[1].barh(np.arange(3),st['mean'],color='#228833',height=.55);axs[1].set(yticks=range(3),yticklabels=['Gradation','Binder\nproperties','Aggregate\nquality'],xlabel='Increase in MAE (pp)',title='(b) Permutation within layers');axs[1].axvline(0,c='black',lw=.7);axs[1].invert_yaxis()
save(fig,'Figure_6_diagnostics')
print('SAVED 6 figures',flush=True)
