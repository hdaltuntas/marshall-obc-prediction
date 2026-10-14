"""Apply the author's binder interpretation without changing fitted models.

Author final correction dated 2026-10-14: modified binders are from Kirikkale.
PMB/KMB identify modified binder types. Conventional binders are from Aliaga.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from revision_validation import metrics

ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--results',required=True);a=ap.parse_args()
p=Path(a.results);raw=pd.read_excel(a.input);meta=pd.read_excel(a.input,sheet_name='Metadata')
meta['binder_label_original']=meta.binder_source_id
labels=meta.binder_label_original.fillna('').str.strip()
assert labels.str.startswith(('PMB','KMB')).sum()==16
meta['binder_type']=np.where(labels.str.startswith('PMB'),'PMB',np.where(labels.str.startswith('KMB'),'KMB','conventional'))
meta['binder_grade']=labels.where(meta.binder_type!='conventional','').str.replace(r'^(PMB|KMB)\s*','',regex=True).replace('',None)
meta['binder_source_id']=np.where(meta.binder_type=='conventional','Aliağa','Kırıkkale')
meta['quarry_id']=meta.quarry_id.str.strip()
meta.design_date=pd.to_datetime(meta.design_date)
meta.to_csv(p/'metadata_normalized.csv',index=False,date_format='%Y-%m-%d')
raw.merge(meta,on='file_name',validate='one_to_one',sort=False).to_csv(p/'analysis_data_normalized.csv',index=False,date_format='%Y-%m-%d')
pred=pd.read_csv(p/'predictions.csv').merge(meta[['file_name','binder_type','binder_grade']],on='file_name',validate='many_to_one')
rows=[]
for protocol,g in pred[pred.protocol.isin(['quarry_nested','record_nested','temporal_2022'])].groupby('protocol',sort=False):
 for subgroup,z in g[g.layer_type=='wearing_course'].groupby(g.binder_type.ne('conventional')):
  for model,col in [('Extra Trees','predicted_obc'),('Layer mean','baseline_obc')]:
   rows.append({'protocol':protocol,'subgroup':'modified_wearing' if subgroup else 'conventional_wearing','model':model,**metrics(z.observed_obc,z[col])})
pd.DataFrame(rows).to_csv(p/'binder_subgroup_metrics.csv',index=False)
pr=pd.read_csv(p/'protocol.json') if False else json.loads((p/'protocol.json').read_text())
pr['binder_clarification']={'date':'2026-10-14','source':'author final correction: modifiye bitümleri kırıkkale olarak al','refinery':'Aliağa for 188 conventional records; Kırıkkale for 15 PMB and 1 KMB records','types':meta.binder_type.value_counts().to_dict(),
 'note':'Binder type, grade, source and batch are not inputs in this analysis. Modified/unmodified comparisons are descriptive within wearing course.'}
(p/'protocol.json').write_text(json.dumps(pr,ensure_ascii=False,indent=2))
source={}
for sheet in pd.ExcelFile(a.input).sheet_names:
 d=pd.read_excel(a.input,sheet_name=sheet)
 if sheet=='Metadata':d=meta
 source[sheet]={'headers':d.columns.tolist(),'rows':d.astype(object).where(pd.notna(d),None).values.tolist()}
source['modeling_count']=len(raw)
(p/'workbook_source_204.json').write_text(json.dumps(source,ensure_ascii=False,default=lambda x:x.strftime('%Y-%m-%d') if isinstance(x,pd.Timestamp) else x,allow_nan=False))
print('BINDER_COUNTS',meta.binder_type.value_counts().to_dict())
print(pd.DataFrame(rows).to_string(index=False))
