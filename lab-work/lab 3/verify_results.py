"""Verify metrics against saved predictions and reload the actual trained model."""
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from common import ROOT,configure

def main():
    configure()
    import numpy as np
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    out=ROOT/'outputs/exercise2'
    results=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    assert not results['smoke_test'], 'This verification requires a full training run.'
    rows=list(csv.DictReader((out/'test_predictions.csv').open(encoding='utf-8')))
    y=np.array([int(r['true_label']) for r in rows]);pred=np.array([int(r['predicted_label']) for r in rows])
    assert len(rows)==results['sizes']['test']
    tp=int(((y==1)&(pred==1)).sum());fp=int(((y==0)&(pred==1)).sum());fn=int(((y==1)&(pred==0)).sum())
    assert abs(float((y==pred).mean())-results['test']['test_accuracy'])<1e-10
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
    assert abs(f1-results['test']['test_f1'])<1e-10
    tokenizer=AutoTokenizer.from_pretrained(out/'model',local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(out/'model',local_files_only=True)
    model.eval()
    reloaded=[]
    with torch.no_grad():
        for start in range(0,len(rows),16):
            batch=tokenizer([r['text'] for r in rows[start:start+16]],padding=True,truncation=True,max_length=128,return_tensors='pt')
            reloaded.extend(model(**batch).logits.argmax(-1).tolist())
    assert reloaded==pred.tolist(), 'Saved model predictions differ from recorded predictions.'
    splits={s:{r['text'].strip().lower() for r in csv.DictReader((ROOT/'data'/(s+'.csv')).open(encoding='utf-8'))} for s in ['train','validation','test']}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not splits[a]&splits[b], f'Duplicate texts between {a} and {b}'
    print(f'PASS: metrics recomputed, {len(rows)} saved-model predictions match, no cross-split exact text duplicates.')

if __name__=='__main__':
    main()
