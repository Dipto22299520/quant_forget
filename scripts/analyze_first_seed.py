"""Descriptive pilot analysis; paired item uncertainty is conditional on one seed."""
import json
import tarfile
from pathlib import Path
import sys
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from collapse.eval.calibration import ece, auroc

DATA = ROOT / 'docs/audit'
SOURCES = {
    '1.7B English': 'a100_pilot_1.7b_en_s1_summary.json',
    '1.7B Bangla': 'acl17b-bn-seed1-results.tgz',
    '8B English': 'acl8b-complete-results.tgz',
    '8B Bangla': 'acl8b-bn-seed1-results.tgz',
}

def load(name):
    path = DATA / name
    if path.suffix == '.json':
        return json.loads(path.read_text()), {}, []
    with tarfile.open(path) as archive:
        names = archive.getnames()
        summary = json.load(archive.extractfile(next(n for n in names if n.endswith('/summary.json'))))
        records = {}
        for g in range(6):
            member = next(n for n in names if f'/gen{g}/eval_global_mmlu_' in n and n.endswith('.jsonl'))
            records[g] = [json.loads(line) for line in archive.extractfile(member) if line.strip()]
        training = [json.load(archive.extractfile(n)) for n in names if '/ckpt/model_' in n and n.endswith('/_done.json')]
    return summary, records, training

def stat(rows):
    c = np.array([r['p_pred'] for r in rows])
    y = np.array([r['correct_scored'] for r in rows])
    return np.array([y.mean(), ece(c,y), auroc(c,y), c.mean(), np.mean((c-y)**2)])

def main():
    report = {'scope': 'Exploratory one-seed g5-minus-g0; 2000 paired item bootstrap draws, conditional on stored adjusted scores. No seed uncertainty or multiplicity adjustment.', 'runs': {}}
    fig, axes = plt.subplots(1, 3, figsize=(12,4))
    for label, source in SOURCES.items():
        summary, records, training = load(source)
        curve = []
        for m in summary['metrics']:
            e = next(iter(m['eval'].values()))
            sig = e['signals']['p_pred']
            curve.append({'g':m['generation'], 'accuracy':e['accuracy_scored'],
                          'generated_accuracy':e['accuracy_generated'], 'ece':sig['ece'],
                          'auroc':sig['auroc'], 'mean_conf':sig['mean_conf'],
                          'hedge':e['hedge_rate'], 'marker_auroc':e['signals']['marker_conf']['auroc']})
        out = {'curve':curve, 'delta':{k:curve[-1][k]-curve[0][k] for k in curve[0] if k!='g'},
               'training_pairs':[{'synthetic':x.get('n_syn_pairs'), 'real':x.get('n_real_pairs')} for x in training]}
        if records:
            a={r['uid']:r for r in records[0]}; b={r['uid']:r for r in records[5]}
            ids=sorted(a.keys() & b.keys())
            assert len(a)==len(records[0]) and len(b)==len(records[5]), 'Duplicate UIDs'
            assert set(a)==set(b), 'Endpoint item sets changed'
            aa=[a[i] for i in ids]; bb=[b[i] for i in ids]
            for g, rows in records.items():
                calculated=stat(rows)
                for pos,key in enumerate(['accuracy','ece','auroc','mean_conf']):
                    assert abs(calculated[pos]-curve[g][key])<1e-8, (label,g,key)
            rng=np.random.default_rng(20260912)
            draws=[]
            for _ in range(2000):
                idx=rng.integers(0,len(ids),len(ids))
                draws.append(stat([bb[i] for i in idx])-stat([aa[i] for i in idx]))
            ci=np.quantile(draws,[.025,.975],axis=0)
            out['paired_item_delta_ci95']={key:ci[:,j].tolist() for j,key in enumerate(['accuracy','ece','auroc','mean_conf','binary_brier'])}
            out['n_paired']=len(ids)
            out['subjects']=dict(Counter(str(r.get('meta',{}).get('subject','unknown')) for r in aa))
            out['answer_changes']={'wrong_to_right':sum(not a['correct_scored'] and b['correct_scored'] for a,b in zip(aa,bb)),
                                   'right_to_wrong':sum(a['correct_scored'] and not b['correct_scored'] for a,b in zip(aa,bb))}
        else:
            out['limitation']='Only summary available; no paired confidence intervals.'
        report['runs'][label]=out
        for ax,key,title in zip(axes,['accuracy','ece','auroc'],['Option-scored accuracy','Chosen-option ECE (15 bins)','Chosen-option AUROC']):
            ax.plot([c['g'] for c in curve],[c[key] for c in curve],marker='o',label=label)
            ax.set_title(title); ax.set_xlabel('Generation'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Exploratory first-seed pilots — descriptive trajectories')
    fig.tight_layout()
    fig.savefig(DATA/'first_seed_comparison.png',dpi=180)
    (DATA/'first_seed_analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for label,r in report['runs'].items():
        print(label, json.dumps(r))

if __name__=='__main__':
    main()
