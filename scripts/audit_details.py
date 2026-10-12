import json, ast, sys, collections
from pathlib import Path
import numpy as np
root=Path.cwd()
sys.path.insert(0,str(root/'src'))
from collapse.config import load_config
s=json.loads((root/'docs/audit/snapshot.json').read_text(encoding='utf-8'))
for prefix in ['pilot_en_s','pilot_bn_s','pilot_mmlu_en_s','pilot_mmlu_bn_s']:
    group=[r for r in s['runs'] if r['name'].startswith(prefix)]
    print('GROUP',prefix)
    for end in [0,-1]:
        ee=[next(iter(r['generations'][end]['eval'].values())) for r in group]
        print('END',end,{k:round(float(np.mean([e[k] for e in ee])),6) for k in ['accuracy','accuracy_generated','hedge_rate']},'SIGNALS',{n:{k:round(float(np.mean([e['signals'][n][k] for e in ee])),6) for k in ['ece','auroc']} for n in ee[0]['signals'] if n in ['p_pred','marker_conf','mean_token_prob']})
    for r in group[:1]:
        b=json.loads((root/'runs'/r['name']/'summary.json').read_text(encoding='utf-8'))
        print('STORED_SUMMARY', {n:{k:c[k] for k in ['ece_internal','accuracy','probe_auroc','ivd_auroc_gap']} for n,c in b['summary']['curves'].items()})
for r in s['runs']:
    if r['name'] in ['pilot_mmlu_en_s1','pilot_mmlu_bn_s1','pilot_bn_s1','bn_qwen3_r100']:
        print('POOLS',r['name'],r['pools'])
        print('PAIRS',[(t['n_syn_pairs'],t['n_real_pairs'],t['mix']['effective_ratio']) for t in r['training']])
        print('PROBES',[(g['generation'],{n:e['probe'] for n,e in g['eval'].items()}) for g in r['generations']])
print('VERDICTS', {n:(v['decision'],v['h1_supported_seeds'],v['h1_refuted_seeds']) for n,v in s['verdicts'].items()})
print('EXAMPLES',json.dumps(s['examples'],ensure_ascii=True))
sm={}
for p in sorted((root/'src').rglob('*.py')):
    text=p.read_text(encoding='utf-8'); tree=ast.parse(text)
    sm[p.relative_to(root).as_posix()]={'lines':len(text.splitlines()),'symbols':{n.name:n.lineno for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))}}
(root/'docs/audit/source_map.json').write_text(json.dumps(sm,indent=2),encoding='utf-8')
print('SOURCE',len(sm),'files',sum(v['lines'] for v in sm.values()),'lines')
print('CONFIGS',[(p.name,load_config(p).gen.top_p,load_config(p).data.seed_pool) for p in (root/'configs').glob('*.yaml')])

from collapse.analysis.pilot import load_pilot_runs, pilot_verdict
fresh={}
for prefix in ['pilot_en_s','pilot_bn_s','pilot_mmlu_en_s','pilot_mmlu_bn_s']:
    v=pilot_verdict(load_pilot_runs(root/'runs',prefix))
    fresh[prefix]=v
    print('CURRENT_VERDICT',prefix,v['decision'],v['h1_supported_seeds'],v['h1_refuted_seeds'],list(v['degrading_metrics']))
(root/'docs/audit/current_verdicts.json').write_text(json.dumps(fresh,indent=2),encoding='utf-8')
import torch, importlib.metadata
print('ENV',sys.version,torch.__version__,torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU',torch.cuda.get_device_name(0),torch.cuda.get_device_properties(0).total_memory)
print('VERSIONS',{n:importlib.metadata.version(n) for n in ['transformers','peft','numpy','scikit-learn','bitsandbytes','datasets']})
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for prefix,label,color in [('pilot_mmlu_en_s','English','#0072B2'),('pilot_mmlu_bn_s','Bangla','#D55E00')]:
    group=[r for r in s['runs'] if r['name'].startswith(prefix)]
    for j,(metric,title) in enumerate([('accuracy','Option-scoring accuracy'),('accuracy_generated','Generated-answer accuracy'),('p_pred','Chosen-option ECE')]):
        arr=np.array([[next(iter(g['eval'].values()))['signals']['p_pred']['ece'] if metric=='p_pred' else next(iter(g['eval'].values()))[metric] for g in r['generations']] for r in group])
        for row in arr: axes[j].plot(range(6),row,color=color,alpha=.2,lw=1)
        axes[j].plot(range(6),arr.mean(0),color=color,label=label,lw=2,marker='o')
        axes[j].set(title=title,xlabel='Generation',ylim=(0,1),xticks=range(6))
        axes[j].grid(alpha=.2)
axes[0].axhline(.25,color='gray',ls=':',lw=1,label='4-option chance')
axes[0].legend(fontsize=8)
fig.suptitle('Saved Global-MMLU pilots: descriptive results, measurement issues unresolved\nBold = mean of 3 seeds; thin = individual seeds; 400 fixed items',fontsize=11)
fig.savefig(root/'docs/audit/pilot_curves.png',dpi=180)
plt.close(fig)
lines=['# Saved experiment inventory','', 'Generated from saved artifacts on 2026-09-06. These are historical results, not a fresh training run.','', '| Run | Model | Evaluated generations | Summary | Requested ratio | First recorded effective ratio | Weights retained |','|---|---|---:|---|---:|---:|---:|']
for r in s['runs']:
    mix=r['training'][0]['mix']['effective_ratio'] if r['training'] else 'n/a'
    lines.append(f"| {r['name']} | {r['model']} | {len(r['generations'])} | {r['summary_exists']} | {r['ratio_requested']} | {mix} | {len(r['weights_present'])} |")
lines += ['','## Raw data row counts','','| Dataset | Mapped rows on disk |','|---|---:|']
for n,v in s['raw_data'].items(): lines.append(f'| {n} | {v:,} |')
lines += ['','## Source map','','| Module | Lines | Functions/classes (line) |','|---|---:|---|']
for n,v in sm.items():
    syms=', '.join(f'{k} ({line})' for k,line in v['symbols'].items())
    lines.append(f"| [{n}](../../{n}) | {v['lines']} | {syms} |")
(root/'docs/audit/INVENTORY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
