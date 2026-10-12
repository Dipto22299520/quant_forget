"""Read-only research audit; writes a separate snapshot under docs/audit only."""
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from collapse.config import load_config
from collapse.utils import code_fingerprint

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def rows(p):
    with p.open(encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]

def main():
    out = {'code_fingerprint': code_fingerprint(), 'runs': [], 'raw_data': {}, 'verdicts': {}, 'examples': [], 'checks': {}}
    for p in sorted((ROOT/'data/raw').glob('*.jsonl')):
        with p.open('rb') as f:
            out['raw_data'][p.stem] = sum(bool(x.strip()) for x in f)
    for d in sorted((ROOT/'runs').iterdir()):
        if not d.is_dir():
            continue
        c = load_config(d/'config.yaml')
        entry = {'name': d.name, 'model': c.model.name, 'lang': c.data.lang, 'seed': c.seed, 'ratio_requested': c.loop.ratio, 'sft_size_requested': c.loop.sft_size, 'filter': c.loop.filter, 'loss': c.train.loss, 'mode': c.train.mode, 'chain': c.loop.chain, 'generations_requested': c.loop.generations, 'summary_exists': (d/'summary.json').exists(), 'generations': [], 'training': [], 'pools': {}, 'weights_present': []}
        for p in sorted((d/'pools').glob('*.jsonl')):
            rr = rows(p)
            entry['pools'][p.stem] = {'n':len(rr), 'with_context':sum(bool(r.get('context')) for r in rr), 'subjects':dict(collections.Counter(r.get('meta',{}).get('subject','') for r in rr))}
        for p in sorted(d.glob('gen*/_metrics.json'), key=lambda p:int(p.parent.name[3:])):
            m = read(p)
            ge = {'generation':m['generation'], 'fingerprint':m.get('code_fingerprint'), 'rescored':m.get('rescored',False), 'eval':{}}
            for name,e in m['eval'].items():
                rr=rows(p.parent/('eval_'+name+'.jsonl'))
                ge['eval'][name] = {k:e.get(k) for k in ['n','accuracy','accuracy_generated','accuracy_scored','hedge_rate','refusal_rate','truncation_rate']}
                ge['eval'][name].update({'empty_responses':sum(not r.get('response','').strip() for r in rr), 'records_with_context':sum(bool(r.get('context')) for r in rr), 'signals':{s:{k:v.get(k) for k in ['ece','auroc','overconfidence','coverage']} for s,v in e.get('signals',{}).items()}, 'probe':{k:e.get('probe',{}).get(k) for k in ['best_auroc','best_layer','error','stale']}, 'ivd':e.get('ivd',{})})
                if d.name.endswith('_s1') and m['generation'] in (0,5):
                    out['examples'].append({'run':d.name,'generation':m['generation'],'dataset':name,'records':[{k:r.get(k) for k in ['uid','question','answers','response','response_raw','correct_generated','correct_scored','marker_conf','mean_token_prob']} for r in rr[:2]]})
            entry['generations'].append(ge)
        for p in sorted(d.glob('ckpt/model_*/_done.json')):
            m=read(p)
            entry['training'].append({k:m.get(k) for k in ['generation_from','generation_to','code_fingerprint','mix','n_syn_pairs','n_real_pairs','filter_quality','train']})
        entry['weights_present']=[str(p.relative_to(d)) for p in d.glob('ckpt/**/*') if p.suffix in ('.safetensors','.bin','.pt')]
        out['runs'].append(entry)
    for p in (ROOT/'runs').glob('_verdict*.json'):
        v=read(p)
        out['verdicts'][p.stem]={k:v.get(k) for k in ['decision','reason','n_seeds','accuracy_g0','degrading_metrics','h1_supported_seeds','h1_refuted_seeds']}
    from collapse.markers.extract import marker_implied_confidence, extract_verbalized_confidence
    from collapse.eval.ivd import _spearman
    from collapse.eval.grade import parse_mcq_choice, contains_match
    import numpy as np
    out['checks']={'empty_marker_conf':marker_implied_confidence('', 'en')['marker_conf'], 'one_percent_parsed':extract_verbalized_confidence('Confidence: 1%'), 'constant_spearman':_spearman(np.ones(30),np.ones(30)), 'mcq_I_think_C':parse_mcq_choice('I think C.', ['one','two','three','four']), 'negated_gold_contains':contains_match('It is not Paris.', ['Paris'])}
    target=ROOT/'docs/audit/snapshot.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FINGERPRINT',out['code_fingerprint'])
    print('RAW',out['raw_data'])
    print('CHECKS',out['checks'])
    for r in out['runs']:
        print('RUN',r['name'],r['model'],'gens',len(r['generations']),'summary',r['summary_exists'],'pools',{k:v['n'] for k,v in r['pools'].items()}, 'mix',[t['mix'] for t in r['training'][:1]])
        if r['name'].startswith('pilot'):
            for g in r['generations']:
                for n,e in g['eval'].items():
                    print(' CURVE',g['generation'],n,'acc',e['accuracy'],'genacc',e['accuracy_generated'],'hedge',e['hedge_rate'],'empty',e['empty_responses'],'ece',{s:round(v['ece'],4) for s,v in e['signals'].items() if s in ('p_pred','marker_conf','mean_token_prob')},'fp',g['fingerprint'])
    print('VERDICTS',json.dumps(out['verdicts']))

if __name__=='__main__':
    main()
