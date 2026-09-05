"""Register SevenOfNine/Gemma-4-26B-A4B-It-Abliterated-FP8-Dynamic in johnny, cloned from the gemma placements."""
import yaml, copy, time, shutil
R='/home/rick/.config/johnny/registry.yaml'
shutil.copy(R, R+f".bak-{time.strftime('%Y%m%d-%H%M%S')}-pre-gemma-abl")
reg=yaml.safe_load(open(R))
src=reg['models']['gemma-4-26B-A4B-it-FP8-Dynamic']
name='gemma-4-26B-abliterated-FP8-Dynamic'
m={'identity': {**copy.deepcopy(src['identity']),
               'repo_id': 'SevenOfNine/Gemma-4-26B-A4B-It-Abliterated',
               'local_path': 'SevenOfNine/Gemma-4-26B-A4B-It-Abliterated-FP8-Dynamic',
               'vendor': 'SevenOfNine',
               'recommended_use': 'uncensored gemma-4 chat (Heretic v1.3 abliteration, KL 0.0845); FP8-Dynamic quantized locally 2026-09-04 with the RedHat recipe'},
   'capabilities': copy.deepcopy(src.get('capabilities') or {}),
   'placements': []}
for pid,newid in (('gemma-tp2-c4-mml262144-v0202','abl-gemma-tp2-c4-mml262144-v0202'),
                  ('induct-tp2-gmu0.92-seqs32-bt16384-mml110832','abl-gemma-tp2-seqs32-mml110832')):
    p=copy.deepcopy(next(p for p in src['placements'] if p['id']==pid))
    p['id']=newid; p['source']='manual'; p['validated_at']=None
    p.pop('perf',None); p.pop('quality',None)
    p.setdefault('extra',{})['note']=f'Cloned 2026-09-04 from gemma-4-26B-A4B-it-FP8-Dynamic/{pid} for the locally FP8-Dynamic-quantized Heretic abliteration; untested until benched.'
    m['placements'].append(p)
reg['models'][name]=m
yaml.safe_dump(reg,open(R,'w'),sort_keys=False,allow_unicode=True,width=1000)
print('registered', name, [p['id'] for p in m['placements']])
