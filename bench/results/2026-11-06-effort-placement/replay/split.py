import json,sys
sys.path.insert(0,'/home/mikaeljo/git/strata/eddoursul'); sys.path.insert(0,'/home/mikaeljo/git/strata/eddoursul/serve')
from pathlib import Path
import serve.server as SV
import strata_tokenizer as ST
TD=Path('/d/nvme_512/strata-data/packs/iq3_s/tokenizer'); v=json.load(open(TD/'vocab.json')); t=[None]*len(v)
for k,i in v.items(): t[i]=k
tok=ST.Tokenizer(t,(TD/'merges.txt').read_text(encoding='utf-8').split('\n'),json.load(open(TD/'token_type.json')))
P=json.load(open(sys.argv[1]))
tail=tok.encode(SV.EFFORT_TURN.format(SV.EFFORT_TEXT['medium']),parse_special=True); gen=tok.encode('<|im_start|>assistant\n<think>\n',parse_special=True)
for k,p in P.items():
    e=p['end']; n=len(tail)+len(gen)
    assert e[-n:]==tail+gen, k
    assert p['start'][-len(gen):]==gen
    p['top_xhigh_only']=e[:-n]+gen
    p['tail_only']=p['start'][:-len(gen)]+tail+gen
    print(k,len(e),len(p['start']),len(p['top_xhigh_only']),len(p['tail_only']))
json.dump(P,open(sys.argv[1],'w'))
