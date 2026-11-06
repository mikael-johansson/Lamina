import sys,json,re
R='/home/mikaeljo/git/strata/eddoursul'; sys.path[:0]=[R,R+'/serve']
from pathlib import Path
import serve.server as SV, strata_tokenizer as ST
S=sys.argv[1]
TD=Path('/d/nvme_512/strata-data/packs/iq3_s/tokenizer'); v=json.load(open(TD/'vocab.json')); t=[None]*len(v)
for k,i in v.items(): t[i]=k
tok=ST.Tokenizer(t,(TD/'merges.txt').read_text(encoding='utf-8').split('\n'),json.load(open(TD/'token_type.json')))
sent=re.search(r"'(Reasoning effort is set to xhigh\.[^']*)'",open(R+'/serve/chat_template.jinja').read()).group(1)
gen=tok.encode('<|im_start|>assistant\n<think>\n',parse_special=True); turn=tok.encode(SV.EFFORT_TURN.format(sent),parse_special=True)
O=json.load(open(S+'/replay/prompts.json')); ctl=set(open(S+'/replay/ctl.txt').read().split()); out={}
for k,p in O.items():
    none=p['start']; assert none[-len(gen):]==gen
    close=none[-len(gen)-2:-len(gen)]; assert tok.decode(close)=='<|im_end|>\n',k
    top=p['top_xhigh_only']; assert len(top)-len(none)==38,k
    out[k]=dict(bucket='random' if k in ctl else 'long' if p['logged']>100000 else 'failing',none=none,top=top,tail=none[:-len(gen)]+turn+gen,last=none[:-len(gen)-2]+tok.encode('\n\n'+sent,parse_special=False)+close+gen)
    print(k,out[k]['bucket'],p['logged'])
json.dump(out,open(S+'/r4/prompts.json','w'))
