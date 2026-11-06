import sys,json,re,csv,random
R='/home/mikaeljo/git/strata/eddoursul'; sys.path[:0]=[R,R+'/serve']
from pathlib import Path
import serve.server as SV, strata_tokenizer as ST
S,GPQA=sys.argv[1],sys.argv[2]
TD=Path('/d/nvme_512/strata-data/packs/iq3_s/tokenizer'); v=json.load(open(TD/'vocab.json')); t=[None]*len(v)
for k,i in v.items(): t[i]=k
tok=ST.Tokenizer(t,(TD/'merges.txt').read_text(encoding='utf-8').split('\n'),json.load(open(TD/'token_type.json')))
sent=re.search(r"'(Reasoning effort is set to xhigh\.[^']*)'",open(R+'/serve/chat_template.jinja').read()).group(1)
gen=tok.encode('<|im_start|>assistant\n<think>\n',parse_special=True); turn=tok.encode(SV.EFFORT_TURN.format(sent),parse_special=True)
P=json.load(open(S+'/r3/prompts.json')); c=P['d#1']; print('context',c['logged'],c['file'][:50])
hn=c['none'][:-len(gen)]; ht=c['top'][:-len(gen)]; assert len(ht)-len(hn)==38
rows=list(csv.DictReader(open(GPQA,encoding='utf-8')))
out={}
for i in sorted(random.Random(20261103).sample(range(len(rows)),40)):
    r=rows[i]; ch=[r[k].strip() for k in ("Correct Answer","Incorrect Answer 1","Incorrect Answer 2","Incorrect Answer 3")]
    perm=list(range(4)); random.Random(f"0-{i}").shuffle(perm)
    q=("Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n"+r["Question"].strip()+"\n\n"+"\n".join(f"{'ABCD'[j]}) {ch[p]}" for j,p in enumerate(perm)))
    enc=lambda s:tok.encode(s,parse_special=False)
    op=tok.encode('<|im_start|>user\n',parse_special=True); cl=tok.encode('<|im_end|>\n',parse_special=True)
    u=op+enc(q)+cl; ul=op+enc(q+'\n\n'+sent)+cl
    out[f'g{i}']=dict(bucket='gpqa',gold='ABCD'[perm.index(0)],none=hn+u+gen,top=ht+u+gen,last=hn+ul+gen)
print(len(out),list(out)[:6],len(next(iter(out.values()))['none']))
json.dump(out,open(S+'/r5/prompts.json','w'))
