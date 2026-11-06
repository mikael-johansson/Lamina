import sys,json,re,csv,random,types
R='/home/mikaeljo/git/strata/eddoursul'; sys.path[:0]=[R,R+'/serve']
from pathlib import Path
from serve.frontend import ChatTemplate, openai_to_messages, literal_tags
import serve.server as SV, strata_tokenizer as ST
S,GPQA,CTX,OUT=sys.argv[1:5]
TD=Path('/d/nvme_512/strata-data/packs/iq3_s/tokenizer'); v=json.load(open(TD/'vocab.json')); t=[None]*len(v)
for k,i in v.items(): t[i]=k
tok=ST.Tokenizer(t,(TD/'merges.txt').read_text(encoding='utf-8').split('\n'),json.load(open(TD/'token_type.json')))
tpl=TD/'chat_template.jinja'; template=ChatTemplate(tpl if tpl.exists() else Path(R+'/serve/chat_template.jinja'))
sv=types.SimpleNamespace(tok=tok,template=template,effort_end=False,prompts=None,literals=literal_tags(getattr(tok,'control_tokens',())))
sv.render_prompt=lambda m,tt,k: SV.Service.render_prompt(sv,m,tt,k)
sent=re.search(r"'(Reasoning effort is set to xhigh\.[^']*)'",open(R+'/serve/chat_template.jinja').read()).group(1)
gen=tok.encode('<|im_start|>assistant\n<think>\n',parse_special=True)
rows=list(csv.DictReader(open(GPQA,encoding='utf-8'))); out={}
if CTX!='zero':
    c=json.load(open(S+'/r3/prompts.json'))[CTX]; print('context',CTX,c['logged'],c['lastrole']); hn=c['none'][:-len(gen)]; ht=c['top'][:-len(gen)]
op=tok.encode('<|im_start|>user\n',parse_special=True); cl=tok.encode('<|im_end|>\n',parse_special=True); enc=lambda s:tok.encode(s,parse_special=False)
for i in sorted(random.Random(20261103).sample(range(len(rows)),40)):
    r=rows[i]; ch=[r[k].strip() for k in ("Correct Answer","Incorrect Answer 1","Incorrect Answer 2","Incorrect Answer 3")]
    perm=list(range(4)); random.Random(f"0-{i}").shuffle(perm)
    q=("Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n"+r["Question"].strip()+"\n\n"+"\n".join(f"{'ABCD'[j]}) {ch[p]}" for j,p in enumerate(perm)))
    if CTX=='zero':   # through the template, as a plain chat request would be
        def e(eff,text):
            m,tt,k=openai_to_messages({"messages":[{"role":"user","content":text}],"reasoning_effort":eff}); k=dict(k); k['reasoning_effort']=eff
            return SV.Service.encode_prompt(sv,m,tt,k)
        none,top,last=e('medium',q),e('xhigh',q),e('medium',q+'\n\n'+sent)
    else:
        none=hn+op+enc(q)+cl+gen; top=ht+op+enc(q)+cl+gen; last=hn+op+enc(q+'\n\n'+sent)+cl+gen
    out[f'g{i}']=dict(bucket='gpqa',gold='ABCD'[perm.index(0)],none=none,top=top,last=last)
x=next(iter(out.values())); print(len(out),{a:len(x[a]) for a in('none','top','last')}); print(repr(tok.decode(x['top'][:60]))); print(repr(tok.decode(x['none'][:30])))
json.dump(out,open(OUT,'w'))
