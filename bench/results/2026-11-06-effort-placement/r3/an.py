import sys,json,re,statistics as st,collections as C,math,random
R0='/home/mikaeljo/git/strata/eddoursul'; sys.path[:0]=[R0,R0+'/serve']
from pathlib import Path
import serve.server as SV, strata_tokenizer as ST
TD=Path('/d/nvme_512/strata-data/packs/iq3_s/tokenizer'); v=json.load(open(TD/'vocab.json')); t=[None]*len(v)
for k,i in v.items(): t[i]=k
tok=ST.Tokenizer(t,(TD/'merges.txt').read_text(encoding='utf-8').split('\n'),json.load(open(TD/'token_type.json')))
F=sys.argv[1]; R=[json.loads(l) for l in open(F)]
ARMS=[a for a in ('none','top','tail','last') if any(r['arm']==a for r in R)]
print('rows',len(R),'errors',sum(r['cls']=='err' for r in R)); R=[r for r in R if r['cls']!='err']
for r in R:
    th=r['text'].partition('</think>')[0]; r['tt']=len(tok.encode(th,parse_special=False)) if th else 0
    r['mention']=bool(re.search(r'reasoning effort|xhigh',th,re.I))
bk=sorted({r['bucket'] for r in R})
print('\nOUTCOMES (rows)'); cl=['no_close','call_in_thinking','think_only','length','tool','text']
print(f"{'':16}{'arm':6}{'n':>5}"+''.join(f"{c[:10]:>11}" for c in cl))
for b in bk+['ALL']:
    for a in ARMS:
        x=[r for r in R if r['arm']==a and (b=='ALL' or r['bucket']==b)]; c=C.Counter(r['cls'] for r in x)
        print(f"{b:16}{a:6}{len(x):5}"+''.join(f"{c[k]:11}" for k in cl))
print('\nTHINKING TOKENS: median reply / median per-prompt ratio to none / prompts longer than none / sign p')
def med(tag,a): 
    x=[r['tt'] for r in R if r['tag']==tag and r['arm']==a]; return st.median(x) if x else None
def sign(w,n):
    return min(1,2*sum(math.comb(n,i) for i in range(w,n+1))/2**n) if w>n/2 else min(1,2*sum(math.comb(n,i) for i in range(0,w+1))/2**n)
for b in bk+['ALL']:
    tags=sorted({r['tag'] for r in R if b=='ALL' or r['bucket']==b})
    for a in ARMS:
        x=[r['tt'] for r in R if r['arm']==a and r['tag'] in tags]
        line=f"{b:12}{a:6} n={len(x):4} med={st.median(x):7.0f} mean={st.mean(x):7.0f} p90={sorted(x)[int(.9*len(x))]:6}"
        if a!='none':
            ps=[(med(g,a),med(g,'none')) for g in tags]; ps=[p for p in ps if p[0] is not None and p[1] is not None]
            rt=[(p[0]+1)/(p[1]+1) for p in ps]; d=[p for p in ps if p[0]!=p[1]]; w=sum(p[0]>p[1] for p in d)
            line+=f"  ratio={st.median(rt):5.2f} longer={w}/{len(d)} p={sign(w,len(d)):.3f}"
        print(line)
print('\nbootstrap 95% CI for median per-prompt ratio to none, and tail vs top (resampling prompts)')
tags=sorted({r['tag'] for r in R}); rng=random.Random(1)
def ci(fa,fb):
    v=[(med(g,fa)+1)/(med(g,fb)+1) for g in tags if med(g,fa) is not None and med(g,fb) is not None]
    bs=sorted(st.median(rng.choices(v,k=len(v))) for _ in range(4000)); return st.median(v),bs[100],bs[3899],len(v)
for x,y in [('top','none'),('tail','none'),('last','none'),('tail','top'),('last','top')]:
    if x in ARMS and y in ARMS: print(x,'vs',y,'%.2f [%.2f, %.2f] n=%d'%ci(x,y))
print('\nPROMPT-LEVEL STOPS (prompts with >=1 no_close)')
for a in ARMS: 
    s=C.Counter(r['tag'] for r in R if r['arm']==a and r['cls']=='no_close'); print(a,len(s),dict(s))
print('\nmentions of effort in thinking, per arm:',{a:sum(r['mention'] for r in R if r['arm']==a) for a in ARMS})
print('\nno_close tails:')
for r in [r for r in R if r['cls']=='no_close'][:12]: print(' ',r['arm'],r['tag'],r['ct'],repr(r['text'][-110:]))
print('\nmention samples:')
for r in [r for r in R if r['mention'] and r['arm']!='none'][:6]:
    m=re.search(r'.{70}(reasoning effort|xhigh).{70}',r['text'],re.I|re.S); print(' ',r['arm'],r['tag'],repr(m.group(0)) if m else '')
