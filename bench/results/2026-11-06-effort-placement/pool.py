import json,sys,math,collections as C,statistics as st
S=sys.argv[1]
ST=[('none (0 tokens)','r6'),('78k','r7'),('142k','r5'),('167k','r8')]
def load(d):
    R=[json.loads(l) for l in open(f'{S}/{d}/results.jsonl')]; R=[r for r in R if r['cls']!='err']
    # complete rounds only: a round counts when all three forms have all 40 questions
    full={rr for rr in {r.get('r',0) for r in R} if all(sum(1 for r in R if r.get('r',0)==rr and r['arm']==a)==40 for a in('none','top','last'))}
    return [r for r in R if r.get('r',0) in full],len(full),len(R)
def sign(w,l):
    n=w+l; return min(1,2*sum(math.comb(n,i) for i in range(0,min(w,l)+1))/2**n) if n else 1
tot={a:[0,0] for a in('none','top','last')}; P={}
print(f"{'context':16}{'rounds':>7}  {'none':>12}{'top':>12}{'last':>12}   {'top-none':>15}{'last-none':>15}  median tokens n/t/l  nonanswers n/t/l")
for name,d in ST:
    R,k,raw=load(d)
    def acc(a): x=[r for r in R if r['arm']==a]; return sum(r['correct'] for r in x),len(x)
    def pq(a,b):
        w=l=0
        for t in {r['tag'] for r in R}:
            x=[r['correct'] for r in R if r['tag']==t and r['arm']==a]; y=[r['correct'] for r in R if r['tag']==t and r['arm']==b]
            w+=sum(x)>sum(y); l+=sum(x)<sum(y)
            P.setdefault((a,b),[]).append(sum(x)-sum(y))
        return w,l
    line=f"{name:16}{k:7}  "
    for a in ('none','top','last'):
        c,n=acc(a); tot[a][0]+=c; tot[a][1]+=n; line+=f"{c:4}/{n:<3} {100*c/n:3.0f}%"
    for a in ('top','last'):
        w,l=pq(a,'none'); line+=f"  +{w}/-{l} p={sign(w,l):.2f}"
    md=[st.median(r['ct'] for r in R if r['arm']==a) for a in ('none','top','last')]
    na=[sum(1 for r in R if r['arm']==a and r['cls']!='ok') for a in ('none','top','last')]
    print(line+f"  {md[0]:.0f}/{md[1]:.0f}/{md[2]:.0f}  {na[0]}/{na[1]}/{na[2]}  (rows on disk {raw})")
print('pooled:',{a:f"{c}/{n} {100*c/n:.1f}%" for a,(c,n) in tot.items()})
for k,v in P.items():
    w=sum(x>0 for x in v); l=sum(x<0 for x in v); print(k,'question-context pairs better',w,'worse',l,'p=%.3f'%sign(w,l))
