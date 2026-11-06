import json,collections as C,statistics as st,sys,math,re
R=[json.loads(l) for l in open(sys.argv[1])]; R=[r for r in R if r['cls']!='err']
reps=sorted({r.get('r',0) for r in R})
ARMS=('none','last','top')
for rr in reps+['all']:
    print('== rep',rr)
    for a in ARMS:
        x=[r for r in R if r['arm']==a and (rr=='all' or r.get('r',0)==rr)]
        if not x: continue
        tc=[r['ct'] for r in x]
        print(f"{a:5} n={len(x):3} correct={sum(r['correct'] for r in x):3} ({100*sum(r['correct'] for r in x)/len(x):4.1f}%) {dict(C.Counter(r['cls'] for r in x))} med_tokens={st.median(tc):6.0f} mean={st.mean(tc):6.0f} tool_call={sum('<tool_call' in r['text'] for r in x)}")
# paired by question
def acc(tag,a): x=[r['correct'] for r in R if r['tag']==tag and r['arm']==a]; return sum(x)/len(x) if x else None
tags=sorted({r['tag'] for r in R})
for a,b in (('last','none'),('top','none'),('last','top')):
    d=[(acc(t,a),acc(t,b)) for t in tags]; d=[x for x in d if None not in x]
    w=sum(x>y for x,y in d); l=sum(x<y for x,y in d); n=w+l
    p=min(1,2*sum(math.comb(n,i) for i in range(0,min(w,l)+1))/2**n) if n else 1
    print(f"{a} vs {b}: questions better {w}, worse {l}, same {len(d)-n}, sign p={p:.3f}")
print('\nwrong/no-answer by arm:')
for a in ARMS:
    print(a,sorted(r['tag']+':'+str(r['ans']) for r in R if r['arm']==a and not r['correct']))
print('\nnon-answers (tool calls / resumed task):')
for r in R:
    if r['cls']!='ok' or '<tool_call' in r['text']:
        aft=r['text'].partition('</think>')[2]
        print(' ',r['arm'],r['tag'],r.get('r',0),r['cls'],r['ct'],repr(r['text'].partition('</think>')[0][-150:]))
