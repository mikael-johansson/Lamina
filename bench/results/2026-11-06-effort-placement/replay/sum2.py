import json,sys,collections as C
R=[json.loads(l) for l in open(sys.argv[1]+'/results.jsonl')]
ctl=set(open(sys.argv[1]+'/ctl.txt').read().split())
def group(r): return 'random' if r['tag'] in ctl else 'long' if r['tag'] in('m234-0675ba','m230-97d565','m232-2bc96e') else 'short'
g=C.defaultdict(C.Counter); bytag=C.defaultdict(C.Counter); cts=C.defaultdict(list)
for r in R:
    g[(group(r),r['arm'])][r['cls']]+=1
    bytag[(r['tag'],r['arm'])][r['cls']]+=1
    if r['cls']=='no_close': cts[r['arm']].append(r['ct'])
for k in sorted(g):
    c=g[k]; n=sum(c.values()); print(f"{k[0]:7}{k[1]:16}{n:5} no_close {c['no_close']:4} {100*c['no_close']/max(n,1):6.1f}%  {dict(c)}")
print()
for k in sorted(bytag):
    if k[0] in ctl: c=bytag[k]; print(k,dict(c))
