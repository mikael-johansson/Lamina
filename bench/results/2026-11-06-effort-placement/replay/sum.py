import json,sys,collections as C,math
R=[json.loads(l) for l in open(sys.argv[1]+'/results.jsonl')]
g=C.defaultdict(C.Counter); dt=C.Counter()
for r in R: g[(r['tag'],r['arm'])][r['cls']]+=1; dt[r['arm']]+=r['dt']
for k in sorted(g): print(k,dict(g[k]))
for arm in ('end','start'):
    c=C.Counter(); 
    for k,v in g.items():
        if k[1]==arm: c.update(v)
    n=sum(c.values()); print(arm,n,dict(c),'think_only %.1f%%'%(100*c['think_only']/max(n,1)),'secs',round(dt[arm]))
