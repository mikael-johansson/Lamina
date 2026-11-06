import json,random,sys,time,urllib.request,threading,queue
S,N,TAGS,ARMS,W=sys.argv[1],int(sys.argv[2]),sys.argv[3].split(','),sys.argv[4].split(','),int(sys.argv[5])
P=json.load(open(S+'/prompts.json')); key=json.load(open('/home/mikaeljo/git/strata/configs/fork-iq3_s-split-prod.json'))['api_key']
out=open(S+'/results.jsonl','a'); L=threading.Lock(); Q=queue.Queue(); rng=random.Random(N*len(TAGS)+W)
for r in range(N):
    for tag in TAGS:
        a=ARMS[:]; rng.shuffle(a)
        for arm in a: Q.put((tag,arm,r))
def call(ids):
    b=dict(prompt=ids,max_tokens=1500,temperature=0.7,top_p=0.95,top_k=20,min_p=0.0)
    rq=urllib.request.Request('http://127.0.0.1:8091/v1/completions',json.dumps(b).encode(),{'Content-Type':'application/json','Authorization':'Bearer '+key})
    return json.load(urllib.request.urlopen(rq,timeout=1800))
def work():
    while True:
        try: tag,arm,r=Q.get_nowait()
        except queue.Empty: return
        t0=time.time()
        try:
            o=call(P[tag][arm]); c=o['choices'][0]; x=c['text']; fin=c['finish_reason']; u=o.get('usage',{})
            after=x.split('</think>',1)[1].strip() if '</think>' in x else None
            cls='length' if fin=='length' else 'no_close' if after is None else 'think_only' if not after else 'tool' if '<tool_call' in after or '<function' in after else 'text'
            rec=dict(tag=tag,arm=arm,r=r,cls=cls,fin=fin,ct=u.get('completion_tokens'),dt=round(time.time()-t0,1),tail=x[-220:],head=x[:80])
        except Exception as e:
            rec=dict(tag=tag,arm=arm,r=r,cls='err',err=str(e)[:200],dt=round(time.time()-t0,1))
        with L: out.write(json.dumps(rec)+'\n'); out.flush()
ts=[threading.Thread(target=work) for _ in range(W)]
[t.start() for t in ts]; [t.join() for t in ts]; print('done')
