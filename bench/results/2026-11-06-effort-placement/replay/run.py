import json,random,sys,time,urllib.request,subprocess,ssl
S=sys.argv[1]; N=int(sys.argv[2]); ONLY=sys.argv[3].split(',')
P=json.load(open(S+'/prompts.json'))
key=json.load(open('/home/mikaeljo/git/strata/configs/fork-iq3_s-split-prod.json'))['api_key']
out=open(S+'/results.jsonl','a')
def call(ids):
    body=dict(prompt=ids,max_tokens=1500,temperature=0.7,top_p=0.95,top_k=20,min_p=0.0)
    r=urllib.request.Request('http://127.0.0.1:8091/v1/completions',json.dumps(body).encode(),{'Content-Type':'application/json','Authorization':'Bearer '+key})
    return json.load(urllib.request.urlopen(r,timeout=1800))
rng=random.Random(7)
for round_ in range(N):
    for tag in ONLY:
        arms=['end','start']; rng.shuffle(arms)
        for arm in arms:
            t0=time.time()
            try:
                o=call(P[tag][arm]); c=o['choices'][0]; x=c['text']; fin=c['finish_reason']; u=o.get('usage',{})
                after=x.split('</think>',1)[1].strip() if '</think>' in x else None
                cls='length' if fin=='length' else 'no_close' if after is None else 'think_only' if not after else 'tool' if '<tool_call' in after or '<function' in after else 'text'
                rec=dict(tag=tag,arm=arm,r=round_,cls=cls,fin=fin,ct=u.get('completion_tokens'),dt=round(time.time()-t0,1),tail=x[-220:],head=x[:80])
            except Exception as e:
                rec=dict(tag=tag,arm=arm,r=round_,cls='err',err=str(e)[:200],dt=round(time.time()-t0,1))
            out.write(json.dumps(rec)+'\n'); out.flush()
print('done')
