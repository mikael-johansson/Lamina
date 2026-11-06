import json,random,sys,time,urllib.request,threading,queue,re,os
S,W,MAXT,REPS=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
P=json.load(open(S+'/prompts.json')); key=json.load(open('/home/mikaeljo/git/strata/configs/fork-iq3_s-split-prod.json'))['api_key']
done=set()
if os.path.exists(S+'/results.jsonl'):
    for l in open(S+'/results.jsonl'):
        x=json.loads(l)
        if x['cls']!='err': done.add((x['tag'],x['arm'],x.get('r',0)))
out=open(S+'/results.jsonl','a'); L=threading.Lock(); Q=queue.Queue(); rng=random.Random(9)
tags=list(P); rng.shuffle(tags)
# no-sentence prefix arms first (cache shared), then top; every question once per arm
for r in range(REPS):
  for arm in ('none','last','top'):
    for tag in tags:
        if (tag,arm,r) not in done: Q.put((tag,arm,r))
def call(ids):
    b=dict(prompt=ids,max_tokens=MAXT,temperature=0.7,top_p=0.95,top_k=20,min_p=0.0)
    rq=urllib.request.Request('http://127.0.0.1:8091/v1/completions',json.dumps(b).encode(),{'Content-Type':'application/json','Authorization':'Bearer '+key})
    return json.load(urllib.request.urlopen(rq,timeout=3600))
def work():
    while True:
        try: tag,arm,r=Q.get_nowait()
        except queue.Empty: return
        t0=time.time()
        try:
            o=call(P[tag][arm]); c=o['choices'][0]; x=c['text']; fin=c['finish_reason']; u=o.get('usage',{})
            th,sep,aft=x.partition('</think>'); m=re.findall(r"(?i)answer\s*[:：]\s*\**\s*\(?\$?\\?(?:boxed\{|text\{)?([A-D])\b",aft)
            ans=m[-1].upper() if m else None
            cls='length' if fin=='length' else 'no_close' if not sep else 'ok' if ans else 'no_answer'
            rec=dict(tag=tag,arm=arm,r=r,cls=cls,fin=fin,ct=u.get('completion_tokens'),pt=u.get('prompt_tokens'),gold=P[tag]['gold'],ans=ans,correct=ans==P[tag]['gold'],think_chars=len(th),dt=round(time.time()-t0,1),text=x)
        except Exception as e:
            rec=dict(tag=tag,arm=arm,r=r,cls='err',err=str(e)[:200],dt=round(time.time()-t0,1))
        with L: out.write(json.dumps(rec)+'\n'); out.flush()
ts=[threading.Thread(target=work) for _ in range(W)]
[t.start() for t in ts]; [t.join() for t in ts]; print('done')
