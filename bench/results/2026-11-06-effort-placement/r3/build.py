import sys,json,re,random,hashlib,os,types
R='/home/mikaeljo/git/strata/eddoursul'
sys.path.insert(0,R); sys.path.insert(0,R+'/serve')
from pathlib import Path
from serve.frontend import ChatTemplate, openai_to_messages, literal_tags
import serve.server as SV
import strata_tokenizer as ST
S=sys.argv[1]; OUT=S+'/r3/prompts.json'
TD=Path('/d/nvme_512/strata-data/packs/iq3_s/tokenizer'); v=json.load(open(TD/'vocab.json')); t=[None]*len(v)
for k,i in v.items(): t[i]=k
tok=ST.Tokenizer(t,(TD/'merges.txt').read_text(encoding='utf-8').split('\n'),json.load(open(TD/'token_type.json')))
tpl=TD/'chat_template.jinja'; template=ChatTemplate(tpl if tpl.exists() else Path(R+'/serve/chat_template.jinja'))
sv=types.SimpleNamespace(tok=tok,template=template,effort_end=False,prompts=None,literals=literal_tags(getattr(tok,'control_tokens',())))
sv.render_prompt=lambda m,tt,k: SV.Service.render_prompt(sv,m,tt,k)
sent=re.search(r"'(Reasoning effort is set to xhigh\.[^']*)'",open(R+'/serve/chat_template.jinja').read()).group(1)
print('SENTENCE:',sent)
gen=tok.encode('<|im_start|>assistant\n<think>\n',parse_special=True)
turn=tok.encode(SV.EFFORT_TURN.format(sent),parse_special=True)
E=[e for e in json.load(open(S+'/eff.json')) if e['d']=='strata-requests' and e['eff'] in('medium','xhigh') and e['pt']]
B={'a_20-40k':(20000,40000,12),'b_40-80k':(40000,80000,12),'c_80-130k':(80000,130000,8),'d_130k+':(130000,10**7,8)}
rng=random.Random(20261105); out={}; seen=set()
for b,(lo,hi,want) in B.items():
    c=[e for e in E if lo<=e['pt']<hi]; rng.shuffle(c); got=0
    for e in c:
        if got>=want: break
        f='/d/hdd_800/logs/%s/%s'%(e['d'],e['n'])
        if not os.path.exists(f): continue
        x=open(f,errors='replace').read(); h=x[:x.find('=== RESPONSE')]
        try: req=json.JSONDecoder().raw_decode(h[h.find('{\n'):])[0]
        except Exception: continue
        ms=req.get('messages',[])
        if any(isinstance(m.get('content'),list) and any(p.get('type')!='text' for p in m['content']) for m in ms): continue
        us=[m for m in ms if m.get('role')=='user']
        conv=hashlib.sha1(json.dumps(us[:1]).encode()).hexdigest()
        if conv in seen: continue
        def enc(eff):
            r=json.loads(json.dumps(req)); r['reasoning_effort']=eff
            m,tt,k=openai_to_messages(r); k=dict(k); k['reasoning_effort']=eff
            return SV.Service.encode_prompt(sv,m,tt,k)
        try: none,top=enc('medium'),enc('xhigh')
        except Exception as ex: print('skip',e['n'][:40],ex); continue
        if len(top)-len(none)!=38 or none[-len(gen):]!=gen or top[-len(gen):]!=gen: print('skip shape',len(top)-len(none)); continue
        close=none[-len(gen)-2:-len(gen)]
        if tok.decode(close)!='<|im_end|>\n': print('skip close',repr(tok.decode(close))); continue
        seen.add(conv); got+=1
        tag='%s#%d'%(b[0],got)
        out[tag]=dict(bucket=b,logged=e['pt'],orig_eff=e['eff'],lastrole=ms[-1].get('role'),file=e['n'],none=none,top=top,
            tail=none[:-len(gen)]+turn+gen,
            last=none[:-len(gen)-2]+tok.encode('\n\n'+sent,parse_special=False)+close+gen)
        print(tag,e['ts'],e['eff'],e['pt'],len(none),len(top),ms[-1].get('role'),'|',tok.decode(out[tag]['tail'][-45:-len(gen)])[:60].replace('\n',' '))
    print(b,'got',got,'of',want,'candidates',len(c))
print(repr(tok.decode(out[next(iter(out))]['last'][-60:])))
json.dump(out,open(OUT,'w')); print('saved',len(out),os.path.getsize(OUT)//10**6,'MB')
