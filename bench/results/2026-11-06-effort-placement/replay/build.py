import sys,json,glob,re,types
sys.path.insert(0,'/home/mikaeljo/git/strata/eddoursul')
sys.path.insert(0,'/home/mikaeljo/git/strata/eddoursul/serve')
from pathlib import Path
from serve.frontend import ChatTemplate, openai_to_messages, literal_tags
import serve.server as SV
import strata_tokenizer as ST
TD=Path('/d/nvme_512/strata-data/packs/iq3_s/tokenizer')
vocab=json.load(open(TD/'vocab.json'))
tokens=[None]*len(vocab)
for k,v in vocab.items(): tokens[v]=k
merges=[l.rstrip('\n') for l in open(TD/'merges.txt',encoding='utf-8') if l.strip() and not l.startswith('#version')]
types_=json.load(open(TD/'token_type.json'))
try:
    tok=ST.Tokenizer(tokens,merges,types_)
except Exception as e:
    print('tok fail',e); raise
tpl=TD/'chat_template.jinja'
template=ChatTemplate(tpl if tpl.exists() else Path('/home/mikaeljo/git/strata/eddoursul/serve/chat_template.jinja'))
def stub(end):
    s=types.SimpleNamespace(tok=tok,template=template,effort_end=end,prompts=None,literals=literal_tags(getattr(tok,'control_tokens',())))
    s.render_prompt=lambda m,t,k: SV.Service.render_prompt(s,m,t,k)
    return s
TAGS=sys.argv[1:]
out={}
for tag in TAGS:
    f=glob.glob('/d/hdd_800/logs/strata-requests/*-'+tag+'-*')[0]
    t=open(f,errors='replace').read(); i=t.find('=== RESPONSE'); h=t[:i]
    req=json.JSONDecoder().raw_decode(h[h.find('{\n'):])[0]
    logged=int(re.search(r'Prompt tokens\s*:\s*(\d+)',t).group(1))
    rec={'logged':logged}
    for arm,end in (('end',True),('start',False)):
        msgs,tools,kw=openai_to_messages(json.loads(json.dumps(req)))
        ids=SV.Service.encode_prompt(stub(end),msgs,tools,kw)
        rec[arm]=ids
    print(tag,'logged',logged,'end',len(rec['end']),'start',len(rec['start']),'match',logged==len(rec['end']),'common_prefix',next((n for n,(a,b) in enumerate(zip(rec['end'],rec['start'])) if a!=b),None))
    print('  end tail :',repr(tok.decode(rec['end'][-45:]))[:330]); print('  start tail:',repr(tok.decode(rec['start'][-12:]))[:120])
    out[tag]={'logged':logged,'end':rec['end'],'start':rec['start']}
json.dump(out,open(sys.argv[0].replace('build.py','prompts.json'),'w'))
