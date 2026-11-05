"""Abstract pickle opcode interpreter: never import/call GLOBAL, REDUCE or BUILD.
Reads bounded data.pkl bytes only, no data/<storage> ZIP member is opened.
Objects are inert Nodes; pickle payload code is never executed.
"""
import collections, hashlib, json, pathlib, pickletools, resource, time, zipfile
BASE=pathlib.Path('/home/acorn0415/Quantization/src/experiments/method_search_20261105/dino_wm_runtime_preparation')
ASSET=pathlib.Path('/home/acorn0415/sfq_data/dino_wm_planning/acquisition_20261106_v1/extracted/outputs/pusht/checkpoints/model_latest.pth')
class Node:
 def __init__(self,kind,payload=None):self.kind=kind;self.payload=payload;self.state=None;self.items=[]
def scan(data):
 stack=[];memo={};mark=object();global_records=[];pid_records=[];ops=collections.Counter()
 def marked():
  idx=next(i for i in range(len(stack)-1,-1,-1) if stack[i] is mark);v=stack[idx+1:];del stack[idx:];return v
 for op,arg,pos in pickletools.genops(data):
  n=op.name;ops[n]+=1
  if n=='PROTO':continue
  if n=='MARK':stack.append(mark)
  elif n=='EMPTY_DICT':stack.append(Node('dict'))
  elif n=='EMPTY_LIST':stack.append(Node('list'))
  elif n=='EMPTY_TUPLE':stack.append(())
  elif n in ('BINUNICODE','BININT1','BININT2','BININT','BINFLOAT'):stack.append(arg)
  elif n=='NONE':stack.append(None)
  elif n in ('NEWTRUE','NEWFALSE'):stack.append(n=='NEWTRUE')
  elif n in ('BINPUT','LONG_BINPUT'):memo[arg]=stack[-1]
  elif n in ('BINGET','LONG_BINGET'):stack.append(memo[arg])
  elif n=='GLOBAL':
   v=Node('global',arg);stack.append(v);global_records.append({'position':pos,'qualified_name':arg})
  elif n=='TUPLE':stack.append(tuple(marked()))
  elif n in ('TUPLE1','TUPLE2','TUPLE3'):
   k=int(n[-1]);v=tuple(stack[-k:]);del stack[-k:];stack.append(v)
  elif n in ('NEWOBJ','REDUCE'):
   args=stack.pop();fn=stack.pop();stack.append(Node(n,(fn,args)))
  elif n=='BUILD':state=stack.pop();stack[-1].state=state
  elif n=='BINPERSID':
   pid=stack.pop();node=Node('persistent_id',pid);stack.append(node);pid_records.append(node)
  elif n=='SETITEM':
   value=stack.pop();key=stack.pop();stack[-1].items.append((key,value))
  elif n=='SETITEMS':
   values=marked();assert len(values)%2==0;stack[-1].items.extend(zip(values[0::2],values[1::2]))
  elif n=='APPEND':value=stack.pop();stack[-1].items.append(value)
  elif n=='APPENDS':values=marked();stack[-1].items.extend(values)
  elif n=='STOP':assert len(stack)==1;root=stack[0]
  else:raise ValueError(('unsupported opcode',n,pos))
 assert root.kind=='dict'
 return root,global_records,pid_records,dict(ops)
def walk(value,seen=None):
 if seen is None:seen=set()
 if id(value) in seen:return
 seen.add(id(value));yield value
 if isinstance(value,Node):
  yield from walk(value.payload,seen);yield from walk(value.state,seen);yield from walk(value.items,seen)
 elif isinstance(value,(tuple,list)):
  for e in value:yield from walk(e,seen)
def summarize(value,infos):
 nodes=list(walk(value));globals_=sorted({n.payload for n in nodes if isinstance(n,Node) and n.kind=='global'})
 pids=[n.payload for n in nodes if isinstance(n,Node) and n.kind=='persistent_id'];stores={str(p[2]):{'location':p[3],'numel':p[4],'storage_type':p[1].payload,'archive_bytes':infos.get('model_2/data/'+str(p[2]))} for p in pids}
 return {'value_kind':value.kind if isinstance(value,Node) else type(value).__name__,'referenced_globals':globals_,'unique_storage_members':len(stores),'storage_member_uncompressed_bytes':sum(v['archive_bytes'] or 0 for v in stores.values()),'storage_metadata':stores}
def main():
 resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3));start=time.time()
 with zipfile.ZipFile(ASSET) as z:
  infos={i.filename:i.file_size for i in z.infolist()};name='model_2/data.pkl';assert infos[name]<=1024**2
  data=z.read(name)
 root,globals_,pids,ops=scan(data);keys=[k for k,v in root.items];assert all(isinstance(k,str) for k in keys);assert len(keys)==len(set(keys))
 result={'status':'metadata_only_abstract_stack_complete','asset_path':str(ASSET),'asset_size':ASSET.stat().st_size,'pickle_member':name,'pickle_bytes':len(data),'pickle_sha256':hashlib.sha256(data).hexdigest(),'zip_members':len(infos),'top_level_keys':keys,'top_level':{k:summarize(v,infos) for k,v in root.items},'GLOBAL_records':globals_,'opcodes':ops,'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'storage_payload_members_read':0,'torch_or_pickle_load_calls':0,'imports_of_named_globals':0,'scope':'Abstract syntactic stack only: tensor values, shape semantics, load compatibility, and numeric validity unverified. REDUCE/BUILD/NEWOBJ never called.','elapsed_seconds':time.time()-start,'maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
 (BASE/'SAFE_PICKLE_METADATA.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['top_level_keys','pickle_bytes','elapsed_seconds','maxrss_bytes']}));print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='storage_metadata'} for k,v in result['top_level'].items()},indent=2))
if __name__=='__main__':main()
