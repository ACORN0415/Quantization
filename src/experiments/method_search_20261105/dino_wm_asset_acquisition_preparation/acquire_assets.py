"""Bounded two-ZIP acquisition; requires an external root-approved manifest.
Payloads are never deserialized/imported/executed. --selftest uses only tiny local ZIPs.
"""
import argparse,hashlib,http.client,json,os,re,resource,signal,stat,sys,tempfile,time,traceback
import urllib.error,urllib.parse,urllib.request,zipfile
from pathlib import Path,PurePosixPath
HERE=Path(__file__).resolve().parent
SPEC=json.loads((HERE/'ROOT_SPEC.json').read_text())
LIMITS=SPEC['planned_future_execution_limits_not_authorized']
METADATA=Path(SPEC['inputs'])
class Reject(Exception):pass

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def write_json(path,value):
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    tmp=Path(str(path)+'.tmp');tmp.write_text(text);tmp.replace(path)
def tree_bytes(path):
    n=0
    if not Path(path).exists():return 0
    for p in Path(path).rglob('*'):
        try:
            if p.is_file():n+=p.stat().st_size
        except FileNotFoundError:pass
    return n

def safe_name(name):
    if not isinstance(name,str) or not name or '\x00' in name or '\\' in name:raise Reject('invalid member name')
    p=PurePosixPath(name)
    if p.is_absolute() or re.match(r'^[A-Za-z]:',name) or any(x in ('','..','.') for x in name.rstrip('/').split('/')):raise Reject('unsafe member path')
    return p

def junk(name):return '__MACOSX' in PurePosixPath(name).parts or PurePosixPath(name).name=='.DS_Store'
def inspect_info(info):
    safe_name(info.filename)
    if info.flag_bits&1:raise Reject('encrypted member')
    mode=(info.external_attr>>16)&0o170000
    if info.create_system==3 and mode not in (0,stat.S_IFREG,stat.S_IFDIR):raise Reject('symlink or special member')
    if info.file_size<0 or info.compress_size<0:raise Reject('negative member size')

def selected_infos(zf,names):
    if not names or len(names)!=len(set(names)):raise Reject('empty or duplicate explicit selectors')
    table={}
    for info in zf.infolist():
        inspect_info(info)
        if info.filename in table:raise Reject('duplicate archive name')
        table[info.filename]=info
    selected=[]
    for name in names:
        safe_name(name)
        if junk(name):raise Reject('junk member selected')
        if name not in table or table[name].is_dir():raise Reject('missing member or directory selector')
        selected.append(table[name])
    return selected

class Budget:
    def __init__(self,limits,deadline,initial_disk=0):
        self.limits=limits;self.deadline=deadline;self.http=0;self.extracted=0;self.disk=initial_disk;self.events=[]
    def check(self):
        if time.time()>self.deadline:raise Reject('absolute execution deadline')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>self.limits['RSS_bytes']:raise Reject('RSS cap')
        if self.http>self.limits['HTTP_body_bytes'] or self.extracted>self.limits['extracted_member_bytes'] or self.disk>self.limits['disk_bytes']:raise Reject('byte cap')
    def before_http(self,n):
        self.check()
        if self.http+n>self.limits['HTTP_body_bytes']:raise Reject('HTTP body cap')
    def before_write(self,n,extraction=False):
        self.check()
        if self.disk+n>self.limits['disk_bytes']:raise Reject('combined disk cap')
        if extraction and self.extracted+n>self.limits['extracted_member_bytes']:raise Reject('extraction cap')

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):return None

def open_archive(url,budget):
    opener=urllib.request.build_opener(NoRedirect())
    for hop in range(9):
        budget.check()
        req=urllib.request.Request(url,headers={'User-Agent':'bounded-asset-acquisition/1','Accept-Encoding':'identity'})
        try:r=opener.open(req,timeout=min(20,max(.1,budget.deadline-time.time())))
        except urllib.error.HTTPError as exc:r=exc
        u=urllib.parse.urlsplit(url)
        budget.events.append({'url':urllib.parse.urlunsplit((u.scheme,u.netloc,u.path,'','')),'status':r.code,'body_bytes':0})
        if r.code in (301,302,303,307,308):
            location=r.headers.get('Location');r.close()
            if not location:raise Reject('redirect without location')
            url=urllib.parse.urljoin(url,location)
            if urllib.parse.urlsplit(url).scheme!='https':raise Reject('non-HTTPS redirect')
            continue
        return r
    raise Reject('too many redirects; no retry')

def download_one(asset,archive_dir,budget):
    expected=asset['size'];budget.before_http(expected);budget.before_write(expected)
    target=archive_dir/(PurePosixPath(asset['path']).name)
    part=Path(str(target)+'.part')
    if target.exists() or part.exists():raise Reject('existing archive/part; no resume or overwrite')
    r=open_archive(asset['url'],budget)
    try:
        if r.code!=200:raise Reject(f'full archive HTTP{r.code}; body not read')
        if r.headers.get('Content-Length')!=str(expected):raise Reject('full archive Content-Length mismatch')
        if r.headers.get('Content-Encoding','identity').lower() not in ('','identity'):raise Reject('encoded HTTP payload rejected')
        h=hashlib.sha256();count=0
        with part.open('xb') as out:
            while count<expected:
                budget.check();take=min(1024**2,expected-count)
                try:b=r.read1(take)
                except http.client.IncompleteRead as exc:
                    budget.before_http(len(exc.partial));budget.http+=len(exc.partial);budget.events[-1]['body_bytes']+=len(exc.partial)
                    raise Reject('incomplete HTTP body') from exc
                if not b:raise Reject('truncated HTTP body')
                budget.before_http(len(b));budget.http+=len(b);budget.events[-1]['body_bytes']+=len(b)
                budget.before_write(len(b));out.write(b);budget.disk+=len(b);count+=len(b);h.update(b)
            out.flush();os.fsync(out.fileno())
        if count!=expected or h.hexdigest()!=asset['sha256']:raise Reject('full ZIP size/SHA256 mismatch')
        if target.exists():raise Reject('archive final target appeared')
        os.rename(part,target);target.chmod(0o400)
        return target,{'path':str(target),'bytes':count,'sha256':h.hexdigest(),'verified_before_extraction':True}
    finally:r.close()

def make_parents(root,relative):
    here=root
    for part in relative.parts[:-1]:
        here=here/part
        try:here.mkdir(mode=0o700)
        except FileExistsError:pass
        if here.is_symlink() or not here.is_dir():raise Reject('unsafe extraction parent')
    return here/relative.name

def extract_selected(archive,names,root,budget):
    results=[]
    with zipfile.ZipFile(archive,'r',allowZip64=True) as zf:
        infos=selected_infos(zf,names)
        declared=sum(x.file_size for x in infos);budget.before_write(declared,extraction=True)
        # All selected names are explicit. Never call extract/extractall.
        for info in infos:
            budget.check();rel=safe_name(info.filename);dest=make_parents(root,rel)
            if dest.exists() or dest.is_symlink():raise Reject('selected destination already exists')
            count=0;h=hashlib.sha256()
            with zf.open(info,'r') as src,dest.open('xb') as out:
                while True:
                    budget.check()
                    b=src.read(min(1024**2,max(1,info.file_size-count+1)))
                    if not b:break  # ZipExtFile reaches EOF and verifies CRC.
                    if count+len(b)>info.file_size:raise Reject('member exceeds declared uncompressed size')
                    budget.before_write(len(b),extraction=True)
                    out.write(b);budget.disk+=len(b);budget.extracted+=len(b);count+=len(b);h.update(b)
                out.flush();os.fsync(out.fileno())
            if count!=info.file_size:raise Reject('member uncompressed count mismatch')
            results.append({'member':info.filename,'path':str(dest),'bytes':count,'sha256':h.hexdigest(),'CRC32':f'{info.CRC:08x}','CRC_verified_by_zipfile':True})
    return results

def validate_manifest(path):
    p=Path(path)
    if p.stat().st_size>1024**2:raise Reject('manifest too large')
    m=json.loads(p.read_text())
    if not (m.get('root_execution_authorized') is True and m.get('actual_claude_source_review_completed') is True and m.get('status')=='approved_for_one_dino_wm_asset_acquisition'):raise Reject('root/actual source review approval missing')
    if m['source_sha256']!=digest(__file__) or m['metadata_result_sha256']!=digest(METADATA) or m['root_spec_sha256']!=digest(HERE/'ROOT_SPEC.json'):raise Reject('code/spec/metadata source binding mismatch')
    if m['limits']!=LIMITS:raise Reject('manifest limits changed')
    evidence=m.get('review_evidence_sha256',{})
    if not evidence:raise Reject('review evidence not bound')
    for f,s in evidence.items():
        if digest(f)!=s:raise Reject('review evidence SHA mismatch')
    data=json.loads(METADATA.read_text());pinned=data['HF_metadata'];archives=m['archives']
    expected={a['path']:a for a in data['archives']}
    if len(archives)!=2 or {a['path'] for a in archives}!=set(expected):raise Reject('exact two named archives required')
    if m['revision']!=pinned['revision']:raise Reject('revision mismatch')
    all_paths=set()
    for a in archives:
        original=expected[a['path']]['HF_file_metadata'];lfs=original['lfs']
        url=f"https://huggingface.co/datasets/{pinned['repo']}/resolve/{pinned['revision']}/{a['path']}"
        if a['url']!=url or a['size']!=original['size'] or a['sha256']!=lfs['sha256']:raise Reject('archive URL/size/SHA not pinned')
        if not isinstance(a['members'],list) or not a['members']:raise Reject('explicit member list required')
        for n in a['members']:
            safe_name(n)
            if junk(n) or n in all_paths:raise Reject('junk/duplicate destination selector')
            all_paths.add(n)
    if sum(a['size'] for a in archives)>LIMITS['HTTP_body_bytes']:raise Reject('archives exceed HTTP cap')
    dest=Path(m['destination'])
    if not dest.is_absolute() or '..' in dest.parts or not dest.is_relative_to('/home/acorn0415') or dest==Path('/home/acorn0415'):raise Reject('destination must be a new absolute task directory below account')
    if dest.exists() or dest.is_symlink():raise Reject('destination already exists; no overwrite/reuse')
    if not dest.parent.is_dir() or dest.parent.resolve()!=dest.parent:raise Reject('destination parent missing or symlinked')
    if HERE==dest or HERE.is_relative_to(dest) or dest.is_relative_to(HERE):raise Reject('destination must not overlap source/record directory')
    return m,dest,digest(p)

def run(manifest_path):
    start=time.time();deadline=start+LIMITS['wall_seconds']
    m,dest,manifest_sha=validate_manifest(manifest_path)
    if (HERE/'RUN.json').exists():raise Reject('one execution only')
    with (HERE/'RUN.json').open('x') as f:json.dump({'attempt':1,'started_utc':start,'deadline_utc':deadline,'manifest_sha256':manifest_sha,'destination':str(dest)},f)
    archive_dir=dest/'archives';asset_dir=dest/'extracted'
    state={'status':'running','started_utc':start,'deadline_utc':deadline,'manifest_sha256':manifest_sha,'source_sha256':digest(__file__),'destination':str(dest),'archives':[],'GPU_calls':0,'payload_deserializations':0,'installations':0}
    # Conservative record/temp-file reserve; never spend the entire disk cap on payload.
    budget=Budget(LIMITS,deadline,tree_bytes(HERE)+32*1024**2)
    def stop(sig,frame):raise KeyboardInterrupt(f'signal {sig}')
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:
        os.umask(0o077);dest.mkdir(mode=0o700);archive_dir.mkdir();asset_dir.mkdir()
        for asset in m['archives']:
            path,record=download_one(asset,archive_dir,budget);state['archives'].append(record)
            record['selected_members']=extract_selected(path,asset['members'],asset_dir,budget)
            budget.check();write_json(HERE/'PROGRESS.json',state)
        budget.check()
        if digest(manifest_path)!=manifest_sha or digest(__file__)!=state['source_sha256'] or digest(HERE/'ROOT_SPEC.json')!=m['root_spec_sha256'] or digest(METADATA)!=m['metadata_result_sha256']:raise Reject('source/spec/metadata/manifest changed during acquisition')
        actual=tree_bytes(dest)+tree_bytes(HERE)
        if actual>LIMITS['disk_bytes']:raise Reject('final combined disk cap')
        budget.check()
        state.update(status='completed_verified_selected_assets_not_model_execution',disk_bytes_observed=actual)
    except BaseException as exc:
        state.update(status='failed_incomplete_assets_do_not_use',error=repr(exc),traceback=traceback.format_exc())
    finally:
        state.update(finished_utc=time.time(),HTTP_body_bytes=budget.http,extracted_member_bytes=budget.extracted,
                     peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,HTTP_events=budget.events)
        write_json(HERE/'RESULT.json',state)
    if not state['status'].startswith('completed_'):raise SystemExit(1)
    print(json.dumps({'status':state['status'],'HTTP_body_bytes':budget.http,'extracted_member_bytes':budget.extracted}))

def selftest():
    checks=[];tiny={**LIMITS,'HTTP_body_bytes':4096,'extracted_member_bytes':4096,'disk_bytes':16384}
    with tempfile.TemporaryDirectory(prefix='local_zip_test_',dir=HERE) as td:
        td=Path(td);z=td/'tiny.zip'
        with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:f.writestr('bundle/config.yaml',b'a: 1\n');f.writestr('bundle/unused.txt',b'excluded')
        out=td/'out';out.mkdir();b=Budget(tiny,time.time()+30)
        r=extract_selected(z,['bundle/config.yaml'],out,b)
        assert (out/'bundle/config.yaml').read_bytes()==b'a: 1\n' and not (out/'bundle/unused.txt').exists();checks.append('only selected member extracted')
        assert r[0]['sha256']==hashlib.sha256(b'a: 1\n').hexdigest() and b.extracted==5;checks.append('exact bytes SHA CRC and count')
        for bad in ('../escape','/absolute','C:/drive','a/../../x','a\\x','a/./b'):
            try:safe_name(bad)
            except Reject:checks.append('unsafe path rejected')
            else:raise AssertionError(bad)
        link=zipfile.ZipInfo('link');link.create_system=3;link.external_attr=(stat.S_IFLNK|0o777)<<16
        try:inspect_info(link)
        except Reject:checks.append('symlink rejected')
        else:raise AssertionError('symlink')
        enc=zipfile.ZipInfo('encrypted');enc.flag_bits=1
        try:inspect_info(enc)
        except Reject:checks.append('encryption rejected')
        else:raise AssertionError('encrypted')
        with zipfile.ZipFile(z) as f:
            for names in (['missing'],['bundle/config.yaml']*2):
                try:selected_infos(f,names)
                except Reject:checks.append('missing or duplicate selector rejected')
                else:raise AssertionError('selector')
        small=Budget({**tiny,'extracted_member_bytes':4},time.time()+30)
        new=td/'small';new.mkdir()
        try:extract_selected(z,['bundle/config.yaml'],new,small)
        except Reject:checks.append('declared extraction cap rejected before write')
        else:raise AssertionError('cap')
        assert not list(new.rglob('*'))
        try:extract_selected(z,['bundle/config.yaml'],out,Budget(tiny,time.time()+30))
        except Reject:checks.append('existing destination never overwritten')
        else:raise AssertionError('overwrite')
        # Local in-memory HTTP stand-in: no opener, socket, or remote asset access.
        import io
        payload=b'local ZIP transfer fixture';old_open=globals()['open_archive']
        class Response:
            code=200
            headers={'Content-Length':str(len(payload))}
            def __init__(self):self.f=io.BytesIO(payload)
            def read1(self,n):return self.f.read(n)
            def close(self):self.f.close()
        def fake_open(url,budget):
            budget.events.append({'url':'synthetic-only','status':200,'body_bytes':0})
            return Response()
        globals()['open_archive']=fake_open
        try:
            archive_dir=td/'archives';archive_dir.mkdir()
            asset={'path':'fixture.zip','url':'not-a-network-URL','size':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
            got,record=download_one(asset,archive_dir,Budget(tiny,time.time()+30))
            assert got.read_bytes()==payload and record['verified_before_extraction'] and not Path(str(got)+'.part').exists()
            checks.append('full size and SHA checked before atomic final rename')
            bad={**asset,'path':'bad.zip','sha256':'0'*64}
            try:download_one(bad,archive_dir,Budget(tiny,time.time()+30))
            except Reject:checks.append('SHA mismatch retains part and rejects final archive')
            else:raise AssertionError('bad SHA accepted')
            assert (archive_dir/'bad.zip.part').exists() and not (archive_dir/'bad.zip').exists()
        finally:globals()['open_archive']=old_open
        unauthorized=td/'unauthorized.json';unauthorized.write_text('{"root_execution_authorized":false}')
        try:validate_manifest(unauthorized)
        except Reject:checks.append('unapproved manifest rejected before acquisition')
        else:raise AssertionError('unauthorized accepted')
    return {'status':'local_tiny_ZIP_checks_passed','checks':checks,'count':len(checks),'network_requests':0,'GPU':0,'model_tensor_loads':0,'source_sha256':digest(__file__)}

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--manifest');g.add_argument('--selftest',action='store_true');a=p.parse_args()
    if a.selftest:print(json.dumps(selftest()))
    else:run(a.manifest)
