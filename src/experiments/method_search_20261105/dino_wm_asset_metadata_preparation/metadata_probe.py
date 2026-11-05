"""Strict HTTP Range ZIP metadata reader. Never opens a member or reads HTTP200 archive bodies."""
import collections,hashlib,json,os,re,resource,struct,time,traceback,urllib.error,urllib.parse,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent
SPEC=json.loads((HERE/'ROOT_SPEC.json').read_text())
LIMIT=SPEC['limits'];END=SPEC['deadline_epoch']
class Unsupported(Exception):pass
class Invalid(Exception):pass
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):return None

def sha(b):return hashlib.sha256(b).hexdigest()
def clean_url(url):
    u=urllib.parse.urlsplit(url);return urllib.parse.urlunsplit((u.scheme,u.netloc,u.path,'',''))
def save(name,obj):
    text=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    disk=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
    if disk+len(text.encode())>LIMIT['output_bytes']:raise Invalid('output disk cap')
    tmp=HERE/(name+'.tmp');tmp.write_text(text);tmp.replace(HERE/name)
def guard():
    if time.time()>END:raise Invalid('registered absolute deadline')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>LIMIT['RSS_bytes']:raise Invalid('RSS cap')

class HTTP:
    def __init__(self):self.bytes=0;self.requests=[];self.opener=urllib.request.build_opener(NoRedirect())
    def fetch(self,url,*,byte_range=None,expected_total=None,max_bytes=None):
        guard();original=url;redirects=[]
        for hop in range(9):
            headers={'User-Agent':'bounded-zip-metadata-probe/1','Accept-Encoding':'identity'}
            if byte_range is not None:headers['Range']=f'bytes={byte_range[0]}-{byte_range[1]}'
            req=urllib.request.Request(url,headers=headers)
            timeout=min(15,max(.1,END-time.time()))
            try:r=self.opener.open(req,timeout=timeout)
            except urllib.error.HTTPError as e:r=e
            status=r.code;record={'url':clean_url(url),'status':status,'range':headers.get('Range'),'body_bytes_read':0}
            self.requests.append(record)
            if status in (301,302,303,307,308):
                loc=r.headers.get('Location');r.close()
                if not loc:raise Invalid('redirect without Location')
                next_url=urllib.parse.urljoin(url,loc)
                if urllib.parse.urlsplit(next_url).scheme!='https':raise Invalid('non HTTPS redirect')
                redirects.append(clean_url(next_url));url=next_url;continue
            try:
                if byte_range is not None:
                    if status==200:raise Unsupported('Range ignored: HTTP200 closed without body read')
                    if status!=206:raise Unsupported(f'archive Range HTTP{status}; body not read')
                    lo,hi=byte_range;n=hi-lo+1
                    cr=r.headers.get('Content-Range','')
                    m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',cr)
                    if not m or tuple(map(int,m.groups()))!=(lo,hi,expected_total):raise Invalid('incorrect Content-Range')
                    if r.headers.get('Content-Length')!=str(n):raise Invalid('Range requires exact Content-Length')
                    record['content_range']=cr;limit=n
                else:
                    if status!=200:raise Unsupported(f'metadata HTTP{status}; body not read')
                    limit=max_bytes
                    cl=r.headers.get('Content-Length')
                    if cl is not None and int(cl)>limit:raise Unsupported('metadata body larger than permitted bound')
                if r.headers.get('Content-Encoding','identity').lower() not in ('','identity'):raise Invalid('encoded body rejected')
                remaining=LIMIT['HTTP_total_body_bytes']-self.bytes
                if limit>remaining:raise Unsupported('requested metadata exceeds remaining HTTP byte budget')
                out=bytearray()
                while len(out)<limit:
                    guard();take=min(65536,limit-len(out),LIMIT['HTTP_total_body_bytes']-self.bytes)
                    if take<=0:raise Invalid('HTTP budget exhausted')
                    b=r.read(take);self.bytes+=len(b);record['body_bytes_read']+=len(b)
                    if not b:break
                    out.extend(b)
                if byte_range is not None and len(out)!=limit:raise Invalid('truncated Range body')
                if byte_range is None and len(out)==limit:
                    # Metadata JSON endpoint, not an archive; no unbounded EOF read.
                    if r.headers.get('Content-Length')!=str(limit):raise Unsupported('metadata reached bound without proven EOF')
                record['body_sha256']=sha(out);record['redirects']=redirects
                return bytes(out)
            finally:r.close()
        raise Unsupported('redirect count exceeded; no retry')

def eocd(tail,tail_start,total):
    pos=tail.rfind(b'PK\x05\x06')
    while pos>=0:
        if pos+22<=len(tail):
            fields=struct.unpack_from('<4s4H2IH',tail,pos)
            if pos+22+fields[-1]==len(tail):break
        pos=tail.rfind(b'PK\x05\x06',0,pos)
    if pos<0:raise Unsupported('valid terminal EOCD not found')
    _,disk,cd_disk,ndisk,nall,cdsize,cdoff,comment=fields
    if disk!=0 or cd_disk!=0:raise Unsupported('multidisk ZIP')
    return {'eocd_offset':tail_start+pos,'entries':nall,'entries_this_disk':ndisk,'cd_size':cdsize,'cd_offset':cdoff,'comment_bytes':comment,
            'zip64':nall==65535 or ndisk==65535 or cdsize==0xffffffff or cdoff==0xffffffff}

def zip64_info(locator,prefix):
    if len(locator)!=20 or locator[:4]!=b'PK\x06\x07':raise Invalid('ZIP64 locator missing')
    _,disk,offset,disks=struct.unpack('<4sIQI',locator)
    if disk!=0 or disks!=1:raise Unsupported('multidisk ZIP64')
    if len(prefix)!=56 or prefix[:4]!=b'PK\x06\x06':raise Invalid('ZIP64 EOCD missing')
    _,size,made,need,disk,cd_disk,ndisk,nall,cdsize,cdoff=struct.unpack('<4sQ2H2I4Q',prefix)
    if size<44 or disk!=0 or cd_disk!=0 or ndisk!=nall:raise Unsupported('invalid/multidisk ZIP64 record')
    return {'entries':nall,'entries_this_disk':ndisk,'cd_size':cdsize,'cd_offset':cdoff,'zip64_record_offset':offset,'zip64_record_size':size+12}

def member_values(extra,uncomp,comp,local,disk):
    blocks={};p=0
    while p<len(extra):
        if p+4>len(extra):raise Invalid('truncated central extra header')
        kind,n=struct.unpack_from('<HH',extra,p);p+=4
        if p+n>len(extra):raise Invalid('truncated central extra')
        if kind in blocks:raise Invalid('duplicate central extra field')
        blocks[kind]=extra[p:p+n];p+=n
    if uncomp==0xffffffff or comp==0xffffffff or local==0xffffffff or disk==65535:
        b=blocks.get(1,b'');q=0;out=[]
        for v,sentinel,width in [(uncomp,0xffffffff,8),(comp,0xffffffff,8),(local,0xffffffff,8),(disk,65535,4)]:
            if v==sentinel:
                if q+width>len(b):raise Invalid('missing ZIP64 member value')
                v=int.from_bytes(b[q:q+width],'little');q+=width
            out.append(v)
        return out
    return [uncomp,comp,local,disk]

def parse_cd(data,expected,archive_size):
    if expected>len(data)//46:raise Invalid('entry count exceeds minimum central headers')
    p=0;entries=[];seen=set();anomalies=[]
    for index in range(expected):
        if index%1024==0:guard()
        if p+46>len(data):raise Invalid('truncated central header')
        f=struct.unpack_from('<4s6H3I5H2I',data,p)
        if f[0]!=b'PK\x01\x02':raise Invalid('wrong central signature')
        _,made,need,flags,method,mtime,mdate,crc,comp,uncomp,nlen,xlen,clen,disk,internal,external,local=f
        end=p+46+nlen+xlen+clen
        if end>len(data):raise Invalid('truncated central entry')
        raw=data[p+46:p+46+nlen];extra=data[p+46+nlen:p+46+nlen+xlen]
        uncomp,comp,local,disk=member_values(extra,uncomp,comp,local,disk)
        if disk!=0:raise Unsupported('member on another disk')
        if local>=archive_size or comp>archive_size:raise Invalid('member archive bounds')
        encoding='utf-8' if flags&2048 else 'cp437'
        try:name=raw.decode(encoding)
        except UnicodeDecodeError:name=raw.decode(encoding,errors='replace');anomalies.append({'index':index,'kind':'filename_encoding'})
        bad=[]
        if '\x00' in name:bad.append('NUL')
        if name.startswith(('/', '\\')) or re.match(r'^[A-Za-z]:',name):bad.append('absolute_path')
        if '..' in re.split(r'[/\\]',name):bad.append('parent_segment')
        if '\\' in name:bad.append('backslash_path')
        if name in seen:bad.append('duplicate_name')
        seen.add(name)
        symlink=(made>>8)==3 and ((external>>16)&0o170000)==0o120000
        if symlink:bad.append('symlink')
        if bad:anomalies.append({'index':index,'name':name,'kind':bad})
        entries.append({'name':name,'compressed_bytes':comp,'uncompressed_bytes':uncomp,'method':method,
                        'encrypted':bool(flags&1),'directory':name.endswith('/'),'symlink':symlink,'CRC32':f'{crc:08x}','local_header_offset':local})
        p=end
    # The optional central-directory digital signature is metadata, not a member.
    if p!=len(data):
        if p+6>len(data) or data[p:p+4]!=b'PK\x05\x05' or p+6+int.from_bytes(data[p+4:p+6],'little')!=len(data):raise Invalid('unexpected trailing central bytes')
    return entries,anomalies

def summarize(entries,anomalies):
    groups={};ext=collections.Counter();methods=collections.Counter();selected=[]
    for e in entries:
        n=e['name'];parts=[v for v in n.split('/') if v]
        group='/'.join(parts[:min(3,len(parts))])
        # Files at root group together; three directory levels otherwise.
        if not e['directory']:group='/'.join(parts[:-1][:3]) or '(root)'
        g=groups.setdefault(group,{'members':0,'compressed_bytes':0,'uncompressed_bytes':0})
        g['members']+=1;g['compressed_bytes']+=e['compressed_bytes'];g['uncompressed_bytes']+=e['uncompressed_bytes']
        suffix=Path(n).suffix.lower() or '(none)';ext[suffix]+=1;methods[str(e['method'])]+=1
        if any(x in n.lower() for x in ('checkpoint','config','normaliz','stat','pusht','push_t')) or suffix in ('.ckpt','.pth','.pt','.yaml','.yml','.json','.npz'):
            if len(selected)<2000:selected.append(e)
    return {'member_count':len(entries),'file_count':sum(not e['directory'] for e in entries),
            'compressed_member_bytes':sum(e['compressed_bytes'] for e in entries),
            'uncompressed_member_bytes':sum(e['uncompressed_bytes'] for e in entries),
            'encrypted_members':sum(e['encrypted'] for e in entries),'methods':dict(methods),'extensions':dict(ext),
            'groups':groups,'selected_member_metadata':selected,'selected_member_limit':2000,
            'selected_metadata_is_complete':len(selected)<2000,'path_anomaly_count':len(anomalies),'path_anomalies':anomalies[:100],
            'sample_names':[e['name'] for e in entries[:40]],
            'limitation':'Central-directory declarations only; no CRC/payload/integrity/decompression or loader requirement validation.'}

def probe(http,url,size):
    if size<22:raise Invalid('archive too small')
    start=max(0,size-65557);tail=http.fetch(url,byte_range=(start,size-1),expected_total=size)
    meta=eocd(tail,start,size)
    def section(a,n):
        if a<0 or n<0 or a+n>size:raise Invalid('metadata bounds')
        if a>=start and a+n<=size:return tail[a-start:a-start+n]
        return http.fetch(url,byte_range=(a,a+n-1),expected_total=size)
    if meta['zip64']:
        loc=section(meta['eocd_offset']-20,20)
        if len(loc)!=20 or loc[:4]!=b'PK\x06\x07':raise Invalid('ZIP64 locator missing')
        off=struct.unpack('<4sIQI',loc)[2];z=zip64_info(loc,section(off,56))
        if off+z['zip64_record_size']!=meta['eocd_offset']-20:raise Invalid('ZIP64 record end mismatch')
        meta.update(z)
    if meta['entries']!=meta['entries_this_disk']:raise Unsupported('split central directory')
    if meta['cd_offset']+meta['cd_size']>meta.get('zip64_record_offset',meta['eocd_offset']):raise Invalid('central directory overlaps EOCD')
    if meta['cd_size']>LIMIT['HTTP_total_body_bytes']-http.bytes:raise Unsupported('central directory exceeds remaining body cap; no full download fallback')
    if meta['cd_size']==0:cd=b''
    else:cd=section(meta['cd_offset'],meta['cd_size'])
    entries,anomalies=parse_cd(cd,meta['entries'],size)
    return {'zip':meta,'archive_bytes':size,'tail_sha256':sha(tail),'central_directory_sha256':sha(cd),**summarize(entries,anomalies)}

def selftest():
    tests=[]
    n=b'folder/config.yaml';extra=b''
    header=struct.pack('<4s6H3I5H2I',b'PK\x01\x02',20,20,0,0,0,0,0,3,3,len(n),0,0,0,0,0,0)
    cd=header+n;e,a=parse_cd(cd,1,1000)
    assert e[0]['name']==n.decode() and e[0]['uncompressed_bytes']==3 and not a;tests.append('ordinary central record')
    trailer=struct.pack('<4s4H2IH',b'PK\x05\x06',0,0,1,1,len(cd),100,0)
    m=eocd(trailer,200,222);assert m['eocd_offset']==200 and not m['zip64'];tests.append('EOCD')
    zextra=struct.pack('<HHQQQI',1,28,2**33,2**32+4,1024,0)
    assert member_values(zextra,0xffffffff,0xffffffff,0xffffffff,65535)==[2**33,2**32+4,1024,0];tests.append('ZIP64 per member')
    loc=struct.pack('<4sIQI',b'PK\x06\x07',0,123,1)
    z=struct.pack('<4sQ2H2I4Q',b'PK\x06\x06',44,45,45,0,0,70000,70000,12345,321)
    assert zip64_info(loc,z)['entries']==70000;tests.append('ZIP64 EOCD')
    for malformed in (b'',cd[:-1],b'XXXX'+cd[4:]):
        try:parse_cd(malformed,1,1000)
        except (Invalid,Unsupported):tests.append('truncation/signature rejected')
        else:raise AssertionError('malformed accepted')
    return tests

def main():
    guard();os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (HERE/'ATTEMPT.json').open('x') as f:json.dump({'attempt':1,'started_utc':time.time(),'registered_start_utc':SPEC['start_epoch'],'deadline_utc':END,'source_sha256':sha(Path(__file__).read_bytes())},f)
    http=HTTP();out={'status':'started','registered_start_utc':SPEC['start_epoch'],'deadline_utc':END,'source_sha256':sha(Path(__file__).read_bytes()),'archives':[],'selftests':selftest()}
    try:
        repo=SPEC['target_repo'];rev=SPEC['repo_revision']
        url=f'https://huggingface.co/api/datasets/{repo}/revision/{rev}?blobs=true'
        body=http.fetch(url,max_bytes=1024**2);metadata=json.loads(body)
        if metadata.get('sha')!=rev:raise Invalid('HF revision mismatch')
        out['HF_metadata']={'repo':repo,'revision':rev,'metadata_sha256':sha(body),'private':metadata.get('private'),'gated':metadata.get('gated')}
        siblings={r['rfilename']:r for r in metadata.get('siblings',[])}
        for path in SPEC['paths']:
            guard();r=siblings.get(path)
            if r is None:raise Invalid(f'named archive absent from pinned metadata: {path}')
            size=r.get('size') or (r.get('lfs') or {}).get('size')
            if not isinstance(size,int) or size<=0:raise Invalid('archive size unavailable')
            record={'path':path,'HF_file_metadata':r,'status':'started'}
            out['archives'].append(record)
            asset=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}'
            try:record.update(probe(http,asset,size),status='metadata_parsed')
            except Unsupported as exc:record.update(status='unsupported_metadata_read',error=str(exc))
            except Exception as exc:record.update(status='invalid_metadata',error=repr(exc));raise
        out['status']='completed_metadata_only' if all(r['status']=='metadata_parsed' for r in out['archives']) else 'partial_or_unsupported_metadata_only'
    except BaseException as exc:
        out.update(status='invalid_or_incomplete_metadata_only',error=repr(exc),traceback=traceback.format_exc())
    finally:
        out.update(completed_utc=time.time(),HTTP_body_bytes_read=http.bytes,HTTP_requests=http.requests,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            source_unchanged=sha(Path(__file__).read_bytes())==out['source_sha256'],
            GPU_calls=0,tensor_model_loads=0,full_asset_downloads=0,extracted_members=0)
        save('RESULT.json',out)
    print(json.dumps({'status':out['status'],'HTTP_body_bytes_read':http.bytes,'archives':[(x['path'],x['status']) for x in out['archives']]}))

if __name__=='__main__':main()
