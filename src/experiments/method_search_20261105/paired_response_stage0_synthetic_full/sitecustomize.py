"""Only this isolated synthetic harness installs this file-I/O audit hook."""
import atexit,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
reads=set();denied=[]
def audit(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    path=Path(os.fsdecode(args[0])).resolve()
    if path.suffix=='.pt':
        if not path.is_relative_to(ROOT):
            denied.append(str(path));raise PermissionError('Synthetic fixture forbids outside .pt access: '+str(path))
        reads.add(str(path))
sys.addaudithook(audit)
def finish():
    (ROOT/f'IO_AUDIT_{os.getpid()}.json').write_text(json.dumps({'pid':os.getpid(),'synthetic_tensor_paths':sorted(reads),'denied_external_tensor_paths':denied,'scope':'audit-hook .pt path events; source files permitted'},indent=2)+'\n')
atexit.register(finish)
