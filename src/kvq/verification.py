"""Live gate reports stay outside curated source documentation."""
from pathlib import Path


def write_progress(status, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    lines=['# Live verification progress','','| Gate | State | Evidence |','|---|---|---|']
    for gate,record in status.items():
        lines.append(f"| {gate} | {record['state']} | {record.get('detail','')} |")
    path=output/'progress.md';path.write_text('\n'.join(lines)+'\n')
    return path


def public_snapshot(value, *, home, project, data):
    """Replace workstation paths in a public copy; raw digests stay raw digests."""
    replacements={str(project):'${PROJECT_ROOT}',str(data):'${DATA_ROOT}',
                  str(Path(home)/'sf-kvquant'):'${RECONSTRUCTION_SNAPSHOT}',str(home):'${HOME}'}
    if isinstance(value,str):
        for old,new in sorted(replacements.items(),key=lambda item:len(item[0]),reverse=True):
            value=value.replace(old,new)
        return value
    if isinstance(value,list):return [public_snapshot(x,home=home,project=project,data=data) for x in value]
    if isinstance(value,dict):return {k:public_snapshot(v,home=home,project=project,data=data) for k,v in value.items()}
    return value
