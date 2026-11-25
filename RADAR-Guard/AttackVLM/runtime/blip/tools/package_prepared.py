"""Package the handoff after both recorded-environment OOF exports pass."""
from pathlib import Path
import hashlib
import json
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    for scenario in ['natural','stress']:
        folder=ROOT/'artifacts'/scenario
        audit=json.loads((folder/'audit.json').read_text())
        if audit['sklearn']!='1.6.1' or audit['records']!=2000:
            raise SystemExit('Expected recorded-environment exports for '+scenario)
        if any(audit[k]!=0 for k in ['prediction_mismatches','fold_mismatches','routing_mismatches']):
            raise SystemExit('OOF mismatch for '+scenario)
        if audit['max_margin_error']>1e-8 or audit['max_risk_error']>1e-8:
            raise SystemExit('OOF numerical tolerance exceeded')
        model_names=['fold_%d.joblib'%i for i in range(1,6)]
        if audit.get('deployment_model_exported'):
            model_names.append('deployment.joblib')
        for name in model_names:
            if not (folder/name).is_file():raise SystemExit('Missing model '+str(folder/name))
    archive=ROOT.parent/('BLIP_Ready_Handoff_'+time.strftime('%Y%m%d_%H%M%S')+'.zip')
    entries=[]
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix in ['.pyc','.zip']:
                continue
            data=p.read_bytes();name=p.relative_to(ROOT).as_posix()
            z.writestr(name,data)
            entries.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        z.writestr('prepared_handoff_manifest.json',json.dumps(entries,indent=2))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise RuntimeError('ZIP integrity check failed')
    print('READY_HANDOFF_ZIP',archive)

if __name__=='__main__':main()
