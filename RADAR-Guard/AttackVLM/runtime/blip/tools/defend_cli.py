"""Run real BLIP defense on one image. This loads models; do not run during GIT jobs."""
from pathlib import Path
import argparse
import contextlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--scenario',choices=['natural','stress'],default='natural')
    parser.add_argument('--image-id',help='Set only to replay a recorded held-out OOF case.')
    parser.add_argument('--model',type=Path)
    parser.add_argument('--repaired-output',type=Path,default=ROOT/'outputs/repaired')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    model=args.model
    if model is None:
        folder=ROOT/'artifacts'/args.scenario
        if args.image_id is None:
            raise SystemExit('Formal replay requires --image-id. For a separately validated deployment model, pass --model explicitly.')
        else:
            splits=json.loads((folder/'fold_test_ids.json').read_text())
            matching=[fold for fold,ids in splits.items() if args.image_id in ids]
            if len(matching)!=1:raise SystemExit('image_id not present in this scenario')
            model=folder/('fold_%s.joblib'%matching[0])
    from integration.service import BlipDefenseService
    with contextlib.redirect_stdout(sys.stderr):
        service=BlipDefenseService(model,device=args.device)
        result=service.defend(args.image,args.repaired_output,image_id=args.image_id)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
