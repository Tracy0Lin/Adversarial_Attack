"""New wrapper invoking the three unchanged formal AttackVLM scripts.

This command loads models and runs a real attack. Do not run it while GIT
occupies the GPU. It does not start automatically during import or packaging.
"""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-image',type=Path,required=True)
    parser.add_argument('--target-image',type=Path,required=True)
    parser.add_argument('--target-caption',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source,target=args.source_image.resolve(),args.target_image.resolve()
    if not source.is_file() or not target.is_file():raise SystemExit('Input image missing')
    if not args.target_caption.strip() or '\n' in args.target_caption or '\r' in args.target_caption:
        raise SystemExit('Provide exactly one target caption')
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    clean=output/'input/source/class0';goal=output/'input/target/class0'
    clean.mkdir(parents=True);goal.mkdir(parents=True)
    shutil.copyfile(source,clean/('0001'+source.suffix.lower()))
    shutil.copyfile(target,goal/('0001'+target.suffix.lower()))
    (output/'target.txt').write_text(args.target_caption.strip()+'\n',encoding='utf-8')
    scripts=ROOT/'original/project/work_code'
    mfii=output/'mfii';mftt=output/'mftt';captions=output/'mfii_captions.txt'
    commands=[
        [sys.executable,str(scripts/'mf_ii_transfer.py'),'--source_dir',str(clean.parent),
         '--target_dir',str(goal.parent),'--output',str(mfii),'--model_name','blip_caption',
         '--model_type','base_coco','--batch_size','1','--num_workers','0','--num_samples','1',
         '--steps','100','--epsilon','8','--alpha','1'],
        [sys.executable,str(scripts/'generate_captions.py'),'--img_path',str(mfii),
         '--output_path',str(captions),'--model_name','blip_caption','--model_type','base_coco',
         '--batch_size','1','--num_samples','1','--num_workers','0'],
        [sys.executable,str(scripts/'mf_tt_query.py'),'--model_name','blip_caption',
         '--model_type','base_coco','--batch_size','1','--num_samples','1','--input_res','384',
         '--alpha','1','--epsilon','8','--steps','8','--output',str(output/'mftt_captions'),
         '--image_output',str(mftt),'--data_path',str(mfii),'--text_path',str(captions),
         '--clean_dir',str(clean.parent),'--target_text_path',str(output/'target.txt'),
         '--num_workers','0','--delta','zero','--num_query','100','--num_sub_query','50','--sigma','8'],
    ]
    (output/'commands.json').write_text(json.dumps({'created_at':datetime.now(timezone.utc).isoformat(),
                                                   'commands':commands},indent=2))
    for name,command in zip(['mfii','caption','mftt'],commands):
        with (output/(name+'.log')).open('w') as log:
            subprocess.run(command,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,check=True)
    expected=mftt/'class0/0001.png'
    if not expected.is_file():raise RuntimeError('Final attack PNG was not generated')
    print(json.dumps({'status':'done','mfii_image':str(mfii/'class0/0001.png'),
                      'mftt_image':str(expected),'caption_file':str(output/'mftt_captions.txt')}))

if __name__=='__main__':main()
