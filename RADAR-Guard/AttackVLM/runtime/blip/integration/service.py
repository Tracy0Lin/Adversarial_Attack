"""New single-image adapter around the frozen formal BLIP operations.

Importing this module does not load BLIP, CLIP or torch. Instantiate the service
only when resources are available; this adapter has not had GPU end-to-end QA.
"""
from pathlib import Path
from threading import Lock
import uuid

class BlipDefenseService:
    def __init__(self, detector_path, device='cuda'):
        import torch
        import clip
        import numpy as np
        from lavis.models import load_model_and_preprocess
        from .detector import load_detector
        from . import formal_stage1, formal_features, formal_repair
        self.torch,self.clip,self.np = torch,clip,np
        self.stage1,self.features,self.repair = formal_stage1,formal_features,formal_repair
        self.detector = load_detector(detector_path)
        self.device = device
        self.lock = Lock()
        self.model,self.processors,_ = load_model_and_preprocess(
            name='blip_caption',model_type='base_coco',is_eval=True,device=device)
        self.model.eval()
        self.clip_model,_ = clip.load('ViT-B/32',device=device,jit=False)
        self.clip_model.eval()
        self.preset = dict(use_nucleus_sampling=False,num_beams=3,max_length=30,
                           min_length=10,top_p=.9,repetition_penalty=1.0,num_captions=1)

    def _captions(self, images):
        output=[]
        # Preserve the original Stage-1 maximum batch size; new images are never paired with unrelated requests.
        for start in range(0,len(images),16):
            batch=images[start:start+16]
            tensor=self.torch.stack([self.processors['eval'](image) for image in batch]).to(self.device)
            captions=self.model.generate({'image':tensor},**self.preset)
            if len(captions)!=len(batch):
                raise RuntimeError('BLIP caption count mismatch')
            for image,caption in zip(batch,captions):
                caption=str(caption).strip()
                for _ in range(3):
                    if caption:break
                    tensor=self.processors['eval'](image).unsqueeze(0).to(self.device)
                    retry=self.model.generate({'image':tensor},**self.preset)
                    caption=str(retry[0]).strip() if retry else ''
                if not caption:raise RuntimeError('Empty BLIP caption')
                output.append(caption)
        return output

    def defend(self, image_path, repaired_output_dir, image_id=None):
        from .detector import score_features
        with self.lock,self.torch.inference_mode():
            variants=self.stage1.make_variants(image_path)['variants']
            captions=self._captions([image for _,image in variants])
            unique=list(dict.fromkeys(captions))
            tokens=self.clip.tokenize(unique,truncate=True).to(self.device)
            embedding=self.clip_model.encode_text(tokens).float()
            embedding=self.torch.nn.functional.normalize(embedding,p=2,dim=-1)
            embedding=self.features.normalize_rows(embedding.cpu().numpy().astype(self.np.float64))
            lookup=dict(zip(unique,embedding))
            embedding=self.np.stack([lookup[c] for c in captions])
            feature_result=self.features.calculate_group_features(
                embedding,self.stage1.CAPTION_ORDER,self.stage1.CAPTION_ORDER)
            result=score_features(self.detector,feature_result['features'],image_id=image_id)
            result.update({'caption_model':'BLIP base_coco','features':feature_result['features'],
                           'caption_names':self.stage1.CAPTION_ORDER,'captions':captions,
                           'base_caption':captions[0],'final_caption':None,'repaired_image_path':None})
            if result['action']=='pass':
                result['final_caption']=captions[0]
            elif result['action']=='repair':
                image=self.repair.load_input(image_path)
                repaired=self.repair.apply_bilateral(image,{'diameter':7,'sigma_color':50.,'sigma_space':7.})
                output=Path(repaired_output_dir)
                output.mkdir(parents=True,exist_ok=True)
                path=output/(uuid.uuid4().hex+'.png')
                repaired.save(path,format='PNG')
                result['repaired_image_path']=str(path.resolve())
                result['final_caption']=self._captions([repaired])[0]
            return result
