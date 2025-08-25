import torch
import os
import numpy as np
from modules import TextEncoder

from projected_unimodal_tf import ProjectedUnimodalTF
def build_models(opt):
    
    text_enc = TextEncoder(device=opt.device, latent_dim=opt.dim_motion_latent)
    
    motion_enc = ProjectedUnimodalTF(
    input_dim=opt.dim_pose,
    transformer_d_model=opt.dim_motion_latent,
    num_heads=4,
    num_layers=6,  
    dropout=0.1
)

    
    checkpoint_path = os.path.join(opt.model_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=opt.device)
        text_enc.load_state_dict(checkpoint['text_encoder'])
        motion_enc.load_state_dict(checkpoint['motion_encoder'])
        print("Checkpoint from", checkpoint_path)
    else:
        print("checkpoint not found", checkpoint_path)

    return text_enc, motion_enc


class EvaluatorModelWrapper:
    def __init__(self, opt, text_encoder, motion_encoder):
        self.opt = opt
        self.device = opt.device

        self.text_encoder = text_encoder.to(self.device).eval()
        self.motion_encoder = motion_encoder.to(self.device).eval()

    def get_co_embeddings(self, captions, motions, m_lens):
        
        with torch.no_grad():
            motions = motions.to(self.device).float()

            
            align_idx = np.argsort(m_lens.cpu().numpy())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]
            captions = [captions[i] for i in align_idx]

            motion_emb = self.motion_encoder(motions)
            text_emb = self.text_encoder.encode_text(captions)

        return text_emb, motion_emb

    def get_motion_embeddings(self, motions, m_lens):
        
        with torch.no_grad():
            motions = motions.to(self.device).float()
            align_idx = np.argsort(m_lens.cpu().numpy())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            motion_emb = self.motion_encoder(motions)

        return motion_emb
if __name__ == "__main__":
    class DummyOpt:
        def __init__(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            

    opt = DummyOpt()

   
    text_encoder, motion_encoder = build_models(opt)
    evaluator = EvaluatorModelWrapper(opt, text_encoder, motion_encoder)

   
