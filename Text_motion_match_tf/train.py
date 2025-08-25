# Trainer for CLIP-based contrastive model with motion encoder
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from os.path import join as pjoin
from torch.utils.data import DataLoader, random_split
from easydict import EasyDict as edict
from omegaconf import OmegaConf

from options.train_options import TrainTexMotMatchOptions
from modules import TextEncoder
from projected_unimodal_tf import ProjectedUnimodalTF

from trainers import ContrastiveTrainer
from lib.datasets.datasets import get_dataloader
from lib.models.mano import build_mano_aa
import numpy as np
import argparse
from pathlib import Path



def collate_fn_text_motion(batch):
    texts = [item['text'] for item in batch]
    cap_lens = [len(t.split()) for t in texts]
    motions = []

    for item in batch:
        lhand = item["x_lhand"] 
        rhand = item["x_rhand"] 
        x_objet = item["x_obj"]
        motion = torch.from_numpy(np.concatenate([lhand, rhand, x_objet], axis=1)).float()  
        motions.append(motion)

    m_lens = [item["nframes"] for item in batch]
    max_len = max(m.shape[0] for m in motions)
    padded_motions = torch.zeros(len(motions), max_len, motions[0].shape[1])

    for i, m in enumerate(motions):
        padded_motions[i, :m.shape[0]] = m

    return texts, torch.tensor(cap_lens), padded_motions, torch.tensor(m_lens)



if __name__ == '__main__':
    parser = TrainTexMotMatchOptions()
    opt = parser.parse()

    opt.device = torch.device("cuda:" + str(opt.gpu_id) if opt.gpu_id != -1 else "cpu")
    print(f"Using device: {opt.device}")

    torch.autograd.set_detect_anomaly(True)

    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)

    
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)
    opt.eval_dir = pjoin(opt.save_root, 'eval')

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    base_cfg = OmegaConf.load('../configs/config.yaml')

    
    dataset_name = getattr(opt, "dataset_name", None) or os.getenv("DATASET_NAME") or base_cfg.get("dataset_name")
    if not dataset_name:
        raise ValueError("Please pass --dataset_name or set it in configs/config.yaml")

    
    dataset_cfg_path = Path('../configs/dataset') / f'{dataset_name}.yaml'
    if not dataset_cfg_path.exists():
        raise FileNotFoundError(f"Missing dataset config: {dataset_cfg_path}")

    
    config = OmegaConf.merge(base_cfg, OmegaConf.create({'dataset_name': dataset_name}))
    config.dataset = OmegaConf.load(dataset_cfg_path)

    data_config = config.dataset
    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=data_config.flat_hand).to(opt.device)
    rhand_layer = build_mano_aa(is_rhand=True, flat_hand=data_config.flat_hand).to(opt.device)

    
    full_dataset = get_dataloader("Motion" + data_config.name, config, data_config, test=True, return_dataset=True)
    train_set, val_set = random_split(full_dataset, [int(0.9 * len(full_dataset)), len(full_dataset) - int(0.9 * len(full_dataset))], generator=torch.Generator().manual_seed(config.seed))

   
    motion_encoder = ProjectedUnimodalTF(
    input_dim=opt.dim_pose,                  
    transformer_d_model=opt.dim_motion_latent,
    num_heads=opt.num_heads,
    num_layers=opt.num_layers,
    dropout=opt.dropout                      
        ).to(opt.device)


    text_encoder = TextEncoder(device=opt.device, latent_dim=opt.dim_motion_latent)

    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                          num_workers=8, pin_memory=(opt.gpu_id != -1), collate_fn=collate_fn_text_motion)

    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, drop_last=False,
                        num_workers=4, pin_memory=(opt.gpu_id != -1), collate_fn=collate_fn_text_motion)


    # Trainer
    trainer = ContrastiveTrainer(opt, text_encoder, motion_encoder)
    trainer.train(train_loader, val_loader)
    trainer.test_saved_model(val_loader, model_path=os.path.join(opt.model_dir, "best_model.pth"))
