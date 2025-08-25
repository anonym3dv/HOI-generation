import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from collections import OrderedDict
from tqdm import tqdm
from modules import ContrastiveLoss

from utils.utils import print_current_loss_decomp
from utils.metrics import (
    calculate_top_k,
    euclidean_distance_matrix
)


class ContrastiveTrainer:
    def __init__(self, args, text_encoder, motion_encoder):
        self.opt = args
        self.device = args.device

        self.text_encoder = text_encoder.to(self.device)
        self.motion_encoder = motion_encoder.to(self.device)

        self.optimizer_text = optim.Adam(self.text_encoder.parameters(), lr=args.lr)
        self.optimizer_motion = optim.Adam(self.motion_encoder.parameters(), lr=args.lr)

        self.criterion = ContrastiveLoss(args.negative_margin)

        self.best_val_loss = float('inf')

    def forward(self, batch):
        captions, cap_lens, motions, m_lens = batch

        motions = motions.to(self.device).float()
        idx = np.argsort(m_lens.data.tolist())[::-1].copy() 

        motions = motions[idx]
        m_lens = m_lens[idx]
        captions = [captions[i] for i in idx]

        motion_emb = self.motion_encoder(motions)
        text_emb =self.text_encoder.encode_text(captions)
        


        return text_emb, motion_emb


        

    def compute_loss(self, text_emb, motion_emb):
        
        loss = self.criterion(text_emb, motion_emb)
        return loss, {'loss': loss.item()}



    def update(self, loss):
        self.optimizer_text.zero_grad()
        self.optimizer_motion.zero_grad()

        loss.backward()

        clip_grad_norm_(self.text_encoder.parameters(), 0.5)
        clip_grad_norm_(self.motion_encoder.parameters(), 0.5)

        self.optimizer_text.step()
        self.optimizer_motion.step()

    def evaluate(self, val_loader):
        self.text_encoder.eval()
        self.motion_encoder.eval()

        total_loss = 0
        all_motion_embeddings, all_text_embeddings = [], []
        total_size = 0
        matching_score_sum, top_k_count = 0, 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                text_emb, motion_emb = self.forward(batch)
                loss, loss_dict = self.compute_loss(text_emb, motion_emb)

                total_loss += loss_dict['loss']
                total_size += 1

                text_np = text_emb.cpu().numpy()
                motion_np = motion_emb.cpu().numpy()

                all_text_embeddings.append(text_np)
                all_motion_embeddings.append(motion_np)

                dist_mat = euclidean_distance_matrix(text_np, motion_np)
                diagonal = np.diag(dist_mat)
                avg_pos = diagonal.mean()
                avg_all = dist_mat.mean()
                

                matching_score_sum += dist_mat.trace()
                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

        avg_loss = total_loss / total_size
        matching_score = matching_score_sum / (total_size * val_loader.batch_size)
        r_precision = top_k_count / (total_size * val_loader.batch_size)

        
        all_text_np = np.concatenate(all_text_embeddings, axis=0)
        all_motion_np = np.concatenate(all_motion_embeddings, axis=0)
        X = np.concatenate([all_text_np, all_motion_np], axis=0)
        labels = ['text'] * len(all_text_np) + ['motion'] * len(all_motion_np)

        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import os

        tsne = TSNE(n_components=2, random_state=0, init='pca', learning_rate='auto').fit_transform(X)

        plt.figure(figsize=(6, 5))
        plt.scatter(tsne[:len(all_text_np), 0], tsne[:len(all_text_np), 1], label='text', alpha=0.6)
        plt.scatter(tsne[len(all_text_np):, 0], tsne[len(all_text_np):, 1], label='motion', alpha=0.6)
        plt.legend()
        plt.title("t-SNE Texts embeddings  / motions")
        plt.tight_layout()

        os.makedirs("tsne_figures", exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"tsne_figures/tsne_{timestamp}.png")
        plt.close()
        

        text_all = torch.tensor(np.concatenate(all_text_embeddings, axis=0)).to(self.device)
        motion_all = torch.tensor(np.concatenate(all_motion_embeddings, axis=0)).to(self.device)

        sim_matrix = text_all @ motion_all.T  # (N, N)
        pos_scores = sim_matrix.diag().cpu().numpy()
        avg_pos_score = pos_scores.mean()
        avg_total_score = sim_matrix.cpu().numpy().mean()
        print(f"[SIM] Pos avg: {avg_pos_score:.4f} | Total avg: {avg_total_score:.4f}")

        return avg_loss, matching_score, r_precision


    def train(self, train_loader, val_loader):
        epoch = 0
        iters_per_epoch = len(train_loader)
        total_iters = self.opt.max_epoch * iters_per_epoch

        for epoch in range(self.opt.max_epoch):
            self.text_encoder.train()
            self.motion_encoder.train()

            logs = OrderedDict()
            for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                text_emb, motion_emb = self.forward(batch)
                loss, loss_dict = self.compute_loss(text_emb, motion_emb)

                self.update(loss)

                for k, v in loss_dict.items():
                    logs[k] = logs.get(k, 0) + v

                if (i + 1) % self.opt.log_every == 0:
                    avg_logs = {k: v / self.opt.log_every for k, v in logs.items()}
                    print_current_loss_decomp(time.time(), epoch * iters_per_epoch + i, total_iters, avg_logs, epoch, i)
                    logs.clear()
            self.opt.cur_epoch = epoch
            val_loss, match_score, r_prec = self.evaluate(val_loader)


            
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Matching Score: {match_score:.4f}")
            print("R-Precision:", " ".join([f"(top {i+1}): {v:.4f}" for i, v in enumerate(r_prec)]))


            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save(os.path.join(self.opt.model_dir, "best_model.pth"))
                print("[+] Best model saved")

    def save(self, path):
        torch.save({
            'text_encoder': self.text_encoder.state_dict(),
            'motion_encoder': self.motion_encoder.state_dict(),
            'opt_text': self.optimizer_text.state_dict(),
            'opt_motion': self.optimizer_motion.state_dict()
        }, path)

    def test_saved_model(self, val_loader, model_path):
        print(f"[INFO] Charge model from {model_path}")
        self.load(model_path)
        self.text_encoder.eval()
        self.motion_encoder.eval()

        print("[INFO] Évaluation du modèle chargé...")
        val_loss, match_score, r_prec  = self.evaluate(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Matching Score: {match_score:.4f}")
        print("R-Precision:", " ".join([f"(top {i+1}): {v:.4f}" for i, v in enumerate(r_prec)]))



    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.text_encoder.load_state_dict(checkpoint['text_encoder'])
        self.motion_encoder.load_state_dict(checkpoint['motion_encoder'])
        self.optimizer_text.load_state_dict(checkpoint['opt_text'])
        self.optimizer_motion.load_state_dict(checkpoint['opt_motion'])
