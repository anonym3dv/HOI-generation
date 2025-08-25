import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

import torch

def collate_pad(batch, fixed_len=150):
    texts, motions, lengths = zip(*batch)
    dim = motions[0].shape[1]

    padded = torch.zeros(len(motions), fixed_len, dim)
    for i, m in enumerate(motions):
        length = min(len(m), fixed_len)
        padded[i, :length] = torch.tensor(m[:length])  

    lengths = torch.tensor([min(l, fixed_len) for l in lengths])
    return list(texts), padded, lengths



def collate_gen(batch, fixed_len=150):
    texts, motions, lengths = zip(*batch)
    dim = motions[0].shape[1]

    padded = torch.zeros(len(motions), fixed_len, dim)
    for i, m in enumerate(motions):
        length = min(len(m), fixed_len)
        padded[i, :length] = torch.tensor(m[:length])

    lengths = torch.tensor([min(l, fixed_len) for l in lengths])
    return list(texts), padded, lengths

def collate_mm(batch, fixed_len=150):
    texts, motions, lengths = zip(*batch)
    B = len(motions)
    R, D = motions[0].shape[0], motions[0].shape[2] 

    padded = torch.zeros(B, R, fixed_len, D)
    for i in range(B):
        for j in range(R):
            T = motions[i].shape[1]
            L = min(T, fixed_len)
            padded[i, j, :L] = torch.tensor(motions[i][j, :L])

    lengths = torch.tensor([min(l, fixed_len) for l in lengths])
    return list(texts), padded, lengths


class GTDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.texts = []  
        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path): continue
            gt_path = os.path.join(folder_path, 'gt.npy')
            if os.path.exists(gt_path):
                self.data.append(np.load(gt_path))
                self.texts.append(folder.replace("_", " ").strip())  
        self.lengths = [len(x) for x in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.texts[idx], self.data[idx], self.lengths[idx]


class GenDataset(Dataset):
    def __init__(self, root_dir):
        self.texts = []
        self.gen_paths = []

        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            text_path = os.path.join(folder_path, 'text.txt')
            if not os.path.exists(text_path):
                continue

            with open(text_path, 'r') as f:
                text = f.read().strip()

            gen_files = sorted([f for f in os.listdir(folder_path) if f.startswith('gen_') and f.endswith('.npy')])
            if len(gen_files) == 0:
                continue

            gen_full_paths = [os.path.join(folder_path, f) for f in gen_files]

            self.texts.append(text)
            self.gen_paths.append(gen_full_paths)

    def __len__(self):
        return len(self.gen_paths)

    def __getitem__(self, idx):
        gen_file = random.choice(self.gen_paths[idx])
        motion = np.load(gen_file)
        length = len(motion)
        return self.texts[idx], motion, length

class MultiModalDataset(Dataset):
    def __init__(self, root_dir, mm_repeats=20):
        self.data = []
        self.mm_repeats = mm_repeats

        for folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path): continue

            text_path = os.path.join(folder_path, 'text.txt')
            if not os.path.exists(text_path): continue
            with open(text_path, 'r') as f:
                text = f.read().strip()

            gen_files = sorted([f for f in os.listdir(folder_path) if f.startswith('gen_')])
            gen_paths = [os.path.join(folder_path, f) for f in gen_files]
            if len(gen_paths) < mm_repeats:
                continue  

            gen_motions = [np.load(p) for p in gen_paths[:mm_repeats]]
            self.data.append((text, np.stack(gen_motions)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, motion_stack = self.data[idx]
        return text, motion_stack, motion_stack.shape[1] 
    

def get_all_loaders(root, batch_size=32, mm_repeats=5, fixed_len=150):
    gt_loader = DataLoader(GTDataset(root), batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_pad(x, fixed_len))
    gen_loader = DataLoader(GenDataset(root), batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_gen(x, fixed_len))
    mm_loader = DataLoader(MultiModalDataset(root, mm_repeats=mm_repeats), batch_size=1, shuffle=False, collate_fn=lambda x: collate_mm(x, fixed_len))
    return gt_loader, gen_loader, mm_loader

