import os
import torch
from random import randint
from einops import rearrange

class DecodeDataset(torch.utils.data.Dataset):
    
    def __init__(self, patch_root, emb_file, n_embs=3, suffix='.pth', coarse=None, transform=lambda x: x):
        self.transform = transform
        self.n_embs = n_embs
        self._load_patches(patch_root, suffix)
        self._load_inputs(emb_file)        
        self.__no_elements__ = len(self.name)        
        self.coarse = torch.load(coarse) if coarse is not None else None
        
    def _load_patches(self, root, suffix):        
        patch_dict, patch_data = {}, []
        for f in sorted(os.listdir(root)):
            if f.endswith('.pth'):
                key = f.replace(suffix, '')                
                val = torch.load(root+f)[0].detach().cpu()
                patch_dict[key] = len(patch_dict)                
                patch_data.append(val)
        self.patch_dict = patch_dict
        self.patch_data = torch.stack(patch_data)        

    def _load_inputs(self, roots):
        scales, embeddings, names = [], [], []
        for root in roots:
            for f in sorted(os.listdir(root)):
                if f.endswith('.pth'):                
                    scale, embedding, name = torch.load(root+f).values()                
                    scales.append(scale.detach().cpu())
                    embeddings.append(torch.stack(embedding))
                    assert name in self.patch_dict, f'{name} not in known patch names'
                    names.append(name)                
        self.scale = torch.stack(scales)
        self.embedding = torch.stack(embeddings)
        self.name = names        
        

    def __len__(self):        
        return self.__no_elements__
    
    def __getitem__(self, idx):        
        
        emb_idx = idx % len(self.embedding)        
        scale = self.scale[emb_idx]
        name = self.name[emb_idx]
        embedding = self.embedding[emb_idx]
        embedding = embedding.index_select(0, 
            torch.randint(embedding.size(0), (self.n_embs,)))
        
        patch = self.patch_data[self.patch_dict[name]] 
        patch = self.transform(patch)
        patch = patch * scale[:, None, None]
        
        res =  {            
            'patch': patch,
            'embedding': embedding,
            'idx': idx,
        }
        if self.coarse is not None:
           res['coarse'] = self.coarse[ idx]
        return res