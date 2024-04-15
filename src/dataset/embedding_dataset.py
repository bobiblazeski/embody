import os
import torch

class EmbeddingDataset(torch.utils.data.Dataset):
    
    def __init__(self, patch_root, emb_root, slice_sz):        
        self._load_patches(patch_root)  
        self._load_inputs(emb_root)

        self.slice_sz = slice_sz
        patch_sz = self.patch_data[0].size(-1)
        self.slice_ids = self.make_ids(patch_sz - slice_sz + 1)
        
    def _load_patches(self, root):        
        patch_dict, patch_data, mean = {}, [], 0
        for f in sorted(os.listdir(root)):
            if f.endswith('.pth'):
                key = f.replace('.pth', '')
                val = torch.load(root+f)[0].detach().cpu()
                patch_dict[key] = len(patch_dict)
                patch_data.append(val)
                mean += val
        self.patch_dict = patch_dict
        self.patch_data = torch.stack(patch_data)
        self.mean = self.patch_data.mean(dim=0)

    def _load_inputs(self, root):
        scales, embeddings, names = [], [], []       
        for f in sorted(os.listdir(root)):
            scale, embedding, name = torch.load(root+f).values()
            scales.append(scale.detach().cpu())
            embeddings.append(embedding.detach().cpu())
            assert name in self.patch_dict, f'{name} not in known patch names'
            names.append(name)
        self.scale = torch.stack(scales)
        self.embedding = torch.stack(embeddings)
        self.name = names
        

    def __len__(self):        
        return len(self.embedding) * len(self.slice_ids)

    def make_ids(self, n):         
        t = torch.arange(n)
        grid = torch.meshgrid(t, t, indexing='ij')
        return torch.stack(grid, dim=-1).reshape(-1, 2)
    
    def __getitem__(self, idx):
        emb_idx = idx % len(self.embedding)
        embedding = self.embedding[emb_idx]
        scale = self.scale[emb_idx]
        name = self.name[emb_idx]
        
        u, v = self.slice_ids[idx % len(self.slice_ids)]
        piece = self.mean[:, u:u+self.slice_sz, v:v+self.slice_sz]
        
        patch = self.patch_data[self.patch_dict[name]]
        patch = patch[:, u:u+self.slice_sz, v:v+self.slice_sz]
        patch = patch * scale[:, None, None]
        
        return {
            'piece': piece,
            'patch': patch,
            'embedding': embedding,
        }