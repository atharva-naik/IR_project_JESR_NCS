import torch
import torch.nn as nn

# neural net based on concatenation of text and code embeddings.
class ConcatSim(nn.Module):
    def __init__(self, text_emb_size: int=768, code_emb_size: int=768):
        super(ConcatSim, self).__init__()
        self.linear = nn.Linear(text_emb_size+code_emb_size, 1)
        self.relu = nn.ReLU()
        
    def forward(self, text_emb: torch.Tensor, code_emb: torch.Tensor):
        return self.relu(
            self.linear(
                torch.cat(
                    text_emb, 
                    code_emb
                ), 
                axis=1
            )
        )
    
# neural net based on cosine similarity.
# class CosineSim