import torch
import lib.processingTool as pt
import torch.nn.functional as F

# given a query and dataset embeddings, find the rank of a specific frame
def get_frame_rank(text, frame_idx, embeds, model, tokenizer):
  query = tokenizer(text).to(pt.device)

  with torch.no_grad(), torch.amp.autocast(pt.device):
    text_embeds = model.encode_text(query)

    distances = 1 - (F.normalize(text_embeds) @ F.normalize(embeds).T)
    sorted_indices = torch.argsort(distances)[0].tolist()
    frame_rank = sorted_indices.index(frame_idx)
    return frame_rank

def get_frame_rank_and_sorted_indices(text, frame_idx, embeds, model, tokenizer):
  query = tokenizer(text).to(pt.device)

  with torch.no_grad(), torch.amp.autocast(pt.device):
    text_embeds = model.encode_text(query)

    distances = 1 - (F.normalize(text_embeds) @ F.normalize(embeds).T)
    sorted_indices = torch.argsort(distances)[0].tolist()
    frame_rank = sorted_indices.index(frame_idx)
    return frame_rank, sorted_indices
