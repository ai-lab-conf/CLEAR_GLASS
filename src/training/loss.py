import torch
import torch.nn.functional as F
try:
    import wandb
except ImportError:
    wandb = None
# ------------------------------
# Helper: Normalize embeddings.
# ------------------------------
def normalize(x, dim=-1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)
    

def pairwise_clip_loss(images, texts, tau=0.07):
    # images, texts: (M, N, L)
    M, N, L = images.shape
    # Normalize along last dim
    images_norm = normalize(images, dim=-1)
    texts_norm = normalize(texts, dim=-1)
    
    # Flatten the group dimension: shape (M*N, L)
    images_flat = images_norm.view(M * N, L)
    texts_flat = texts_norm.view(M * N, L)
    
    # Compute similarity matrix between all image and text embeddings
    sim_matrix = torch.matmul(images_flat, texts_flat.T)  # (M*N, M*N)
    sim_matrix = sim_matrix / tau  # scale by temperature
    
    # Group labels
    groups = torch.arange(M, device=images.device).unsqueeze(1).repeat(1, N).view(-1)  # (M*N)
    
    loss_img = 0.0
    loss_txt = 0.0
    for i in range(M * N):
        pos_mask = (groups == groups[i])
        # For image->text:
        logits = sim_matrix[i]
        pos_exp = torch.exp(logits[pos_mask]).sum()
        all_exp = torch.exp(logits).sum()
        loss_img += -torch.log(pos_exp / (all_exp + 1e-8) + 1e-8)
        
        # For text->image:
        logits_txt = sim_matrix.T[i]
        pos_exp_txt = torch.exp(logits_txt[pos_mask]).sum()
        all_exp_txt = torch.exp(logits_txt).sum()
        loss_txt += -torch.log(pos_exp_txt / (all_exp_txt + 1e-8) + 1e-8)
    
    loss_img = loss_img / (M * N)
    loss_txt = loss_txt / (M * N)
    return 0.5 * (loss_img + loss_txt)

def pairwise_inner_centroid_loss(images, texts):
    M, N, L = images.shape
    loss_inner = 0.0
    for g in range(M):
        imgs = images[g]   # (N, L)
        txts = texts[g]    # (N, L)
        combined = (imgs.unsqueeze(1) * txts.unsqueeze(0)).view(-1, L)
        combined_norm = normalize(combined, dim=-1)
        centroid = combined_norm.mean(dim=0, keepdim=True)
        centroid = normalize(centroid, dim=-1)
        cos_sim = F.cosine_similarity(combined_norm, centroid, dim=-1)
        loss_inner += (1 - cos_sim).mean()
    return loss_inner / M

def pairwise_loss(image_features_list, text_features_list, alpha=1.2233, tau=0.07):
    beta = 1- alpha
    images, texts = prepare_features(image_features_list, text_features_list)
    clip_loss = pairwise_clip_loss(images, texts, tau)
    inner_loss = pairwise_inner_centroid_loss(images, texts)
    return alpha * clip_loss + beta * inner_loss

def scaled_pairwise_inner_centroid_loss(images, texts, tau_prime=0.07):
    M, N, L = images.shape
    centroids = []
    combined_list = []
    for g in range(M):
        imgs = images[g]   # (N, L)
        txts = texts[g]    # (N, L)
        combined = (imgs.unsqueeze(1) * txts.unsqueeze(0)).view(-1, L)  # (N*N, L)
        combined = normalize(combined, dim=-1)
        centroid = normalize(combined.mean(dim=0, keepdim=True), dim=-1)  # (1, L)
        centroids.append(centroid)
        combined_list.append(combined)
    centroids = torch.cat(centroids, dim=0)
    
    loss_inner = 0.0
    total = 0
    for g in range(M):
        combined = combined_list[g]  # (N*N, L)
        sims = torch.matmul(combined, centroids.T) / tau_prime
        targets = torch.full((combined.shape[0],), g, device=images.device, dtype=torch.long)
        loss_inner += F.cross_entropy(sims, targets, reduction='sum')
        total += combined.shape[0]
    return loss_inner / total

def scaled_pairwise_loss(image_features_list, text_features_list, log_in_wandb, step, alpha=1.2233, tau=0.07, tau_prime=0.07):
    beta = 1 - alpha
    
    clip_loss = pairwise_clip_loss(image_features_list, text_features_list, tau)
    inner_loss = scaled_pairwise_inner_centroid_loss(image_features_list, text_features_list, tau_prime)

    if log_in_wandb:
        wandb.log({"outer_loss": clip_loss.item(), 'step': step})
        wandb.log({"inner_loss": inner_loss.item(), 'step': step})
    
    return alpha * clip_loss + beta * inner_loss

def centroid_based_clip_loss(images, texts, tau=0.07):
    M, N, L = images.shape
    image_centroids = normalize(images.mean(dim=1), dim=-1)  # (M, L)
    text_centroids = normalize(texts.mean(dim=1), dim=-1)    # (M, L)
    
    loss = 0.0
    for g in range(M):
        logits = torch.matmul(image_centroids[g].unsqueeze(0), text_centroids.T).squeeze(0) / tau
        target = torch.tensor([g], device=images.device)
        loss_img = F.cross_entropy(logits.unsqueeze(0), target)
        
        logits = torch.matmul(text_centroids[g].unsqueeze(0), image_centroids.T).squeeze(0) / tau
        loss_txt = F.cross_entropy(logits.unsqueeze(0), target)
        loss += (loss_img + loss_txt)
    return loss / (2 * M)

def centroid_based_inner_loss(images, texts, tau=0.07):
    M, N, L = images.shape
    image_centroids = normalize(images.mean(dim=1), dim=-1)  # (M, L)
    text_centroids = normalize(texts.mean(dim=1), dim=-1)    # (M, L)
    
    loss = 0.0
    total = 0
    images_norm = normalize(images, dim=-1)
    texts_norm = normalize(texts, dim=-1)
    
    for g in range(M):
        for i in range(N):
            logits = torch.matmul(images_norm[g, i].unsqueeze(0), image_centroids.T).squeeze(0) / tau
            target = torch.tensor([g], device=images.device)
            loss += F.cross_entropy(logits.unsqueeze(0), target)
            total += 1
            
            logits = torch.matmul(texts_norm[g, i].unsqueeze(0), text_centroids.T).squeeze(0) / tau
            loss += F.cross_entropy(logits.unsqueeze(0), target)
            total += 1
    return loss / total

def centroid_based_loss(image_features_list, text_features_list, alpha=1.2233, tau=0.07):
    beta = 1- alpha
    
    clip_loss = centroid_based_clip_loss(image_features_list, text_features_list, tau)
    inner_loss = centroid_based_inner_loss(image_features_list, text_features_list, tau)
    return alpha * clip_loss + beta * inner_loss

def scaled_centroid_based_inner_loss(images, texts, tau_prime=0.07):
    M, N, L = images.shape
    image_centroids = normalize(images.mean(dim=1), dim=-1)  # (M, L)
    text_centroids = normalize(texts.mean(dim=1), dim=-1)    # (M, L)
    
    loss = 0.0
    total = 0
    images_norm = normalize(images, dim=-1)
    texts_norm = normalize(texts, dim=-1)
    
    for g in range(M):
        for i in range(N):
            logits = torch.matmul(images_norm[g, i].unsqueeze(0), image_centroids.T).squeeze(0) / tau_prime
            target = torch.tensor([g], device=images.device)
            loss += F.cross_entropy(logits.unsqueeze(0), target)
            total += 1
            
            logits = torch.matmul(texts_norm[g, i].unsqueeze(0), text_centroids.T).squeeze(0) / tau_prime
            loss += F.cross_entropy(logits.unsqueeze(0), target)
            total += 1
    return loss / total

def scaled_centroid_based_loss(image_features_list, text_features_list, log_in_wandb, step, alpha=1.2233, tau=0.07, tau_prime=0.07):
    beta = 1- alpha
    
    clip_loss = centroid_based_clip_loss(image_features_list, text_features_list, tau)
    inner_loss = scaled_centroid_based_inner_loss(image_features_list, text_features_list, tau_prime)

    if log_in_wandb:
        wandb.log({"outer_loss": clip_loss.item(), 'step': step})
        wandb.log({"inner_loss": inner_loss.item(), 'step': step})
    return alpha * clip_loss + beta * inner_loss