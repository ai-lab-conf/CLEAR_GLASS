"""
This file will be used for the training of the model
"""
import json
import logging
import math
import os
import time
from typing import List, Dict, Optional, Tuple, Union, Any
from contextlib import suppress

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from open_clip import ClipLoss

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import ClipLoss

from training.distributed import is_master
from training.loss import scaled_pairwise_loss, pairwise_loss, centroid_based_loss, scaled_centroid_based_loss


# from distributed import is_master
# from loss import scaled_pairwise_loss, pairwise_loss, centroid_based_loss, scaled_centroid_based_loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress

    model.train()
    clip_loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size
    )

    data[f"train_{args.dataset_type}"].set_epoch(epoch)
    dataloader = data[f"train_{args.dataset_type}"].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        # batch now contains lists of groups of images and captions
        batch_images, batch_captions = batch
        
        if len(batch_captions) == 0:
            continue
        
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            total_loss = 0
            
            # batch_size = len(batch_images)  # number of groups in the batch #TODO: a hyperparameter
            
            # Process each group in the batch
            image_features_list = []
            text_features_list = []
            logit_scale_list = []
            
            for group_idx in range(args.batch_size):
                # Get current group of images and captions
                start_idx = group_idx*args.group_size
                end_idx = start_idx + args.group_size
                group_images = batch_images[start_idx:end_idx]
                group_captions = batch_captions[start_idx:end_idx]
                
                # Move group data to device
                #TODO: handle hugging face models
                if isinstance(group_images, list) and len(group_images) > 0 and 'pixel_values' in group_images[0]:
                    pixel_tensors = [item['pixel_values'] for item in group_images]
                    if len(pixel_tensors[0].shape) == 4 and pixel_tensors[0].shape[0] == 1:
                        # If pixel_values already have a batch dimension of 1, remove it and then stack
                        pixel_tensors = [tensor.squeeze(0) for tensor in pixel_tensors]  # Remove the extra batch dim
    
                    group_images = torch.stack(pixel_tensors)
           
                if isinstance(group_captions, list):
                    # Find maximum length
                    max_len = max(len(cap) for cap in group_captions)
                    
                    # Pad all sequences to max_len with your tokenizer's pad_token_id
                    pad_token_id = 49407  # From your tokenizer.json
                    padded_captions = [cap + [pad_token_id] * (max_len - len(cap)) for cap in group_captions]
                    group_captions = torch.tensor(padded_captions)
               
                group_images = group_images.to(device=device, non_blocking=True)
                group_captions = group_captions.to(device=device, non_blocking=True)
                
                if hasattr(model, 'get_image_features'):
                    # HuggingFace implementation
                    image_features = model.get_image_features(group_images)
                    text_features = model.get_text_features(group_captions)
                    # Get the logit scale from the model
                    logit_scale = model.logit_scale.exp()
                else:
                    # Get features for current group
                    image_features, text_features, logit_scale = model(group_images, group_captions)
                
                # Store features
                image_features_list.append(image_features)
                text_features_list.append(text_features)
                logit_scale_list.append(logit_scale)
            image_features_list = torch.stack(image_features_list)
            text_features_list = torch.stack(text_features_list)
            logit_scale_list = torch.stack(logit_scale_list)
            # Calculate loss based on the specified loss function
            if args.loss == 1:
                total_loss = scaled_pairwise_loss(image_features_list, text_features_list,args.wandb, step, tau=args.tau, alpha=args.alpha)
            elif args.loss == 2:
                total_loss = scaled_centroid_based_loss(image_features_list, text_features_list, args.wandb, step,tau=args.tau, alpha=args.alpha)
            else:
                
                batch_size, group_size, feature_dim = image_features_list.shape
                image_features_list = image_features_list.view(-1, feature_dim)  # [batch_size*group_size, feature_dim]
                text_features_list = text_features_list.view(-1, feature_dim)    # [batch_size*group_size, feature_dim]
                logit_scale = logit_scale_list.mean()
                
                total_loss = clip_loss(image_features_list, text_features_list, logit_scale)


        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.norm_gradient_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.norm_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        
        # Logging
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            num_samples = batch_count * args.batch_size * args.world_size * args.group_size  # Updated to account for group size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            loss_m.update(total_loss.item(), args.batch_size)
            logit_scale_scalar = logit_scale_list[0].item()  # Use first group's logit scale for logging
            
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            # Log data to tensorboard/wandb
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": args.batch_size*args.world_size / batch_time_m.val,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # Reset batch stats
            batch_time_m.reset()
            data_time_m.reset()


def evaluate(model, data, epoch, args,dataset_type: Optional[str] = None, eval_type: Optional[str] = None, tb_writer=None):
    print("Starting evaluation function")
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    
    # Unwrap model if needed
    if hasattr(model, 'module'):
        base_model = model.module
        print("Unwrapped once - model type:", type(base_model))
        if hasattr(base_model, 'module'):  # In case of nested wrapping
            base_model = base_model.module
            print("Unwrapped twice - model type:", type(base_model))
    else:
        base_model = model
        print("No unwrapping needed - model type:", type(base_model))
   
    base_model.eval()

    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress

    data_key = f"val_{dataset_type}"

    if data_key in data:
        data[data_key].set_epoch(epoch)
        dataloader = data[data_key].dataloader
        samples_per_val = getattr(dataloader, 'num_samples', len(dataloader.dataset))
        num_samples = 0
        
        scores = []
        tqdm_loader = tqdm(dataloader, desc="Computing retrieval scores")

        with torch.no_grad():
            for batch_num, batch in enumerate(tqdm_loader):
                # Handle EvaluationDataset format
                batch_images, batch_captions = batch
                
                # Skip empty batches
                if len(batch_images) == 0 or len(batch_captions) == 0:
                    logging.info(f"Skipping batch: images={len(batch_images)}, captions={len(batch_captions)}")
                    continue
                
                image_options = []
                for i, image_group in enumerate(batch_images):
                    # Handle image input
                    if isinstance(image_group, torch.Tensor):
                        image_group = image_group.unsqueeze(0)  # Add batch dimension
                    elif hasattr(image_group, 'pixel_values'):
                        # Handle BatchFeature from image processor
                        image_group = torch.tensor(image_group.pixel_values)
                    
                    image_group = image_group.to(device)
                    if hasattr(base_model, 'encode_image'):
                        # open_clip implementation
                        image_embeddings = base_model.encode_image(image_group)
                    elif hasattr(base_model, 'get_image_features'):
                        # HuggingFace implementation
                        image_features = base_model.get_image_features(image_group)
                        image_embeddings = image_features.detach()
                    else:
                        print(f"Model doesn't have encode_text or get_text_features method: {type(base_model)}")
                        continue
                   
                    image_embeddings = image_embeddings.cpu().numpy()
                    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
                    image_options.append(np.expand_dims(image_embeddings, axis=1))
                    
                caption_options = []
                for i, caption_group in enumerate(batch_captions):
                    # Handle text input
                    if isinstance(caption_group, torch.Tensor):
                        caption_group = caption_group.unsqueeze(0)  # Add batch dimension
                    if isinstance(caption_group, list):
                        caption_group = torch.tensor(caption_group)
                        caption_group = caption_group.unsqueeze(0)

                    
                    
                    caption_group = caption_group.to(device)
                    
                    # Handle different model types
                    if hasattr(base_model, 'encode_text'):
                        # open_clip implementation
                        caption_embeddings = base_model.encode_text(caption_group)
                    elif hasattr(base_model, 'get_text_features'):
                        # HuggingFace implementation
                        text_features = base_model.get_text_features(caption_group)
                        caption_embeddings = text_features.detach()
                    else:
                        print(f"Model doesn't have encode_text or get_text_features method: {type(base_model)}")
                        continue
                        
                    caption_embeddings = caption_embeddings.cpu().numpy()
                    caption_embeddings = caption_embeddings / np.linalg.norm(caption_embeddings, axis=1, keepdims=True)
                    caption_options.append(np.expand_dims(caption_embeddings, axis=1))
                # Skip if we don't have enough options
                if len(image_options) == 0 or len(caption_options) == 0:
                    continue

                # Combine all options
                image_options = np.concatenate(image_options, axis=1)  # B x K x D
                caption_options = np.concatenate(caption_options, axis=1)  # B x K x D
                
                # Calculate similarity scores
                batch_scores = np.einsum("nkd,nld->nkl", image_options, caption_options)
                scores.append(batch_scores)
                
                num_samples += image_options.shape[0]

                tqdm_loader.update(1)
        
        if len(scores) == 0:
            print("No valid scores collected during evaluation")
            return metrics
            
        all_scores = np.concatenate(scores, axis=0)
        result_records = evaluate_scores(all_scores, True)
        
        metrics.update({
            **result_records[0],
            "epoch": epoch,
            "num_samples": num_samples
        })

    if not metrics:
        return metrics

    logging.info(f"Evaluation Epoch: {epoch} " + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()]))

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"test/{name}", val, epoch)

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"test/{eval_type}/{name}": val, 'epoch': epoch})

    return metrics

# Keep the original evaluate_scores function
def evaluate_scores(scores, anchor_only):
    """
    Parameters:
    - scores: N x NumImages x NumCaptions
    - anchor_only: whether to calculate the accuracy for t2i and i2t only with respect to the 
        anchor prediction
    Returns: dictionary of accuracies for t2i and i2t
    """
    scores_i2t = scores
    num_images = scores_i2t.shape[1]
    scores_t2i = np.transpose(scores, axes=[0, 2, 1])
    num_captions = scores_t2i.shape[1]
    assert (num_images <= num_captions)
    # truncate tensor so that captions with no image associated do not contribute to the text-to-image accuracy
    scores_t2i = scores_t2i[:, :num_images, :]

    if not anchor_only:
        preds_per_image = np.argmax(scores_i2t, axis=-1)
        answer_per_image = np.tile(
            np.arange(num_images), (preds_per_image.shape[0], 1))

        preds_per_text = np.argmax(scores_t2i, axis=-1)
        answer_per_text = np.tile(
            np.arange(num_images), (preds_per_text.shape[0], 1))
        
    else:
        #it assumes that the correct one is always the first one. 
        preds_per_image = np.argmax(scores_i2t, axis=-1)[:,:1]
        answer_per_image = np.tile(
            0, (preds_per_image.shape[0], 1))

        preds_per_text = np.argmax(scores_t2i, axis=-1)[:,:1]
        answer_per_text = np.tile(
            0, (preds_per_text.shape[0], 1))
        
    i2t_correct_mask = (preds_per_image == answer_per_image)
    i2t_accuracy = i2t_correct_mask.mean()

    t2i_correct_mask = (preds_per_text == answer_per_text)
    t2i_accuracy = t2i_correct_mask.mean()
    return [{"image_to_text_R@1": i2t_accuracy,
             "text_to_image_R@1": t2i_accuracy}]