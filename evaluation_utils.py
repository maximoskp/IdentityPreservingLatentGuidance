from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import numpy as np
import os
from train_utils import make_mixed_batch, full_to_partial_masking
from torch.utils.data import DataLoader
from data_utils import latent_MH_collate_fn

batch_size = 4

def compute_unique_logit_activations(harmony_gt, foreign_ids, logits, threshold=0.95):
    """
    harmony_gt : [B, L] token ids in ground truth
    foreign_ids: [B, L] token ids in guiding harmony
    logits     : [B, seq_len, D] logits produced by the system
    threshold  : value of per-position survival

    Returns
    -------
    home_unique_logits_activations    : [B] sum of logit activations for ground truth-unique token ids
    foreign_unique_logits_activations : [B] sum of logit activations for guiding harmony-unique token ids
    """
    
    # B, D = logits.shape

    # Convert logits to probabilities
    probs = F.softmax(logits, dim=2)   # [B, seq_len, D]
    probs_sum = probs.sum(axis=1)
    B, D = probs_sum.shape

    # Create vocabulary masks
    home_mask = torch.zeros(B, D, dtype=torch.bool, device=logits.device)
    foreign_mask = torch.zeros(B, D, dtype=torch.bool, device=logits.device)

    home_mask.scatter_(1, harmony_gt, True)
    foreign_mask.scatter_(1, foreign_ids, True)

    # Unique tokens
    home_unique_mask = home_mask & (~foreign_mask)
    foreign_unique_mask = foreign_mask & (~home_mask)

    # Sum probabilities
    home_unique_sum = (probs_sum * home_unique_mask).sum(dim=1)
    foreign_unique_sum = (probs_sum * foreign_unique_mask).sum(dim=1)

    # Normalize by total probability mass (softmax sums to 1, but kept explicit)
    total_prob = probs_sum.sum(dim=1)

    home_unique_logits_activations = home_unique_sum / total_prob
    foreign_unique_logits_activations = foreign_unique_sum / total_prob

    # ---- position-wise confidence ----

    # Expand masks to sequence dimension
    home_unique_mask_exp = home_unique_mask.unsqueeze(1)      # [B, 1, D]
    foreign_unique_mask_exp = foreign_unique_mask.unsqueeze(1)

    # Mask probabilities
    home_probs = probs * home_unique_mask_exp                # [B, seq_len, D]
    foreign_probs = probs * foreign_unique_mask_exp

    # # Max probability among unique tokens at each position
    # home_max_probs, _ = home_probs.max(dim=2)                # [B, seq_len]
    # foreign_max_probs, _ = foreign_probs.max(dim=2)

    # # Check threshold
    # home_confident = (home_max_probs >= threshold)           # [B, seq_len]
    # foreign_confident = (foreign_max_probs >= threshold)

    # # Ratio over sequence length
    # home_confident_ratio = home_confident.float().mean(dim=1)     # [B]
    # foreign_confident_ratio = foreign_confident.float().mean(dim=1)

    # Check threshold
    home_confident = (home_probs >= threshold).sum()
    foreign_confident = (foreign_probs >= threshold).sum()

    # Ratio over sequence length
    # home_confident_ratio = home_confident.float()/(home_unique_mask.sum().float()*probs.shape[1])
    # foreign_confident_ratio = foreign_confident.float()/(foreign_unique_mask.sum().float()*probs.shape[1])
    # Ratio over batch size
    home_confident_ratio = home_confident.float()/probs.shape[0]
    foreign_confident_ratio = foreign_confident.float()/probs.shape[0]

    return (
        home_unique_logits_activations,
        foreign_unique_logits_activations,
        home_confident_ratio,
        foreign_confident_ratio
    )
# end compute_unique_logit_activations

def evaluate_iplg_convergence(
        transformer_model,
        home_dataset,
        foreign_dataset,
        logits_loss_fn,
        latent_loss_fn,
        mask_token_id,
        bar_token_id,
        device,
        interpolate=False,
        extrapolate=False
):
    tqdm_position = 0
    epoch = 0
    step = 0
    num_visible = 0

    with torch.no_grad():
        running_foreign_loss = 0
        val_foreign_loss = 0

        running_home_loss = 0
        val_home_loss = 0

        running_no2home_loss = 0
        val_no2home_loss = 0

        running_no2foreign_loss = 0
        val_no2foreign_loss = 0

        # logits losses
        running_foreign_logits_loss = 0
        val_foreign_logits_loss = 0
        running_home_logits_loss = 0
        val_home_logits_loss = 0
        running_no2home_logits_loss = 0
        val_no2home_logits_loss = 0

        # accuracies
        running_foreign_acc = 0
        val_foreign_acc = 0
        running_home_acc = 0
        val_home_acc = 0
        running_no2home_acc = 0
        val_no2home_acc = 0

        # logit activation of unique chords
        # with HOME guidance
        # how many home-unique chords are preserved
        running_hguide_hunique = 0
        hguide_hunique = 0
        # how many foreign-unique chords are preserved
        running_hguide_funique = 0
        hguide_funique = 0
        # how many home-unique chords are surviving 90% threshold
        running_confident_hguide_hunique = 0
        confident_hguide_hunique = 0
        # how many foreign-unique chords are surviving 90% threshold
        running_confident_hguide_funique = 0
        confident_hguide_funique = 0
        # with FOREIGN guidance
        # how many home-unique chords are preserved
        running_fguide_hunique = 0
        fguide_hunique = 0
        # how many foreign-unique chords are preserved
        running_fguide_funique = 0
        fguide_funique = 0
        # how many home-unique chords are surviving 90% threshold
        running_confident_fguide_hunique = 0
        confident_fguide_hunique = 0
        # how many foreign-unique chords are surviving 90% threshold
        running_confident_fguide_funique = 0
        confident_fguide_funique = 0

        batch_num = 0
        for i in range(30):
            home_loader = DataLoader(
                home_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=latent_MH_collate_fn
            )
            foreign_loader = DataLoader(
                foreign_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=latent_MH_collate_fn
            )
            with tqdm(home_loader, unit='batch', position=tqdm_position) as tepoch:
                tepoch.set_description(f'Epoch {epoch}@{step}| val')
                for batch in tepoch:
                    batch_num += 1
                    melody_grid = batch["pianoroll"].to(device)
                    harmony_gt = batch["harmony_ids"].to(device)
                    home_guidance_embeddings = batch["latent"].to(device)
                    foreign_batch = next(iter(foreign_loader))
                    mixed_batch_1 = make_mixed_batch(foreign_batch, "latent")
                    mixed_batch_2 = make_mixed_batch(foreign_batch, "latent")
                    if interpolate:
                        foreign_guidance_embeddings_1 = mixed_batch_1["latent"].to(device)
                        foreign_guidance_embeddings_2 = mixed_batch_2["latent"].to(device)
                        foreign_guidance = 0.5*foreign_guidance_embeddings_1 + 0.5*foreign_guidance_embeddings_2
                    elif extrapolate:
                        foreign_guidance_embeddings = mixed_batch_1["latent"].to(device)
                        foreign_guidance = 10*foreign_guidance_embeddings
                    else:
                        foreign_guidance = mixed_batch_1["latent"].to(device)
                        foreign_ids = mixed_batch_1['harmony_ids'].to(device)

                    harmony_input, harmony_target = full_to_partial_masking(
                        harmony_gt,
                        mask_token_id,
                        num_visible,
                        bar_token_id=bar_token_id
                    )

                    # Step 1: contrastive latent attraction validation
                    logits, hidden = transformer_model(
                        melody_grid.to(device),
                        harmony_input.to(device),
                        foreign_guidance.to(device),
                        return_hidden=True
                    )
                    foreign_guidance_loss = latent_loss_fn(foreign_guidance,hidden)

                    running_foreign_logits_loss += logits_loss_fn(logits.view(-1, logits.size(-1)), harmony_target.view(-1)).item()
                    val_foreign_logits_loss = running_foreign_logits_loss/batch_num

                    if not (interpolate or extrapolate):
                        tmp_home_unique, tmp_foreign_unique, tmp_home_confident, tmp_foreign_confident = compute_unique_logit_activations(
                            harmony_gt,
                            foreign_ids,
                            logits
                        )
                        running_fguide_hunique += tmp_home_unique.sum().item()
                        running_fguide_funique += tmp_foreign_unique.sum().item()
                        running_confident_fguide_hunique += tmp_home_confident.sum().item()
                        running_confident_fguide_funique += tmp_foreign_confident.sum().item()
                        fguide_hunique = running_fguide_hunique/batch_num
                        fguide_funique = running_fguide_funique/batch_num
                        confident_fguide_hunique = running_confident_fguide_hunique/batch_num
                        confident_fguide_funique = running_confident_fguide_funique/batch_num

                    # accuracy
                    predictions = logits.argmax(dim=-1)
                    # mask = torch.logical_and(harmony_target != harmony_input, harmony_target != -100)
                    mask = harmony_target != -100
                    running_foreign_acc += (predictions[mask] == harmony_target[mask]).sum().item()/mask.sum().item()
                    val_foreign_acc = running_foreign_acc/batch_num

                    # Step 2: home attraction validation
                    logits, hidden = transformer_model(
                        melody_grid.to(device),
                        harmony_input.to(device),
                        home_guidance_embeddings.to(device),
                        return_hidden=True
                    )
                    home_guidance_loss = latent_loss_fn(home_guidance_embeddings,hidden)

                    running_home_logits_loss += logits_loss_fn(logits.view(-1, logits.size(-1)), harmony_target.view(-1)).item()
                    val_home_logits_loss = running_home_logits_loss/batch_num

                    if not (interpolate or extrapolate):
                        tmp_home_unique, tmp_foreign_unique, tmp_home_confident, tmp_foreign_confident = compute_unique_logit_activations(
                            harmony_gt,
                            foreign_ids,
                            logits
                        )
                        running_hguide_hunique += tmp_home_unique.sum().item()
                        running_hguide_funique += tmp_foreign_unique.sum().item()
                        running_confident_hguide_hunique += tmp_home_confident.sum().item()
                        running_confident_hguide_funique += tmp_foreign_confident.sum().item()
                        hguide_hunique = running_hguide_hunique/batch_num
                        hguide_funique = running_hguide_funique/batch_num
                        confident_hguide_hunique = running_confident_hguide_hunique/batch_num
                        confident_hguide_funique = running_confident_hguide_funique/batch_num
                        
                    # accuracy
                    predictions = logits.argmax(dim=-1)
                    # mask = torch.logical_and(harmony_target != harmony_input, harmony_target != -100)
                    mask = harmony_target != -100
                    running_home_acc += (predictions[mask] == harmony_target[mask]).sum().item()/mask.sum().item()
                    val_home_acc = running_home_acc/batch_num

                    # partial losses
                    running_foreign_loss += foreign_guidance_loss.item()
                    val_foreign_loss = running_foreign_loss/batch_num
                    running_home_loss += home_guidance_loss.item()
                    val_home_loss = running_home_loss/batch_num

                    # Step 3: no attraction
                    logits, hidden = transformer_model(
                        melody_grid.to(device),
                        harmony_input.to(device),
                        None,
                        return_hidden=True
                    )
                    no_guidance_to_home_loss = latent_loss_fn(home_guidance_embeddings,hidden)

                    no_guidance_to_foreign_loss = latent_loss_fn(foreign_guidance,hidden)

                    running_no2home_logits_loss += logits_loss_fn(logits.view(-1, logits.size(-1)), harmony_target.view(-1)).item()
                    val_no2home_logits_loss = running_no2home_logits_loss/batch_num

                    # accuracy
                    predictions = logits.argmax(dim=-1)
                    # mask = torch.logical_and(harmony_target != harmony_input, harmony_target != -100)
                    mask = harmony_target != -100
                    running_no2home_acc += (predictions[mask] == harmony_target[mask]).sum().item()/mask.sum().item()
                    val_no2home_acc = running_no2home_acc/batch_num

                    # partial losses
                    running_no2home_loss += no_guidance_to_home_loss.item()
                    val_no2home_loss = running_no2home_loss/batch_num
                    running_no2foreign_loss += no_guidance_to_foreign_loss.item()
                    val_no2foreign_loss = running_no2foreign_loss/batch_num

                    loss = 0.25*home_guidance_loss + 0.25*foreign_guidance_loss + 0.25*no_guidance_to_home_loss + 0.25*no_guidance_to_foreign_loss
                    acc = 0.5*val_home_acc + 0.5*val_no2home_acc

                    tepoch.set_postfix(
                        loss=loss.item(),
                        floss=val_foreign_loss,
                        hloss=val_home_loss,
                        acc=acc
                    )
                # end for batch
            # end with tqdm
    # end with no grad
    return {
        '0_foreign_loss': val_foreign_loss,
        '1_home_loss': val_home_loss,
        '2_no2home_loss': val_no2home_loss,
        '3_no2foreign_loss': val_no2foreign_loss,
    },{
        '4_home_logits_loss':  val_home_logits_loss,
        '5_foreign_logits_loss':  val_foreign_logits_loss,
        '6_no2home_logits_loss': val_no2home_logits_loss,
        '7_home_acc': val_home_acc,
        '8_foreign_acc': val_foreign_acc,
        '9_no2home_acc': val_no2home_acc
    },{
        'fguide_hunique': fguide_hunique,
        'fguide_funique': fguide_funique,
        'hguide_hunique': hguide_hunique,
        'hguide_funique': hguide_funique,
    },{
        'confident_fguide_hunique': confident_fguide_hunique,
        'confident_fguide_funique': confident_fguide_funique,
        'confident_hguide_hunique': confident_hguide_hunique,
        'confident_hguide_funique': confident_hguide_funique,
    }
#  end evaluate_iplg_convergence