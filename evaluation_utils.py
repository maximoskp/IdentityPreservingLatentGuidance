from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import numpy as np
import os
from train_utils import make_mixed_batch, full_to_partial_masking

def evaluate_iplg_convergence(
        transformer_model,
        val_loader,
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
        hguide_hunique_unique = 0
        # how many foreign-unique chords are preserved
        running_hguide_funique = 0
        hguide_funique = 0
        # with FOREIGN guidance
        # how many home-unique chords are preserved
        running_fguide_hunique = 0
        fguide_hunique_unique = 0
        # how many foreign-unique chords are preserved
        running_fguide_funique = 0
        fguide_funique = 0

        batch_num = 0
        print('validation')
        with tqdm(val_loader, unit='batch', position=tqdm_position) as tepoch:
            tepoch.set_description(f'Epoch {epoch}@{step}| val')
            for batch in tepoch:
                batch_num += 1
                melody_grid = batch["pianoroll"].to(device)
                harmony_gt = batch["harmony_ids"].to(device)
                home_guidance_embeddings = batch["latent"].to(device)
                mixed_batch_1 = make_mixed_batch(batch, "latent")
                mixed_batch_2 = make_mixed_batch(mixed_batch_1, "latent")
                if interpolate:
                    foreign_guidance_embeddings_1 = mixed_batch_1["latent"].to(device)
                    foreign_guidance_embeddings_2 = mixed_batch_2["latent"].to(device)
                    foreign_guidance = 0.5*foreign_guidance_embeddings_1 + 0.5*foreign_guidance_embeddings_2
                elif extrapolate:
                    foreign_guidance_embeddings = mixed_batch_1["latent"].to(device)
                    foreign_guidance = 10*foreign_guidance_embeddings
                else:
                    foreign_guidance = mixed_batch_1["latent"].to(device)
                    foreing_ids = mixed_batch_1['harmony_ids'].to(device)
                    foreign_unique = 
                    home_unique = 

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
        '4_home_logits_loss':  val_home_logits_loss,
        '5_foreign_logits_loss':  val_foreign_logits_loss,
        '6_no2home_logits_loss': val_no2home_logits_loss,
        '7_home_acc': val_home_acc,
        '8_foreign_acc': val_foreign_acc,
        '9_no2home_acc': val_no2home_acc
    }
#  end evaluate_iplg_convergence