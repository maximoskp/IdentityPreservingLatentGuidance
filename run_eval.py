from evaluation_utils import evaluate_iplg_convergence
import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset
from torch.utils.data import DataLoader
from models import SEFiLMModel
import torch
from torch.nn import CrossEntropyLoss
import os
import pickle
from train_utils import train_IPLG
from data_utils import latent_MH_collate_fn
from pprint import pprint
import pandas as pd

device_name = 'cuda:0'
batch_size = 128

for data_file in ['CA_test.pickle', 'gjt_CA.pickle']:
    val_path = f'data/latent_datasets/{data_file}'

    with open(val_path, 'rb') as f:
        val_dataset = pickle.load(f)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=latent_MH_collate_fn)

    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
    # end device selection

    latent_loss_fn = torch.nn.MSELoss()

    tokenizer = CSGridMLMTokenizer(
        fixed_length=80,
        quantization='4th',
        intertwine_bar_info=True,
        trim_start=False,
        use_pc_roll=True,
        use_full_range_melody=False
    )
    mask_token_id = tokenizer.mask_token_id
    bar_token_id = tokenizer.bar_token_id

    logits_loss_fn =CrossEntropyLoss(ignore_index=-100)

    results_latent = {}
    results_logits = {}
    results_unique = {}
    results_nuc = {}

    for loss_scheme in ['f', 'fh', 'fhl', 'fl', 'hl', 'l']:
        d_model = 512
        transformer_model = SEFiLMModel(
            chord_vocab_size=len(tokenizer.vocab),
            d_model=d_model,
            nhead=8,
            num_layers=8,
            grid_length=80,
            pianoroll_dim=tokenizer.pianoroll_dim,
            guidance_dim=d_model,
            device=device,
        )
        checkpoint = torch.load(f'saved_models/iplg/iplg_{loss_scheme}_loss.pt', map_location=device_name)
        transformer_model.load_state_dict(checkpoint)
        transformer_model.to(device)

        eval_latent, eval_logits, eval_unique, eval_nuc = evaluate_iplg_convergence(
            transformer_model,
            val_loader,
            logits_loss_fn,
            latent_loss_fn,
            mask_token_id,
            bar_token_id,
            device,
            interpolate=False,
            extrapolate=False
        )
        print(loss_scheme, ' ------------- ')
        # pprint(eval_dict)
        results_latent[loss_scheme] = eval_latent
        results_logits[loss_scheme] = eval_logits
        results_unique[loss_scheme] = eval_unique
        results_nuc[loss_scheme] = eval_nuc

    df = pd.DataFrame(results_latent)
    df = df.T
    df.to_csv(f'results/eval_latent_{data_file}.csv', float_format='%.6f')
    latex_table = df.to_latex(float_format="%.6f")
    with open(f'results/eval_latent_{data_file}.tex', "w") as f:
        f.write(latex_table)
    print(df)

    df = pd.DataFrame(results_logits)
    df = df.T
    df.to_csv(f'results/eval_logits_{data_file}.csv', float_format='%.6f')
    latex_table = df.to_latex(float_format="%.6f")
    with open(f'results/eval_logits_{data_file}.tex', "w") as f:
        f.write(latex_table)
    print(df)

    df = pd.DataFrame(results_unique)
    df = df.T
    df['foreign_strength'] = df['fguide_funique'] / df['fguide_hunique']
    df['home_strength'] = df['hguide_hunique'] / df['hguide_funique']
    df.to_csv(f'results/eval_unique_{data_file}.csv', float_format='%.6f')
    latex_table = df.to_latex(float_format="%.6f")
    with open(f'results/eval_unique_{data_file}.tex', "w") as f:
        f.write(latex_table)
    print(df)

    df = pd.DataFrame(results_nuc)
    df = df.T
    # df['foreign_strength'] = df['nucleus_fguide_funique'] / df['nucleus_fguide_hunique']
    # df['home_strength'] = df['nucleus_hguide_hunique'] / df['nucleus_hguide_funique']
    df.to_csv(f'results/eval_nuc_{data_file}.csv', float_format='%.6f')
    latex_table = df.to_latex(float_format="%.6f")
    with open(f'results/eval_nuc_{data_file}.tex', "w") as f:
        f.write(latex_table)
    print(df)