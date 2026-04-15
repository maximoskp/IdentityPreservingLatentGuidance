from evaluation_utils import evaluate_iplg_convergence, evaluate_actisteer_convergence
import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset
from models import EDFiLMModel, EDASModel
import torch
from torch.nn import CrossEntropyLoss
import os
import pickle
from train_utils import train_IPLG
from pprint import pprint
import pandas as pd

device_name = 'cuda:0'

for data_files in [
        ('nottingham_test.pickle', 'gjt_CA_test.pickle'),
        ('gjt_CA_test.pickle', 'nottingham_test.pickle')
    ]:
    home_path = f'data/latent_datasets/ED/{data_files[0]}'
    foreign_path = f'data/latent_datasets/ED/{data_files[1]}'

    with open(home_path, 'rb') as f:
        home_dataset = pickle.load(f)
    with open(foreign_path, 'rb') as f:
        foreign_dataset = pickle.load(f)

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
    results_conf= {}

    for loss_scheme in ['fhl', 'l']:# ['f', 'fh', 'fhl', 'fl', 'hl', 'l']:
        d_model = 512
        transformer_model = EDFiLMModel(
            chord_vocab_size=len(tokenizer.vocab),
            d_model=d_model,
            nhead=4,
            num_layers=4,
            grid_length=80,
            pianoroll_dim=tokenizer.pianoroll_dim,
            guidance_dim=d_model,
            device=device,
        )
        checkpoint = torch.load(f'saved_models/iplg/ED/iplg_{loss_scheme}_loss.pt', map_location=device_name)
        transformer_model.load_state_dict(checkpoint)
        transformer_model.to(device)
        transformer_model.eval()

        eval_latent, eval_logits, eval_unique, eval_conf = evaluate_iplg_convergence(
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
        )
        print(loss_scheme, ' ------------- ')
        # pprint(eval_dict)
        results_latent[loss_scheme] = eval_latent
        results_logits[loss_scheme] = eval_logits
        results_unique[loss_scheme] = eval_unique
        results_conf[loss_scheme] = eval_conf

    main_results_path = 'results/ED/guidance_eval'
    os.makedirs(main_results_path, exist_ok=True)
    df = pd.DataFrame(results_latent)
    df = df.T
    df.to_csv(f'{main_results_path}/latent_{data_files[0]}_{data_files[1]}.csv', float_format='%.6f')
    latex_table = df.to_latex(float_format="%.6f")
    with open(f'{main_results_path}/latent_{data_files[0]}_{data_files[1]}.tex', "w") as f:
        f.write(latex_table)
    print(df)

    df = pd.DataFrame(results_logits)
    df = df.T
    df.to_csv(f'{main_results_path}/logits_{data_files[0]}_{data_files[1]}.csv', float_format='%.6f')
    latex_table = df.to_latex(float_format="%.6f")
    with open(f'{main_results_path}/logits_{data_files[0]}_{data_files[1]}.tex', "w") as f:
        f.write(latex_table)
    print(df)

    df = pd.DataFrame(results_unique)
    df = df.T
    df['foreign_strength'] = df['fguide_funique'] / df['fguide_hunique']
    df['home_strength'] = df['hguide_hunique'] / df['hguide_funique']
    df.to_csv(f'{main_results_path}/unique_{data_files[0]}_{data_files[1]}.csv', float_format='%.2f')
    latex_table = df.to_latex(float_format="%.2f")
    with open(f'{main_results_path}/unique_{data_files[0]}_{data_files[1]}.tex', "w") as f:
        f.write(latex_table)
    print(df)

    df = pd.DataFrame(results_conf)
    df = df.T
    # df['foreign_strength'] = df['confident_fguide_funique'] / df['confident_fguide_hunique']
    # df['home_strength'] = df['confident_hguide_hunique'] / df['confident_hguide_funique']
    df.to_csv(f'{main_results_path}/conf_{data_files[0]}_{data_files[1]}.csv', float_format='%.2f')
    latex_table = df.to_latex(float_format="%.2f")
    with open(f'{main_results_path}/conf_{data_files[0]}_{data_files[1]}.tex', "w") as f:
        f.write(latex_table)
    print(df)

    # ==================== ActiSteer results =========================
    d_model = 512
    transformer_model = EDASModel(
        chord_vocab_size=len(tokenizer.vocab),
        d_model=d_model,
        nhead=4,
        num_layers=4,
        grid_length=80,
        pianoroll_dim=tokenizer.pianoroll_dim,
        guidance_dim=d_model,
        device=device,
    )
    checkpoint = torch.load('saved_models/iplg/ED/iplg_l_loss.pt', map_location=device_name)
    transformer_model.load_state_dict(checkpoint)
    transformer_model.to(device)
    transformer_model.eval()

    main_results_path = 'results/ED/actisteer_eval'
    os.makedirs(main_results_path, exist_ok=True)

    logits_all = {}
    confidence_all = {}
    for layers_to_steer in [ [1,2], [0,1,2,3] ]:
        layers_logits = {}
        layers_confidence = {}
        for alpha in [2.5, 5.0]:
            print(f'actisteer: {data_files[0]} - {data_files[0]}')
            print(f'{layers_to_steer} - {alpha}')
            logits_res, confidence_res = evaluate_actisteer_convergence(
                transformer_model,
                home_dataset,
                foreign_dataset,
                logits_loss_fn,
                latent_loss_fn,
                mask_token_id,
                bar_token_id,
                device,
                layers_to_steer=layers_to_steer,
                alpha=alpha
            )
            layers_logits[repr(alpha)] = logits_res
            layers_confidence[repr(alpha)] = confidence_res
        # end alpha for
        df = pd.DataFrame(layers_logits)
        print(df.head(10))
        df = df.T
        df['foreign_strength'] = df['fguide_funique'] / df['fguide_hunique']
        df['home_strength'] = df['hguide_hunique'] / df['hguide_funique']
        suffix = str(layers_to_steer).replace(', ','_').replace('[', '').replace(']', '')
        df.to_csv(f'{main_results_path}/unique_{data_files[0]}_{data_files[1]}_layers{suffix}.csv', float_format='%.2f')
        latex_table = df.to_latex(float_format="%.2f")
        with open(f'{main_results_path}/unique_{data_files[0]}_{data_files[1]}_layers{suffix}.tex', "w") as f:
            f.write(latex_table)
        print(df)

        df = pd.DataFrame(layers_confidence)
        df = df.T
        # df['foreign_strength'] = df['confident_fguide_funique'] / df['confident_fguide_hunique']
        # df['home_strength'] = df['confident_hguide_hunique'] / df['confident_hguide_funique']
        df.to_csv(f'{main_results_path}/conf_{data_files[0]}_{data_files[1]}_layers{suffix}.csv', float_format='%.2f')
        latex_table = df.to_latex(float_format="%.2f")
        with open(f'{main_results_path}/conf_{data_files[0]}_{data_files[1]}_layers{suffix}.tex', "w") as f:
            f.write(latex_table)
        print(df)
    # end layers_to_steer for