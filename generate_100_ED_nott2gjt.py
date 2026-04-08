from generate_utils import generate_files_with_nucleus, load_EDFiLMModel
import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
import os
from tqdm import tqdm
import pickle
import torch
import numpy as np
from run_midi_metrics_source_target import compute_all_metrics, has_non_positive, source_target_distances
import pandas as pd

total_examples = 100
results_base_path = 'results/metrics_100/ED/'
os.makedirs(results_base_path, exist_ok=True)

device_name = 'cuda:0'

tokenizer = CSGridMLMTokenizer(
    fixed_length=80,
    quantization='4th',
    intertwine_bar_info=True,
    trim_start=False,
    use_pc_roll=True,
    use_full_range_melody=False
)

source_path = 'MIDIs/nottingham/real'
source_files = [f for f in os.listdir(source_path) if f.endswith('.mid') or f.endswith('.midi')]

target_path = 'MIDIs/jazz/real'
target_files = [f for f in os.listdir(target_path) if f.endswith('.mid') or f.endswith('.midi')]

loss_schemes = ['none', 'f', 'fh', 'fhl', 'hl', 'l']

results_bin_all = {}
results_dist_all = {}

for loss_scheme in loss_schemes:
    print(f'loss scheme: {loss_scheme}')
    num_gen = 0
    bin_all = {}
    dist_all = {}
    while num_gen < total_examples:
        print(f'loss scheme: {loss_scheme} - trying: {num_gen} / {total_examples}')
        # home file
        h_idx = np.random.randint(len(source_files))
        input_f_path = os.path.join(source_path, source_files[h_idx])
        # guide file
        g_idx = np.random.randint(len(target_files))
        guide_f_path = os.path.join(target_path, target_files[g_idx])
        mxl_folder_out = None
        prefix = 'gen/ED/' if loss_scheme != 'real' else ''
        midi_folder_out = f'MIDIs/nott2jazz_100/{prefix}{loss_scheme}'
        name_suffix = (loss_scheme == 'real' or loss_scheme == 'none')*str(h_idx) + \
            (loss_scheme != 'real' and loss_scheme != 'none')*f'h{h_idx}_g{g_idx}'

        if loss_scheme == 'real' or loss_scheme == 'none':
            model = load_EDFiLMModel(
                tokenizer,
                'l',
                device_name,
                d_model=512
            )
            # guidance_f_path = None
        else:
            model = load_EDFiLMModel(
                tokenizer,
                loss_scheme,
                device_name,
                d_model=512
            )
        
        g = generate_files_with_nucleus(
            model,
            tokenizer,
            input_f_path,
            mxl_folder_out,
            midi_folder_out,
            name_suffix,
            guidance_f_path=guide_f_path,
            guidance_vec=None,
            use_constraints=False,
            intertwine_bar_info=True, # no bar default
            normalize_tonality=False,
            temperature=1.5,
            p=0.9,
            unmasking_order='certain',
            create_gen = loss_scheme != 'real',
            create_real = loss_scheme == 'real',
            create_guide = False
        )
        # midi out path
        midi_out_path = os.path.join(midi_folder_out, 'gen_' + name_suffix + '.mid')

        try:
            # compute metrics
            source_metrics = compute_all_metrics(input_f_path)
            target_metrics = compute_all_metrics(guide_f_path)
            gen_metrics = compute_all_metrics(midi_out_path)

            # CHE CC CTD CTnCTR PCS MCTD HRHE HRC CBS
            if not has_non_positive(gen_metrics):
                num_gen += 1
                bin_out, dist_diff = source_target_distances(gen_metrics, source_metrics, target_metrics)
                for k, v in bin_out.items():
                    if k in bin_all.keys():
                        bin_all[k] += bin_out[k]
                        dist_all[k] += dist_diff[k]
                    else:
                        bin_all[k] = bin_out[k]
                        dist_all[k] = dist_diff[k]
                # end for items
            # end if has_non_positive
        except:
            print(f'problem with file: {midi_out_path}')
    # end while 100 counter
    # make average metrics
    results_bin_all[loss_scheme] = {}
    results_dist_all[loss_scheme] = {}
    for k in bin_all.keys():
        results_bin_all[loss_scheme][k] = bin_all[k]/total_examples
        results_dist_all[loss_scheme][k] = dist_all[k]/total_examples
    # end for loss_schemes
# save to csvs
df_bin = pd.DataFrame.from_dict(results_bin_all, orient='index')
df_dist = pd.DataFrame.from_dict(results_dist_all, orient='index')

df_bin.to_csv(os.path.join(results_base_path, 'nott2_gjt_bin.csv'))
df_dist.to_csv(os.path.join(results_base_path, 'nott2_gjt_dist.csv'))