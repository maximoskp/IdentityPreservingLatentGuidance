from generate_utils import generate_files_with_nucleus, load_SEASModel, get_actisteer_guidance
import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
import os
from tqdm import tqdm
import pickle
import torch
import numpy as np
from old.run_midi_metrics_source_target import compute_all_metrics, has_non_positive, source_target_distances
import pandas as pd
import numpy as np

num_guides_per_piece = 5
results_base_path = 'results/SE/pbp/ActivationSteering'
os.makedirs(results_base_path, exist_ok=True)

device_name = 'cuda:0'

temperature = 0.5

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

np.random.seed(42)
guide_idxs = {}
for i in range(len(source_files)):
    guide_idxs[i] = np.random.permutation(len(target_files))[:num_guides_per_piece]

loss_scheme = 'l'
model = load_SEASModel(
    tokenizer,
    device_name,
    d_model=512
)

results_bin_all = {}
results_dist_all = {}

for layers_to_steer in [ [4,5,6], [0,1,2,3,4,5,6,7] ]:
    for alpha in [2.5, 5.0]:
        folder_name = 'layers_' + \
            str(layers_to_steer).replace(', ','_').replace('[', '').replace(']', '') + \
            '_alpha_' + str(alpha).replace('.', '_')
        print(folder_name)
        num_gen = 0
        bin_all = {}
        dist_all = {}
        total_examples = len(source_files)*num_guides_per_piece
        for h_idx in range(len(source_files)):
            for g_idx in guide_idxs[h_idx]:
                print(f'folder_name: {folder_name} - trying: {num_gen} / {total_examples}')
                # home file
                input_f_path = os.path.join(source_path, source_files[h_idx])
                # guide file
                guide_f_path = os.path.join(target_path, target_files[g_idx])
                mxl_folder_out = None
                prefix = 'gen/SE/'
                midi_folder_out = f'MIDIs/nott2jazz_pbp/{prefix}{folder_name}'
                name_suffix = (loss_scheme == 'real' or loss_scheme == 'none')*str(h_idx) + \
                    (loss_scheme != 'real' and loss_scheme != 'none')*f'h{h_idx}_g{g_idx}'
                
                # encode input to get reference point
                source_encoded = tokenizer.encode(input_f_path)
                # encode guide to get reference point
                target_encoded = tokenizer.encode(guide_f_path)
                h = {}
                tmp_h = get_actisteer_guidance(
                    model,
                    tokenizer.bar_token_id,
                    tokenizer.mask_token_id,
                    torch.tensor(source_encoded['pianoroll']).unsqueeze(0).to(model.device),
                    torch.tensor(source_encoded['harmony_ids']).unsqueeze(0).to(model.device),
                    torch.tensor(target_encoded['pianoroll']).unsqueeze(0).to(model.device),
                    torch.tensor(target_encoded['harmony_ids']).unsqueeze(0).to(model.device),
                )
                for ii in layers_to_steer:
                    h[ii] = tmp_h[ii]

                g = generate_files_with_nucleus(
                    model,
                    tokenizer,
                    input_f_path,
                    mxl_folder_out,
                    midi_folder_out,
                    name_suffix,
                    guidance_f_path=None,
                    guidance_vec=None,
                    steering_vec=h,
                    steering_alpha=alpha,
                    use_constraints=False,
                    intertwine_bar_info=True, # no bar default
                    normalize_tonality=False,
                    temperature=temperature,
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
                except:
                    print(f'problem with file: {midi_out_path}')
            # end for g_idx counter
        # end for h_idx counter
    # make average metrics
    results_bin_all[folder_name] = {}
    results_dist_all[folder_name] = {}
    for k in bin_all.keys():
        results_bin_all[folder_name][k] = bin_all[k]/total_examples
        results_dist_all[folder_name][k] = dist_all[k]/total_examples
    # end for loss_schemes
# save to csvs
df_bin = pd.DataFrame.from_dict(results_bin_all, orient='index')
df_dist = pd.DataFrame.from_dict(results_dist_all, orient='index')

df_bin.to_csv(os.path.join(results_base_path, 'nott2_gjt_bin.csv'))
df_dist.to_csv(os.path.join(results_base_path, 'nott2_gjt_dist.csv'))

latex_table = df_bin.to_latex(float_format="%.4f")
with open(os.path.join(results_base_path, 'nott2_gjt_bin.tex'), "w") as f:
    f.write(latex_table)
latex_table = df_dist.to_latex(float_format="%.4f")
with open(os.path.join(results_base_path, 'nott2_gjt_dist.tex'), "w") as f:
    f.write(latex_table)