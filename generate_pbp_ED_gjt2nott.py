from generate_utils import generate_files_with_nucleus, load_EDFiLMModel
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
import argparse

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script for training a selected contrastive space.')

    # Define arguments
    parser.add_argument('-l', '--logits_lambda', type=float, help='Logits lambda: the multiplier for the logits loss.', required=False)
    parser.add_argument('-t', '--temperature', type=float, help='Temperature for generation.', required=False)
    parser.add_argument('-g', '--gpu', type=int, help='Specify whether and which GPU will be used by used by index. Not using this argument means use CPU.', required=False)

    logits_lambda = 0.1
    device_name = 'cuda:0'
    temperature = 0.5

    # Parse the arguments
    args = parser.parse_args()
    if args.logits_lambda:
        logits_lambda = args.logits_lambda
    if args.temperature:
        temperature = args.temperature
    if args.gpu is not None:
        if args.gpu > -1:
            device_name = 'cuda:' + str(args.gpu)

    num_guides_per_piece = 5
    results_base_path = 'results/ED/pbp/LatentTrace'
    os.makedirs(results_base_path, exist_ok=True)

    tokenizer = CSGridMLMTokenizer(
        fixed_length=80,
        quantization='4th',
        intertwine_bar_info=True,
        trim_start=False,
        use_pc_roll=True,
        use_full_range_melody=False
    )

    source_path = 'MIDIs/jazz/real'
    source_files = [f for f in os.listdir(source_path) if f.endswith('.mid') or f.endswith('.midi')]

    target_path = 'MIDIs/nottingham/real'
    target_files = [f for f in os.listdir(target_path) if f.endswith('.mid') or f.endswith('.midi')]

    np.random.seed(42)
    guide_idxs = {}
    for i in range(len(source_files)):
        guide_idxs[i] = np.random.permutation(len(target_files))[:num_guides_per_piece]

    # loss_schemes = ['f', 'fh', 'fhl', 'hl', 'l']
    # loss_schemes = ['fhl', 'l', 'f']
    loss_schemes = ['fhl']

    results_bin_all = {}
    results_dist_all = {}

    for loss_scheme in loss_schemes:
        print(f'loss scheme: {loss_scheme}')
        num_gen = 0
        bin_all = {}
        dist_all = {}
        total_examples = len(source_files)*num_guides_per_piece
        for h_idx in range(len(source_files)):
            for g_idx in guide_idxs[h_idx]:
                print(f'loss scheme: {loss_scheme} - trying: {num_gen} / {total_examples}')
                # home file
                input_f_path = os.path.join(source_path, source_files[h_idx])
                # guide file
                guide_f_path = os.path.join(target_path, target_files[g_idx])
                mxl_folder_out = None
                prefix = 'gen/ED/' if loss_scheme != 'real' else ''

                if loss_scheme in ['real', 'none', 'l', 'f']:
                    model = load_EDFiLMModel(
                        tokenizer,
                        'l',
                        device_name,
                        d_model=512
                    )
                    midi_folder_out = f'MIDIs/jazz2nott_pbp/{prefix}{loss_scheme}'
                    name_suffix = (loss_scheme == 'real' or loss_scheme == 'none')*str(h_idx) + \
                        (loss_scheme != 'real' and loss_scheme != 'none')*f'h{h_idx}_g{g_idx}'
                else:
                    model = load_EDFiLMModel(
                        tokenizer,
                        loss_scheme,
                        device_name,
                        logits_lambda=logits_lambda,
                        d_model=512
                    )
                    folder_suffix = str(logits_lambda).replace('.', '_')
                    midi_folder_out = f'MIDIs/jazz2nott_pbp/{prefix}{loss_scheme}_{folder_suffix}'
                    name_suffix = (loss_scheme == 'real' or loss_scheme == 'none')*str(h_idx) + \
                        (loss_scheme != 'real' and loss_scheme != 'none')*f'h{h_idx}_g{g_idx}'
                
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
                    temperature=temperature,
                    p=0.9,
                    unmasking_order='start',
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
            # end for g_idx counter
        # end for h_idx counter
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

    df_bin.to_csv(os.path.join(results_base_path, 'gjt2_nott_bin.csv'))
    df_dist.to_csv(os.path.join(results_base_path, 'gjt2_nott_dist.csv'))

    latex_table = df_bin.to_latex(float_format="%.4f")
    with open(os.path.join(results_base_path, 'gjt2_nott_bin.tex'), "w") as f:
        f.write(latex_table)
    latex_table = df_dist.to_latex(float_format="%.4f")
    with open(os.path.join(results_base_path, 'gjt2_nott_dist.tex'), "w") as f:
        f.write(latex_table)

# end main

if __name__ == '__main__':
    main()