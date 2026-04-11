from generate_utils import generate_files_with_nucleus, load_SEFiLMModel
import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
import os
from tqdm import tqdm
import pickle
import torch
import numpy as np
from run_midi_metrics_source_target import compute_all_metrics, has_non_positive, source_target_distances
import pandas as pd
from frechet_music_distance import FrechetMusicDistance
from frechet_music_distance.gaussian_estimators import LeoditWolfEstimator, MaxLikelihoodEstimator, OASEstimator, BootstrappingEstimator, ShrinkageEstimator
from frechet_music_distance.models import CLaMP2Extractor, CLaMPExtractor
from frechet_music_distance.utils import clear_cache

clear_cache()

extractor = CLaMP2Extractor(verbose=True)
estimator = ShrinkageEstimator(shrinkage=0.1)
fmd = FrechetMusicDistance(feature_extractor=extractor, gaussian_estimator=estimator, verbose=True)

total_examples = 100
results_base_path = 'results/metrics_100/SE/'
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

source_path = 'MIDIs/jazz/real'
source_files = [f for f in os.listdir(source_path) if f.endswith('.mid') or f.endswith('.midi')]

target_path = 'MIDIs/nottingham/real'
target_files = [f for f in os.listdir(target_path) if f.endswith('.mid') or f.endswith('.midi')]

loss_schemes = ['none', 'f', 'fh', 'fhl', 'hl', 'l']

results_bin_all = {}
results_dist_all = {}

results_fmd_bin = {}
results_fmd_ratio = {}

for loss_scheme in loss_schemes:
    print(f'loss scheme: {loss_scheme}')
    num_gen = 0
    bin_all = {}
    dist_all = {}
    fmd_bin = 0
    fmd_ratio = 0
    while num_gen < total_examples:
        print(f'loss scheme: {loss_scheme} - trying: {num_gen} / {total_examples}')
        # home file
        h_idx = np.random.randint(len(source_files))
        input_f_path = os.path.join(source_path, source_files[h_idx])
        # guide file
        g_idx = np.random.randint(len(target_files))
        guide_f_path = os.path.join(target_path, target_files[g_idx])
        mxl_folder_out = None
        prefix = 'gen/SE/' if loss_scheme != 'real' else ''
        midi_folder_out = f'MIDIs/jazz2nott_100/{prefix}{loss_scheme}'
        name_suffix = (loss_scheme == 'real' or loss_scheme == 'none')*str(h_idx) + \
            (loss_scheme != 'real' and loss_scheme != 'none')*f'h{h_idx}_g{g_idx}'

        if loss_scheme == 'real' or loss_scheme == 'none':
            model = load_SEFiLMModel(
                tokenizer,
                'l',
                device_name,
                d_model=512
            )
            # guidance_f_path = None
        else:
            model = load_SEFiLMModel(
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
            temperature=1.0,
            p=0.9,
            unmasking_order='certain',
            create_gen = loss_scheme != 'real',
            create_real = loss_scheme == 'real',
            create_guide = False
        )
        # midi out path
        midi_out_path = os.path.join(midi_folder_out, 'gen_' + name_suffix + '.mid')

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
            # fmd
            source_fmd = fmd.score_individual(
                reference_path=source_path,
                test_song_path=midi_out_path,
            )
            target_fmd = fmd.score_individual(
                reference_path=target_path,
                test_song_path=midi_out_path,
            )
            fmd_bin += int(target_fmd < source_fmd)/total_examples
            fmd_ratio += (target_fmd/source_fmd)/total_examples
        # end if has_non_positive
    # end while 100 counter
    # make average metrics
    results_bin_all[loss_scheme] = {}
    results_dist_all[loss_scheme] = {}
    results_fmd_bin[loss_scheme] = fmd_bin
    results_fmd_ratio[loss_scheme] = fmd_ratio
    for k in bin_all.keys():
        results_bin_all[loss_scheme][k] = bin_all[k]/total_examples
        results_dist_all[loss_scheme][k] = dist_all[k]/total_examples
    # end for loss_schemes
# save to csvs
df_bin = pd.DataFrame.from_dict(results_bin_all, orient='index')
df_dist = pd.DataFrame.from_dict(results_dist_all, orient='index')

df_bin.to_csv(os.path.join(results_base_path, 'gjt2_nott_bin.csv'))
df_dist.to_csv(os.path.join(results_base_path, 'gjt2_nott_dist.csv'))

df_bin = pd.DataFrame.from_dict(results_fmd_bin, orient='index')
df_ratio = pd.DataFrame.from_dict(results_fmd_ratio, orient='index')

df_bin.to_csv(os.path.join(results_base_path, 'nott2_gjt_fmd_bin.csv'))
df_ratio.to_csv(os.path.join(results_base_path, 'nott2_gjt_fmd_ratio.csv'))
latex_table = df_bin.to_latex(float_format="%.4f")
with open(os.path.join(results_base_path, 'nott2_gjt_fmd_bin.tex'), "w") as f:
    f.write(latex_table)
latex_table = df_ratio.to_latex(float_format="%.4f")
with open(os.path.join(results_base_path, 'nott2_gjt_fmd_ratio.tex'), "w") as f:
    f.write(latex_table)