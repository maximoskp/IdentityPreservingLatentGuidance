from frechet_music_distance import FrechetMusicDistance
from frechet_music_distance.gaussian_estimators import LeoditWolfEstimator, MaxLikelihoodEstimator, OASEstimator, BootstrappingEstimator, ShrinkageEstimator
from frechet_music_distance.models import CLaMP2Extractor, CLaMPExtractor
import pandas as pd
import os
from frechet_music_distance.utils import clear_cache

clear_cache()

extractor = CLaMP2Extractor(verbose=True)
estimator = ShrinkageEstimator(shrinkage=0.1)
fmd = FrechetMusicDistance(feature_extractor=extractor, gaussian_estimator=estimator, verbose=True)

results_base_path = 'results/fmd'
os.makedirs(results_base_path, exist_ok=True)

datasets = ['nott2jazz', 'jazz2nott']
# models = ['SE', 'ED']
models = ['SE']
# loss_schemes = ['none', 'f', 'fh', 'fhl', 'hl', 'l']

for i_dataset in range(len(datasets)):
    dataset_results = {}
    tmp_source_root = datasets[i_dataset]
    tmp_target_root = datasets[ (i_dataset+1)%2 ]
    source_path = f"./MIDIs/{tmp_source_root}/real"
    target_path = f"./MIDIs/{tmp_target_root}/real"
    for model in models:
        dataset_results[model] = {}
        loss_schemes = os.listdir(f"./MIDIs/{tmp_source_root}/gen/{model}")
        for loss_scheme in loss_schemes:
            print(f'running for {datasets[i_dataset]} - {model} - {loss_scheme}')
            gen_path = f"./MIDIs/{tmp_source_root}/gen/{model}/{loss_scheme}"
            # compute fmd
            source_score = fmd.score(
                reference_path=source_path,
                test_path=gen_path
            )
            target_score = fmd.score(
                reference_path=target_path,
                test_path=gen_path
            )
            # compute and store ratio
            dataset_results[model][loss_scheme] = f'{source_score:.2f}/{target_score:.2f}'
        # end for loss_schemes
    # end for models
    # save for dataset
    # save to csvs
    df = pd.DataFrame.from_dict(dataset_results, orient='index')
    df.to_csv(os.path.join(results_base_path, f'{datasets[i_dataset]}.csv'))
    latex_table = df.to_latex(float_format="%.2f")
    with open(os.path.join(results_base_path, f'{datasets[i_dataset]}.tex'), "w") as f:
        f.write(latex_table)
# end for datasets