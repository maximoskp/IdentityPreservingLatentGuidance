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

results_base_path = 'results/fmd_pretrained'
os.makedirs(results_base_path, exist_ok=True)

datasets = ['nottingham', 'jazz']
models = ['SE', 'ED']
pretrained_versions = [
    'pretrained',
    'pretrained_epoch168_nvis50',
    'pretrained_epoch177_nvis30',
    'pretrained_epoch182_nvis20',
    'pretrained_epoch189_nvis10',
    'pretrained_epoch196_nvis5',
    'pretrained_epoch200_nvis3',
    'pretrained_epoch203_nvis2',
]

for model in models:
    model_results = {}
    for dataset in datasets:
        model_results[dataset] = {}
        for pretrained_version in pretrained_versions:
            print(f'running for {model} - {dataset} - {pretrained_version}')
            gt_root = f'MIDIs/{dataset}/real'
            gen_root = f'MIDIs/{dataset}/gen/{model}/{pretrained_version}'
            fmd_score = fmd.score(
                reference_path=gt_root,
                test_path=gen_root
            )
            model_results[dataset][pretrained_version] = fmd_score
        # end for pretrained_versions
    # end for datasets
    # save for models
    # save to csvs
    df = pd.DataFrame.from_dict(model_results, orient='index')
    df.loc['avg'] = df.mean()
    df.to_csv(os.path.join(results_base_path, f'{model}.csv'))
    latex_table = df.to_latex(float_format="%.4f")
    with open(os.path.join(results_base_path, f'{model}.tex'), "w") as f:
        f.write(latex_table)
# end for models