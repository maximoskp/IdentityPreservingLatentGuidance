from generate_utils import generate_files_with_nucleus, load_SEFiLMModel
import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
import os
from tqdm import tqdm
import pickle
import torch

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

midis_path = '/mnt/ssd2/maximos/data/mel_harm_other_CA/nottingham_test'
midi_files = [f for f in os.listdir(midis_path) if f.endswith('.mid') or f.endswith('.midi')]

foreign_path = 'data/latent_datasets/SE/gjt_CA_test.pickle'
with open(foreign_path, 'rb') as f:
    foreign_dataset = pickle.load(f)
# compute average foreign latent
guidance_vec = torch.zeros(512)
for item in foreign_dataset:
    guidance_vec += torch.tensor(item['latent'])
guidance_vec /= len(foreign_dataset)
guidance_vec = guidance_vec.unsqueeze(dim=0)

# for loss_scheme in ['real', 'none', 'f', 'fh', 'fhl', 'hl', 'l']:
for loss_scheme in ['real', 'none', 'fhl','l']:
    print(f'loss scheme: {loss_scheme}')
    for i in tqdm(range(len(midi_files))):
        # item = foreign_dataset[i % len(foreign_dataset)]
        # guidance_vec = torch.tensor(item['latent'])
        h_idx = i
        input_f_path = os.path.join(midis_path, midi_files[h_idx])
        mxl_folder_out = None
        prefix = 'gen/SE/' if loss_scheme != 'real' else ''
        midi_folder_out = f'MIDIs/nott2jazz/{prefix}{loss_scheme}'
        name_suffix = (loss_scheme == 'real' or loss_scheme == 'none')*str(h_idx) + \
            (loss_scheme != 'real' and loss_scheme != 'none')*f'{h_idx}'

        if loss_scheme == 'real' or loss_scheme == 'none':
            model = load_SEFiLMModel(
                tokenizer,
                'l',
                device_name,
                d_model=512
            )
            guidance_f_path = None
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
            guidance_f_path=None,
            guidance_vec=guidance_vec,
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