from generate_utils import generate_files_with_nucleus
import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
import os
from tqdm import tqdm
import torch
from models import EDFiLMModel

device_name = 'cuda:0'

tokenizer = CSGridMLMTokenizer(
    fixed_length=80,
    quantization='4th',
    intertwine_bar_info=True,
    trim_start=False,
    use_pc_roll=True,
    use_full_range_melody=False
)

if device_name == 'cpu':
    device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        device = torch.device(device_name)
    else:
        print('Selected device not available: ' + device_name)
# end device selection
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
checkpoint = torch.load(f'saved_models/ED/pretrained.pt', map_location=device_name)
transformer_model.load_state_dict(checkpoint)
transformer_model.to(device)
transformer_model.eval()

jazz_path = '/mnt/ssd2/maximos/data/gjt_melodies/gjt_CA_test'
jazz_files = [f for f in os.listdir(jazz_path) if f.endswith('.mxl') or f.endswith('.xml')]

nottingham_path = '/mnt/ssd2/maximos/data/mel_harm_other_CA/nottingham_test'
nottingham_files = [f for f in os.listdir(nottingham_path) if f.endswith('.mid') or f.endswith('.midi')]

all_paths = [jazz_path, nottingham_path]
all_files = [jazz_files, nottingham_files]
all_names = ['jazz', 'nottingham']

loss_scheme = 'pre_none'
for i in [0,1]:
    data_path = all_paths[i]
    data_files = all_files[i]
    data_name = all_names[i]
    for i in tqdm(range(len(data_files))):
        h_idx = i
        input_f_path = os.path.join(data_path, data_files[h_idx])
        mxl_folder_out = None
        prefix = 'gen/ED/'
        midi_folder_out = f'MIDIs/{data_name}/{prefix}{loss_scheme}'
        name_suffix = str(h_idx)
        
        g = generate_files_with_nucleus(
            transformer_model,
            tokenizer,
            input_f_path,
            None,
            mxl_folder_out,
            midi_folder_out,
            name_suffix,
            use_constraints=False,
            intertwine_bar_info=True, # no bar default
            normalize_tonality=False,
            temperature=0.5,
            p=0.9,
            unmasking_order='start',
            create_gen = True,
            create_real = False,
            create_guide = False
        )