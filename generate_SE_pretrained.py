from generate_utils import generate_files_with_nucleus
import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
import os
from tqdm import tqdm
import torch
from models import SEFiLMModel

device_name = 'cuda:0'

tokenizer = CSGridMLMTokenizer(
    fixed_length=80,
    quantization='4th',
    intertwine_bar_info=True,
    trim_start=False,
    use_pc_roll=True,
    use_full_range_melody=False
)

midis_path = '/mnt/ssd2/maximos/data/hooktheory_midi_hr/CA_test'
midi_files = [f for f in os.listdir(midis_path) if f.endswith('.mid') or f.endswith('.midi')]

loss_scheme = 'pre_none'
for i in tqdm(range(len(midi_files))):
    h_idx = i
    g_idx = (i+1)%len(midi_files)
    input_f_path = os.path.join(midis_path, midi_files[h_idx])
    guidance_f_path = os.path.join(midis_path, midi_files[g_idx])
    mxl_folder_out = None
    prefix = 'gen/SE/' if loss_scheme != 'real' else ''
    midi_folder_out = f'MIDIs/testset/{prefix}{loss_scheme}'
    name_suffix = str(h_idx)

    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
    # end device selection
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
    checkpoint = torch.load(f'saved_models/SE/pretrained.pt', map_location=device_name)
    transformer_model.load_state_dict(checkpoint)
    transformer_model.to(device)
    transformer_model.eval()
    
    g = generate_files_with_nucleus(
        transformer_model,
        tokenizer,
        input_f_path,
        guidance_f_path,
        mxl_folder_out,
        midi_folder_out,
        name_suffix,
        use_constraints=False,
        intertwine_bar_info=True, # no bar default
        normalize_tonality=False,
        temperature=0.5,
        p=0.9,
        unmasking_order='certain',
        create_gen = True,
        create_real = False,
        create_guide = False
    )