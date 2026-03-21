from generate_utils import generate_files_with_nucleus, load_SEFiLMModel
import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
import os
from tqdm import tqdm

device_name = 'cuda:0'

tokenizer = CSGridMLMTokenizer(
    fixed_length=80,
    quantization='4th',
    intertwine_bar_info=True,
    trim_start=False,
    use_pc_roll=True,
    use_full_range_melody=False
)

midis_path = '/mnt/ssd2/maximos/data/gjt_melodies/gjt_CA'
midi_files = [f for f in os.listdir(midis_path) if f.endswith('.mxl') or f.endswith('.xml')]

for loss_scheme in ['real', 'f', 'fh', 'fhl', 'hl', 'l']:
    print(f'loss scheme: {loss_scheme}')
    for i in tqdm(range(len(midi_files))):
        h_idx = i
        g_idx = (i+1)%len(midi_files)
        input_f_path = os.path.join(midis_path, midi_files[h_idx])
        guidance_f_path = os.path.join(midis_path, midi_files[g_idx])
        mxl_folder_out = None
        midi_folder_out = f'MIDIs/jazz/{loss_scheme}'
        name_suffix = (loss_scheme == 'real')*str(h_idx) + \
            (loss_scheme != 'real')*f'h{h_idx}_g{g_idx}'

        if loss_scheme == 'real':
            model = load_SEFiLMModel(
                tokenizer,
                'l',
                device_name,
                d_model=512
            )
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
            guidance_f_path,
            mxl_folder_out,
            midi_folder_out,
            name_suffix,
            use_constraints=False,
            intertwine_bar_info=False, # no bar default
            normalize_tonality=False,
            temperature=1.0,
            p=0.9,
            unmasking_order='certain',
            create_gen = loss_scheme != 'real',
            create_real = loss_scheme == 'real',
            create_guide = False
        )