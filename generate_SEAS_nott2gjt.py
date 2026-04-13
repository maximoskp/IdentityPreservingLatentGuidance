from generate_utils import generate_files_with_nucleus, load_SEASModel, get_actisteer_guidance
import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
import os
from tqdm import tqdm
import pickle
import torch

device_name = 'cuda:0'

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

loss_scheme = 'l'
model = load_SEASModel(
    tokenizer,
    device_name,
    d_model=512
)

for layers_to_steer in [ [4,5,6], [5,6,7], [3,4,5,6,7], [0,1,2,3,4,5,6,7] ]:
    for alpha in [0.5, 1.0, 2.5, 5.0, 7.0, 10.0, 20.0]:
        folder_name = 'layers_' + \
            str(layers_to_steer).replace(', ','_').replace('[', '').replace(']', '') + \
            '_alpha_' + str(alpha).replace('.', '_')
        print(folder_name)
        for i in tqdm(range(len(midi_files))):
            h_idx = i
            input_f_path = os.path.join(midis_path, midi_files[h_idx])
            # compute steering vectors
            # first encode input to get reference point
            source_encoded = tokenizer.encode(input_f_path)
            # compute average foreign latent
            h = {}
            for ii in range(8):
                h[ii] = torch.zeros(1,1,512).to(model.device)
            for item in foreign_dataset:
                tmp_h = get_actisteer_guidance(
                    model,
                    tokenizer.bar_token_id,
                    tokenizer.mask_token_id,
                    torch.tensor(source_encoded['pianoroll']).unsqueeze(0).to(model.device),
                    torch.tensor(source_encoded['harmony_ids']).unsqueeze(0).to(model.device),
                    torch.tensor(item['pianoroll']).unsqueeze(0).to(model.device),
                    torch.tensor(item['harmony_ids']).unsqueeze(0).to(model.device),
                )
                for ii in range(8):
                    h[ii] += tmp_h[ii]/len(foreign_dataset)
            mxl_folder_out = None
            prefix = 'gen/SE/' if loss_scheme != 'real' else ''
            midi_folder_out = f'MIDIs/nott2jazz/{prefix}{folder_name}'
            name_suffix = str(h_idx)
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
                temperature=1.0,
                p=0.9,
                unmasking_order='certain',
                create_gen = loss_scheme != 'real',
                create_real = loss_scheme == 'real',
                create_guide = False
            )