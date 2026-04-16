from generate_utils import generate_files_with_nucleus, load_EDFiLMModel
import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
import os
from tqdm import tqdm
import pickle
import torch
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

    tokenizer = CSGridMLMTokenizer(
        fixed_length=80,
        quantization='4th',
        intertwine_bar_info=True,
        trim_start=False,
        use_pc_roll=True,
        use_full_range_melody=False
    )

    midis_path = '/mnt/ssd2/maximos/data/gjt_melodies/gjt_CA_test'
    midi_files = [f for f in os.listdir(midis_path) if f.endswith('.mxl') or f.endswith('.xml')]

    foreign_path = 'data/latent_datasets/ED/nottingham_test.pickle'
    with open(foreign_path, 'rb') as f:
        foreign_dataset = pickle.load(f)
    # compute average foreign latent
    guidance_vec = torch.zeros(512)
    for item in foreign_dataset:
        guidance_vec += torch.tensor(item['latent'])
    guidance_vec /= len(foreign_dataset)
    guidance_vec = guidance_vec.unsqueeze(dim=0)

    # for loss_scheme in ['real', 'none', 'f', 'fh', 'fhl', 'hl', 'l']:
    for loss_scheme in ['real', 'none', 'fhl','l', 'f']:
        print(f'loss scheme: {loss_scheme}')
        for i in tqdm(range(len(midi_files))):
            # item = foreign_dataset[i % len(foreign_dataset)]
            # guidance_vec = torch.tensor(item['latent'])
            h_idx = i
            input_f_path = os.path.join(midis_path, midi_files[h_idx])
            mxl_folder_out = None
            prefix = 'gen/ED/' if loss_scheme != 'real' else ''

            if loss_scheme in ['real', 'none', 'l', 'f']:
                model = load_EDFiLMModel(
                    tokenizer,
                    'l',
                    device_name,
                    d_model=512
                )
                guidance_f_path = None
                midi_folder_out = f'MIDIs/jazz2nott/{prefix}{loss_scheme}'
                name_suffix = (loss_scheme == 'real' or loss_scheme == 'none')*str(h_idx) + \
                    (loss_scheme != 'real' and loss_scheme != 'none')*f'{h_idx}'
            else:
                model = load_EDFiLMModel(
                    tokenizer,
                    loss_scheme,
                    device_name,
                    logits_lambda=logits_lambda,
                    d_model=512
                )
                folder_suffix = str(logits_lambda).replace('.', '_')
                midi_folder_out = f'MIDIs/jazz2nott/{prefix}{loss_scheme}_{folder_suffix}'
                name_suffix = (loss_scheme == 'real' or loss_scheme == 'none')*str(h_idx) + \
                    (loss_scheme != 'real' and loss_scheme != 'none')*f'{h_idx}'
            
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
                unmasking_order='start',
                create_gen = loss_scheme != 'real',
                create_real = loss_scheme == 'real',
                create_guide = False
            )

# end main

if __name__ == '__main__':
    main()