import torch
import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset
import pickle
from models import SEFiLMModel
import os
import numpy as np
from tqdm import tqdm

device_name = 'cuda:0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = CSGridMLMTokenizer(
    fixed_length=80,
    quantization='4th',
    intertwine_bar_info=True,
    trim_start=False,
    use_pc_roll=True,
    use_full_range_melody=False
)

d_model = 512
model_SE = SEFiLMModel(
    chord_vocab_size=len(tokenizer.vocab),
    d_model=d_model,
    nhead=8,
    num_layers=8,
    grid_length=80,
    pianoroll_dim=tokenizer.pianoroll_dim,
    guidance_dim=d_model,
    device=device,
)
checkpoint = torch.load('saved_models/SE/pretrained_epoch203_nvis2.pt', map_location=device_name)
model_SE.load_state_dict(checkpoint)
model_SE.eval()

def get_SE_embeddings_for_sequence(pianoroll, harmony_ids):
    melody_grid = torch.FloatTensor( pianoroll ).reshape( 1, pianoroll.shape[0], pianoroll.shape[1] )
    harmony_real = torch.LongTensor(harmony_ids).reshape(1, len(harmony_ids))
    _, hidden = model_SE(
        melody_grid=melody_grid.to(model_SE.device),
        harmony_tokens=harmony_real.to(model_SE.device),
        guidance_embedding=None,
        return_hidden=True
    )
    return hidden.detach().cpu().squeeze()
# end SE

def add_latent_to_dataset(dataset):
    new_dataset = []
    for d in tqdm(dataset):
        hidden = get_SE_embeddings_for_sequence(d['pianoroll'], d['harmony_ids'])
        d['latent'] = hidden.detach().cpu().numpy()
        new_dataset.append(d)
    return new_dataset
# end add_latent_to_dataset

def main():
    val_dirs = [
        '/mnt/ssd2/maximos/data/hooktheory_midi_hr/CA_train',
        '/mnt/ssd2/maximos/data/gjt_melodies/gjt_CA_train',
        '/mnt/ssd2/maximos/data/mel_harm_other_CA/nottingham_train',
        '/mnt/ssd2/maximos/data/mel_harm_other_CA/wikifonia_train',
        '/mnt/ssd2/maximos/data/hooktheory_midi_hr/CA_test',
        '/mnt/ssd2/maximos/data/gjt_melodies/gjt_CA_test',
        '/mnt/ssd2/maximos/data/mel_harm_other_CA/nottingham_test',
        '/mnt/ssd2/maximos/data/mel_harm_other_CA/wikifonia_test'
    ]

    full_dirs = []
    names = []
    for d in val_dirs:
        full_dirs.append(d)
        names.append(d.split('/')[-1])

    os.makedirs('data', exist_ok=True)
    os.makedirs('data/latent_datasets', exist_ok=True)
    os.makedirs('data/latent_datasets/SE', exist_ok=True)

    for d, n in zip(full_dirs, names):
        print(f'running for {n} in path: {d}')
        train_dataset = CSGridMLMDataset(d, tokenizer, frontloading=True, name_suffix='Q4_L80_bar_PC')
        bundled = add_latent_to_dataset(train_dataset)
        print('saving...')
        with open(f'data/latent_datasets/SE/{n}.pickle', 'wb') as f:
            pickle.dump(bundled, f)
# end main

if __name__ == '__main__':
    main()