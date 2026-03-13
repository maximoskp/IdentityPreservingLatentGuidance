import GridMLM_tokenizers
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset, CSGridMLM_collate_fn
from torch.utils.data import DataLoader
from models import SEFiLMModel, ContrastiveSpaceModel, contrastive_normalized_loss
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import os
import pickle
from train_utils import train_IPLG
from data_utils import latent_MH_collate_fn
import argparse

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script for training a selected contrastive space.')

    # Define arguments
    parser.add_argument('-g', '--gpu', type=int, help='Specify whether and which GPU will be used by used by index. Not using this argument means use CPU.', required=False)
    parser.add_argument('-e', '--epochs', type=int, help='Specify number of epochs. Defaults to 100.', required=False)
    parser.add_argument('-l', '--learningrate', type=float, help='Specify learning rate. Defaults to 1e-5.', required=False)
    parser.add_argument('-b', '--batchsize', type=int, help='Specify batch size. Defaults to 32.', required=False)

    # Parse the arguments
    args = parser.parse_args()
    device_name = 'cpu'
    if args.gpu is not None:
        if args.gpu > -1:
            device_name = 'cuda:' + str(args.gpu)
    epochs = 100
    if args.epochs:
        epochs = args.epochs
    lr = 1e-5
    if args.learningrate:
        lr = args.learningrate
    batch_size = 32
    if args.batchsize:
        batch_size = args.batchsize

    train_path = 'data/contrastive_dataset/CA_train.pickle'
    val_path = 'data/contrastive_dataset/CA_test.pickle'

    with open(train_path, 'rb') as f:
        train_dataset = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_dataset = pickle.load(f)

    source_dim = train_dataset[0]['latent'].shape[0]
    transformer_dim = train_dataset[0]['transformer_embeddings'].shape[0]

    collator = latent_MH_collate_fn(pad_id=0)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
    # end device selection
    
    latent_loss_fn = torch.nn.MSELoss()

    tokenizer = CSGridMLMTokenizer(
        fixed_length=80,
        quantization='4th',
        intertwine_bar_info=True,
        trim_start=False,
        use_pc_roll=True,
        use_full_range_melody=False
    )

    logits_loss_fn =CrossEntropyLoss(ignore_index=-100)

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
    checkpoint = torch.load('saved_models/SE/pretrained.pt', map_location=device_name)
    transformer_model.load_state_dict(checkpoint)
    transformer_model.to(device)
    optimizer = AdamW(transformer_model.film_parameters(), lr=lr)

    # save results
    results_path = os.path.join( 'results', 'iplg', f'{source_name}.csv' )
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/iplg', exist_ok=True)

    os.makedirs('saved_models/', exist_ok=True)
    os.makedirs('saved_models/iplg/', exist_ok=True)
    save_dir = 'saved_models/lacta/'
    transformer_path = save_dir + f'{source_name}.pt'

    train_lacta(
        transformer_model, contrastive_model, 
        contrastive_loss_fn, logits_loss_fn,
        optimizer, train_loader, val_loader, tokenizer.mask_token_id,
        source_key,
        epochs=epochs,
        exponent=-1,
        results_path=results_path,
        transformer_path=transformer_path,
        bar_token_id=tokenizer.bar_token_id,
        validations_per_epoch=1,
        tqdm_position=0,
        freeze_base=True
    )

# end main

if __name__ == '__main__':
    main()