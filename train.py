import os
import time
import requests
import tarfile
import numpy as np
import argparse

from models import GAT
from utils import load_cora

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

#################################
### TRAIN AND TEST FUNCTIONS  ###
#################################

def train_iter(epoch, model, optimizer, criterion, input, target, mask_train, mask_val, print_every=10):
    start_t = time.time()
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(*input) 
    loss = criterion(output[mask_train], target[mask_train]) # Compute the loss using the training mask

    loss.backward()
    optimizer.step()

    # Evaluate the model performance on training and validation sets
    loss_train, acc_train = test(model, criterion, input, target, mask_train)
    loss_val, acc_val = test(model, criterion, input, target, mask_val)

    if epoch % print_every == 0:
        # Print the training progress at specified intervals
        print(f'Epoch: {epoch:04d} ({(time.time() - start_t):.4f}s) loss_train: {loss_train:.4f} acc_train: {acc_train:.4f} loss_val: {loss_val:.4f} acc_val: {acc_val:.4f}')


def test(model, criterion, input, target, mask):
    model.eval()
    with torch.no_grad():
        output = model(*input)
        output, target = output[mask], target[mask]

        loss = criterion(output, target)
        acc = (output.argmax(dim=1) == target).float().sum() / len(target)
    return loss.item(), acc.item()


if __name__ == '__main__':

    # Training settings
    # All defalut values are the same as in the config used in the main paper

    parser = argparse.ArgumentParser(description='PyTorch Graph Attention Network')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate (default: 0.005)')
    parser.add_argument('--l2', type=float, default=5e-4,
                        help='weight decay (default: 6e-4)')
    parser.add_argument('--dropout-p', type=float, default=0.6,
                        help='dropout probability (default: 0.6)')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='dimension of the hidden representation (default: 64)')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='number of the attention heads (default: 4)')
    parser.add_argument('--concat-heads', action='store_true', default=False,
                        help='wether to concatinate attention heads, or average over them (default: False)')
    parser.add_argument('--val-every', type=int, default=20,
                        help='epochs to wait for print training and validation evaluation (default: 20)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=13, metavar='S',
                        help='random seed (default: 13)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    # Set the device to run on
    if use_cuda:
        device = torch.device('cuda')
    elif use_mps:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using {device} device')

    # Load the dataset
    cora_url = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz'
    path = './cora'

    if os.path.isfile(os.path.join(path, 'cora.content')) and os.path.isfile(os.path.join(path, 'cora.cites')):
        print('Dataset already downloaded...')
    else:
        print('Downloading dataset...')
        with requests.get(cora_url, stream=True) as tgz_file:
            with tarfile.open(fileobj=tgz_file.raw, mode='r:gz') as tgz_object:
                tgz_object.extractall()

    print('Loading dataset...')
    # Load the dataset
    features, labels, adj_mat = load_cora(device=device)
    # Split the dataset into training, validation, and test sets
    idx = torch.randperm(len(labels)).to(device)
    idx_test, idx_val, idx_train = idx[:1200], idx[1200:1600], idx[1600:]


    # Create the model
    # The model consists of a 2-layer stack of Graph Attention Layers (GATs).
    gat_net = GAT(
        in_features=features.shape[1],          # Number of input features per node  
        n_hidden=args.hidden_dim,               # Output size of the first Graph Attention Layer
        n_heads=args.num_heads,                 # Number of attention heads in the first Graph Attention Layer
        num_classes=labels.max().item() + 1,    # Number of classes to predict for each node
        concat=args.concat_heads,               # Wether to concatinate attention heads
        dropout=args.dropout_p,                 # Dropout rate
        leaky_relu_slope=0.2                    # Alpha (slope) of the leaky relu activation
    ).to(device)

    # configure the optimizer and loss function
    optimizer = Adam(gat_net.parameters(), lr=args.lr, weight_decay=args.l2)
    criterion = nn.NLLLoss()

    # Train and evaluate the model
    for epoch in range(args.epochs):
        train_iter(epoch + 1, gat_net, optimizer, criterion, (features, adj_mat), labels, idx_train, idx_val, args.val_every)
        if args.dry_run:
            break
    loss_test, acc_test = test(gat_net, criterion, (features, adj_mat), labels, idx_test)
    print(f'Test set results: loss {loss_test:.4f} accuracy {acc_test:.4f}')