import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import cuda
import numpy as np
from dataset import CustomDataset
from model import SRU
import argparse
import csv

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = 'cuda' if cuda.is_available() else 'cpu'
    device = 'cpu'

    # Building model
    model = SRU(input_size=args.input_size, hidden_size=args.hidden_size)
    model = model.to(device)

    # Creating Dataset
    dataset =  CustomDataset(args.nums_size)
    train_size = int(len(dataset)* args.train_split)
    eval_size = len(dataset)- train_size
    
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    train_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 0
    }
    eval_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 0
    }
    training_loader = DataLoader(train_dataset, **train_params)
    eval_loader = DataLoader(eval_dataset, **eval_params)
    # Setting the loss and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Training Loop
    model.train()
    for epoch in range(1, args.epochs + 1):
        for _, data in enumerate(training_loader):
            input_seq = data['inputs'].permute(1, 0, 2).to(device)
            target= data['label'].permute(1, 0, 2).to(device)
            output = model(input_seq)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print('epoch: {}, loss: {}'.format(epoch, loss.item()))

    # Evaluation
    results = []
    model.eval()
    for _, data in enumerate(eval_loader):
        input = data['inputs'].permute(1, 0, 2).to(device)
        x1 = torch.round(input[0, 0, 0]).int()
        x2 = torch.round(input[1, 0, 0]).int()
        x3 = torch.round(input[2, 0, 0]).int()
        output = model(input)
        next_digit = torch.round(output[2, 0, 0]).int()
        print('Given a sequence: x1={}, x2={}, x3={}, SRU predictd that x4={}'.format(x1, x2, x3, next_digit))
        results.append({'x1': x1.item(), 'x2': x2.item(), 'x3': x3.item(), 'prediction': next_digit.item()})
        
    with open('eval_results.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['x1', 'x2', 'x3', 'prediction'])
        writer.writeheader()
        for result in results:
            writer.writerow(result)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SRU for sequence prediction", add_help=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--input_size", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--nums_size", type=int, default=100, help="The maximum number contained in the sequence.")
    parser.add_argument("--train_split", type=int, default=0.8)
    args = parser.parse_args()
    print(args)
    main(args)