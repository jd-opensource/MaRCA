import torch
import torch.nn as nn
from Dataset import SystemDataset
from system_model import SystemModel
import torch.utils.data as Data
import torch.optim as optim
import numpy as np
from utils import get_args
from tqdm import tqdm
import os

def l2_reg(params, lmbd=1e-5):
    reg_loss = None
    for param in params:
        if reg_loss is None:
            reg_loss = 0.5 * torch.sum(param**2)
        else:
            reg_loss = reg_loss + 0.5 * param.norm(2)**2
    return lmbd * reg_loss

def train(args):
    if args.root_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), './'))
    else:
        root_dir = os.path.abspath(args.root_dir)
    
    saved_modules_dir = os.path.join(root_dir, args.saved_modules_dir)
    os.makedirs(saved_modules_dir, exist_ok=True)
    
    model = SystemModel(args.input_shape, args.output_shape,
                       hidden_units=args.hidden_units,
                       activation=args.activation,
                       output_activation=args.output_activation)
    train_dataset = SystemDataset(root_dir=root_dir, args=args, mode='train')
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True)

    test_dataset = SystemDataset(root_dir=root_dir, args=args, mode='test')
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    losses = []
    valid_losses = []
    print('training start')
    for echo in tqdm(range(args.num_epochs)):
        train_loss = 0
        model.train()
        for i, (X, label) in enumerate(train_loader):
            X = X.unsqueeze(1)
            out = model(X)
            loss = loss_fn(out, label)
            loss += l2_reg(model.parameters())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss)

        losses.append(train_loss / len(train_loader))
        print("echo:" + ' ' + str(echo))
        print("training loss:" + ' ' + str(train_loss / len(train_loader)))
        if (echo + 1) % args.save_rate == 0:
            save_path = os.path.join(saved_modules_dir, f'model_params_{(echo+1) // args.save_rate}.pkl')
            torch.save(model.state_dict(), save_path)
            
        if (echo + 1) % args.save_rate == 0:
            with torch.no_grad():
                valid_loss = 0
                for i, (X, label) in enumerate(test_loader):
                    X = X.unsqueeze(1)
                    out = model(X)
                    loss = loss_fn(out, label)
                    valid_loss += float(loss)
                valid_losses.append(valid_loss / len(test_loader))
                print("validation loss: ", str(valid_loss / len(test_loader)))

if __name__ == '__main__':
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    args = get_args()
    train(args)



