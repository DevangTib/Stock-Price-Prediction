import os
import time
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import R_GATModel
from utils import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--lr',     default=5e-5, type=float)
parser.add_argument('--wd',     default=1e-4, type=float)
parser.add_argument('--hidden', default=64,  type=int)
parser.add_argument('--dropout',default=0.38, type=float)
parser.add_argument('--seed',   default=14,  type=int)
parser.add_argument('--cuda',   action='store_true')
parser.add_argument('--patience', default=20, type=int)
parser.add_argument('--num_relations', default=1, type=int)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
adj = load_data().to(device)
S = adj.size(0)
edge_index = adj.nonzero().t().contiguous()
edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=device)

model = R_GATModel(3, args.hidden, 2, args.dropout, None, S, args.num_relations).to(device)
opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = ReduceLROnPlateau(opt, 'max', factor=0.5, patience=10)
loss_fn = torch.nn.CrossEntropyLoss()

# Data loader
def load_sample(i, mode='train'):
    base = 'train' if mode=='train' else 'test'
    price = torch.from_numpy(np.load(f"./Data/{base}_price/{i:010d}.npy")).float().to(device)
    text  = torch.from_numpy(np.load(f"./Data/{base}_text/{i:010d}.npy")).float().to(device)
    label = torch.from_numpy(np.load(f"./Data/{base}_label/{i:010d}.npy")).long().to(device)
    return text, price, label

def evaluate():
    model.eval()
    preds, trues = [], []
    total_loss = 0
    test_files = len(os.listdir("./Data/test_price"))
    with torch.no_grad():
        for i in range(test_files):
            text, price, label = load_sample(i, mode='test')
            out = model(text, price, edge_index, edge_type)
            total_loss += loss_fn(out, label).item()
            preds.append(out.argmax(dim=1).cpu().numpy())
            trues.append(label.cpu().numpy())
    preds = np.concatenate(preds).ravel()
    trues = np.concatenate(trues).ravel()
    p,r,f,_ = precision_recall_fscore_support(trues, preds, average='binary')
    mcc = matthews_corrcoef(trues, preds)
    acc = (preds==trues).mean()
    return total_loss/test_files, p, r, f, mcc, acc

best_f1, patience = 0, 0
start = time.time()
for ep in range(1, args.epochs+1):
    model.train()
    # single-sample training
    train_files = len(os.listdir("./Data/train_price"))
    idx = random.randrange(train_files)
    text, price, label = load_sample(idx, mode='train')

    opt.zero_grad()
    out = model(text, price, edge_index, edge_type)
    loss = loss_fn(out, label)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    # evaluate every 10 epochs
    if ep % 10 == 0:
        val_loss, precision, recall, f1, mcc, acc = evaluate()
        scheduler.step(f1)
        print(f"Epoch {ep}: Val Loss={val_loss:.4f}, F1={f1:.4f}, Acc={acc:.4f}")
        # save best model
        if f1 > best_f1:
            best_f1 = f1
            patience = 0
            save_path = f"best_model.pth"
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'f1': best_f1
            }, save_path)
            print(f"Saved new best model to {save_path}")
        else:
            patience += 1
        # early stopping
        if patience >= args.patience:
            print(f"Early stopping at epoch {ep}")
            break

total_time = time.time() - start
print(f"Training completed in {total_time:.1f}s, best F1={best_f1:.4f}")
