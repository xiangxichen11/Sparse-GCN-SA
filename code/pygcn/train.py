from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import glob
import torch
import torch.nn.functional as F
import torch.optim as optim

from sample import Sampler
from utils import accuracy, randomsplit, acquiretvt, ratio_homo
from models import GCNSA, GCNSAExphormer
import utils_data
import os

testacc = torch.zeros(100)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cora', help='dataset.')
parser.add_argument('--trainsplit',
                    type=float,
                    default=0.48,
                    help='trainsplit.')
parser.add_argument('--K',
                    type=int,
                    default=3,
                    help='Feature aggregation times.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs',
                    type=int,
                    default=800,
                    help='Number of epochs to train.')
parser.add_argument('--lr',
                    type=float,
                    default=0.03,
                    help='Initial learning rate.')
parser.add_argument('--wd',
                    type=float,
                    default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hd',
                    type=int,
                    default=48,
                    help='Number of hidden units.')
parser.add_argument('--dropout',
                    type=float,
                    default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epsilon',
                    type=float,
                    default=0.9,
                    help='threshold value of similarity')
parser.add_argument('--r', type=int, default=3, help='minimum of new neighbors')
parser.add_argument('--patience', type=int, default=200, help='Patience')
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.dataset in {'cora', 'citeseer', 'pubmed'}:
  sampler = Sampler(args.dataset, "../data/", "full")
  labels = sampler.get_label_and_idxes(True)
  (adj, features) = sampler.randomedge_sampler(percent=1.0,
                                               normalization="AugNormAdj",
                                               cuda=True)
else:
  adj, features, labels = utils_data.load_data(args.dataset, args.trainsplit)

n = adj.size(0)
n_samples = adj.size(0)
n_features = features.size(1)

testacc = np.zeros(10)
for j in range(10):
  idx_train, idx_val, idx_test = acquiretvt(args.dataset, args.trainsplit,
                                            labels, j)
  model = GCNSAExphormer(nfeat=features.shape[1],
                         nhid=args.hd,
                         nclass=labels.max().item() + 1,
                         dropout1=args.dropout,
                         epsilon=args.epsilon,
                         n_samples=n_samples,
                         r=args.r)

  torch.cuda.manual_seed(args.seed)
  model.cpu()
  features = features.cpu()
  adj = adj.cpu()
  labels = labels.cpu()
  idx_train = idx_train.cpu()
  idx_val = idx_val.cpu()
  idx_test = idx_test.cpu()

  optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

  def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output, _ = model(features, adj, args.K)
    loss_train = F.nll_loss(
        output[idx_train],
        labels[idx_train])  #torch.nn.functional.cross_entropy
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    model.eval()
    output, _ = model(features, adj, args.K)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    if epoch % 100 == 0:
      print('Epoch: {:04d}'.format(epoch),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t))
    return loss_val.data.item()

  def test():
    model.eval()
    output, _ = model(features, adj, args.K)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])  #F.nll_loss
    acc_test = accuracy(output[idx_test], labels[idx_test])

    print("Test set results:", "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()), "dp1:", args.dropout,
          "epsilon:", args.epsilon)
    return acc_test

  # Train model

  t_total = time.time()
  loss_values = []
  bad_counter = 0
  best = args.epochs + 1
  best_epoch = 0

  for epoch in range(args.epochs):
    loss_values.append(train(epoch))
    torch.save(model.state_dict(), '{}.pkl'.format(epoch))

    if loss_values[-1] < best:
      best = loss_values[-1]
      best_epoch = epoch
      bad_counter = 0
    else:
      bad_counter += 1

    if bad_counter == args.patience:
      break

    files = glob.glob('*.pkl')
    for file in files:
      epoch_nb = int(file.split('.')[0])
      if epoch_nb < best_epoch:
        os.remove(file)

  files = glob.glob('*.pkl')
  for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
      os.remove(file)

  print("Optimization Finished!")
  print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

  # Restore best model
  print('Loading {}th epoch'.format(best_epoch))
  model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

  # Testing
  testacc[j] = test()
  print("testacc:", testacc[j])

print("testacc:", testacc)
print("mean(testacc):", np.mean(testacc))
print("std(testacc):", np.std(testacc))
