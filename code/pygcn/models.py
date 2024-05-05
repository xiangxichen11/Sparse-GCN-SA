from layers import *

from bigbird_layer import BigBirdBlockSparseAttention


class GCNSA(nn.Module):

  def __init__(self, nfeat, nhid, nclass, dropout1, epsilon, n_samples, r):
    super(GCNSA, self).__init__()

    self.fc0 = torch.nn.Linear(nfeat, nhid)  # torch.nn.Linear
    self.structurel = adjcentmatrix(nhid, epsilon, n_samples, 2 * nhid,
                                    dropout1, r)

    self.hp = dotproduct(nhid * 8)
    self.fc1 = torch.nn.Linear(nhid * 8, nclass)
    self.dropout1 = dropout1
    self.layers = torch.nn.ModuleList(
        [EncoderLayer1(nclass, 1, dropout1) for _ in range(1)])
    self.modifiedt = EncoderLayer1(nhid, 4, dropout1)
    self.linear = torch.nn.Linear(nhid * 2, nhid * 2)

  def forward(self, feature, adj, K):
    y1 = F.relu(self.fc0(feature))
    y = F.dropout(y1, self.dropout1, training=self.training)
    x = self.modifiedt(y)
    z = torch.cat([x, y], dim=1)  # new feature 2p

    _, attention = self.structurel(y1)

    aaz = az = z
    for i in range(K - 1):
      az = torch.spmm(adj, az)  # 2p
      aaz = torch.spmm(adj, az)  # 2p

    z3 = torch.cat([aaz, z, az, aaz], dim=1)
    z3 = F.dropout(z3, self.dropout1, training=self.training)

    z3 = F.relu(self.hp(z3))
    z3 = F.dropout(z3, self.dropout1, training=self.training)

    z3 = self.fc1(z3)
    for layer in self.layers:
      z3 = layer(z3)

    z3 = F.log_softmax(z3, dim=1)

    return z3, attention


from yacs.config import CfgNode as CN


class GCNSAExphormer(nn.Module):

  def __init__(self, nfeat, nhid, nclass, dropout1, epsilon, n_samples, r):
    super(GCNSAExphormer, self).__init__()
    self.fc0 = torch.nn.Linear(nfeat, nhid)  # torch.nn.Linear
    self.sl_sparse = ExphormerAttention(
        in_dim=nhid,
        out_dim=nhid,
        num_heads=4,
        use_bias=True,
    )

    self.hp = dotproduct(nhid * 8)
    self.fc1 = torch.nn.Linear(nhid * 8, nclass)
    self.dropout1 = dropout1
    self.layers = torch.nn.ModuleList(
        [EncoderLayer1(nclass, 1, dropout1) for _ in range(1)])
    self.modifiedt = EncoderLayer1(nhid, 4, dropout1)
    self.linear = torch.nn.Linear(nhid * 2, nhid * 2)

  def forward(self, feature, adj, K):
    y1 = F.relu(self.fc0(feature))
    y = F.dropout(y1, self.dropout1, training=self.training)
    x = self.modifiedt(y)
    z = torch.cat([x, y], dim=1)  # new feature 2p

    attention = self.sl_sparse(y1)
    print(f"EXPHORMER ATTENTION SHAPE: {attention}")

    aaz = az = z
    for i in range(K - 1):
      az = torch.spmm(adj, az)  # 2p
      aaz = torch.spmm(adj, az)  # 2p

    z3 = torch.cat([aaz, z, az, aaz], dim=1)
    z3 = F.dropout(z3, self.dropout1, training=self.training)

    z3 = F.relu(self.hp(z3))
    z3 = F.dropout(z3, self.dropout1, training=self.training)

    z3 = self.fc1(z3)
    for layer in self.layers:
      z3 = layer(z3)

    z3 = F.log_softmax(z3, dim=1)

    return z3, attention
