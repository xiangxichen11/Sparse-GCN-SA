import torch
import torch.nn.functional as F

def innerproduct(a, b):
    c = torch.mm(a, b.t())
    c = torch.div(c, torch.max(c))
    return c

def consinesimilar(u,v):
    intersection = torch.logical_and(u.bool(), v.bool()).sum(dim=1)
    union = torch.logical_or(u.bool(), v.bool()).sum(dim=1)
    jaccard = intersection.float() / union.float()
    return jaccard.view(-1, 1)  # Reshape to (n_samples, 1)
    return cosine_distance

def ouclideandist(a,b):
    nrow_a = a.size(0)
    nrow_b = b.size(0)
    c = torch.zeros(nrow_a, nrow_b)
    i = 0
    for row_b in b:
        mulrow_b = row_b.repeat(nrow_a, 1)
        c[:, i] = F.pairwise_distance(a, mulrow_b, p=2)
        i = i + 1
    # maxvaluec = torch.max(c, dim=1).values
    # maxvaluesc = maxvaluec.repeat(nrow_b, 1)
    # maxvaluesc = maxvaluesc.t()
    c = torch.div(c, torch.max(c))
    return c

def rbfsimilar(a,b):
    ouclidedist = ouclideandist(a, b)
    oudists = ouclidedist.pow(2)
    oudists = -1*oudists/0.005
    c = torch.exp(oudists)
    return c

def normalized(a):
    max_a = torch.max(a)
    min_a = torch.min(a)
    b = (a-min_a)/(max_a-min_a)
    return b

def gcnnormalization(a):
    sum_a = torch.sum(a, dim=1)
    sqrt_sum_a = torch.pow(sum_a, -0.5)
    diag_sqrt_sum_a = torch.diag(sqrt_sum_a)
    b = torch.mm(diag_sqrt_sum_a, a)
    c = torch.mm(b, diag_sqrt_sum_a)
    return c

def rbfkernel(feature,rand_feature,para):
    feature_size = feature.size(0)
    NUM = rand_feature.size(0)
    xx = torch.sum(torch.mul(feature, feature), dim=1).repeat(NUM, 1)
    yy = torch.sum(torch.mul(rand_feature, rand_feature), dim=1).reshape(NUM, 1).repeat(1, feature_size)
    yx = torch.mm(rand_feature, feature.t())
    dis = -(xx + yy - 2.0 * yx)
    k = torch.exp(dis / (2.0 * para))
    new_feature = k.t()
    return new_feature

