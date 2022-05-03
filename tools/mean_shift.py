import torch
from tqdm.auto import trange
import numpy as np


def mean_shift_step(data, bandwidth=50, kernel='flat', batch_size=1000):
    data = data.cuda()
    dist = []
    norm_sq = data.norm(dim=1).pow(2)
    shift = []
    movement = -1
    for start in trange(0, data.size(0), batch_size, desc='mean shift inner'):
        end = min(data.size(0), start + batch_size)
        batch = data[start:end]
        d_sq = norm_sq[start:end].view(-1, 1) + norm_sq.view(1, -1) - 2 * batch @ data.transpose(0, 1)
        if kernel == 'flat':
            weight = d_sq.le(bandwidth ** 2).float()
        elif kernel == 'rbf':
            weight = d_sq.mul(- 0.5 / bandwidth**2).exp()
        else:
            raise NotImplementedError
        weight /= weight.sum(dim=1, keepdim=True)
        s = weight @ data
        movement = max((s - batch).norm(dim=1).max().item(), movement)
        shift.append(s.cpu())
        
    shift = torch.cat(shift, dim=0)
    return shift, movement


def get_centroid(data, bandwidth, batch_size=1000):
    data = data.cuda()
    counts = []
    nbrs = []
    norm_sq = data.norm(dim=1).pow_(2)
    movement = -1
    unique = torch.ones(data.shape[0], dtype=torch.bool)
    for start in range(0, data.size(0), batch_size):
        end = min(data.size(0), start + batch_size)
        batch = data[start:end]
        d_sq = norm_sq[start:end].view(-1, 1) + norm_sq.view(1, -1) - 2 * batch @ data.transpose(0, 1)
        nbr = d_sq.le(bandwidth ** 2)
        counts.append(nbr.long().sum(dim=1).cpu())
        nbrs.append(nbr.cpu())
        
    counts = torch.cat(counts, dim=0)
    nbrs = torch.cat(nbrs)
    _, indices = counts.sort(descending=True)
    for i in indices:
        i = i.item()
        if unique[i]:
            unique[nbrs[i]] = False
            unique[i] = True
    centroids = data[unique].cpu()
    return centroids, unique


class MeanShift(object):
    def __init__(self, bandwidth, kernel='flat', threshold=0.01):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.threshold = threshold
        self._centroids = None

    @property
    def centroids(self):
        return self._centroids

    def __repr__(self) -> str:
        return f"MeanShift(bandwidth={self.bandwidth}, kernel={self.kernel}, threshold={self.threshold})"

    def fit(self, data, batch_size=1000, max_iter=50):
        shift = data
        for i in trange(max_iter, desc='fitting mean shift'):
            shift, movement = mean_shift_step(shift, bandwidth=self.bandwidth, kernel=self.kernel, batch_size=batch_size)
            if movement < self.bandwidth * self.threshold:
                break
        self._centroids, _ = get_centroid(data, self.bandwidth, batch_size)
        
    def transform(self, data, batch_size=1000):
        centroids = self.centroids.cuda()
        norm_sq = centroids.norm(dim=1).pow_(2)
        clusters = []
        for start in range(0, data.size(0), batch_size):
            end = min(data.size(0), start + batch_size)
            batch = data[start:end].cuda()
            d_sq = batch.norm(dim=1).pow_(2).view(-1, 1) + norm_sq.view(1, -1) - 2 * batch @ centroids.transpose(0, 1)
            clusters.append(d_sq.argmax(dim=1).cpu())

        clusters = torch.cat(clusters, dim=0)
        return clusters

    def fit_transform(self, data, batch_size=1000, max_iter=50):
        self.fit(data, batch_size, max_iter)
        return self.transform(data, batch_size)