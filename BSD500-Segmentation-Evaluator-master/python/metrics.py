"""Error Metrics
"""
import sys
import numpy as np
from skimage import measure


class Results:
    def __init__(self):
        self.sumSC = 0.0
        self.sumPRI = 0.0
        self.sumVI = 0.0

        self.count = 0

    def initialization(self):
        self.sumSC = 0.0
        self.sumPRI = 0.0
        self.sumVI = 0.0

        self.count = 0

    def update(self, pred, gt):
        """
        update the sum of all error metrics
        :param pred: prediction numpy array (H * W)
        :param gt: groundtruth numpy array (H * w)
        """
        # PRI and VI
        ri, vi = _PRIandVI(pred, gt)
        self.sumPRI += ri
        self.sumVI += vi

        # SC
        R, _ = _segmentation_cover(pred, gt)
        self.sumSC += R

        self.count += 1

        return ri, vi, R

    def get_results(self):
        if self.count == 0:
            print('No any results !!!')
            return self.sumSC, self.sumPRI, self.sumVI
        else:
            meanSC = self.sumSC / self.count
            meanPRI = self.sumPRI / self.count
            meanVI = self.sumVI / self.count
            return meanSC, meanPRI, meanVI


def _segmentation_cover(pred, gt):
    regionsGT = []
    regionsPred = []
    total_gt = 0

    cntR = 0
    sumR = 0
    cntP = 0
    sumP = 0

    propsGT = measure.regionprops(gt)
    for prop in propsGT:
        regionsGT.append(prop.area)
    regionsGT = np.array(regionsGT).reshape(-1, 1)
    total_gt = total_gt + np.max(gt)

    best_matchesGT = np.zeros((1, total_gt))

    matches = _match_segmentation(pred, gt)

    matchesPred = np.max(matches, axis=1).reshape(-1, 1)
    matchesGT = np.max(matches, axis=0).reshape(1, -1)

    propsPred = measure.regionprops(pred)
    for prop in propsPred:
        regionsPred.append(prop.area)
    regionsPred = np.array(regionsPred).reshape(-1, 1)

    for r in range(regionsPred.shape[0]):
        cntP += regionsPred[r] * matchesPred[r]
        sumP += regionsPred[r]

    for r in range(regionsGT.shape[0]):
        cntR += regionsGT[r] * matchesGT[:, r]
        sumR += regionsGT[r]

    best_matchesGT = np.maximum(best_matchesGT, matchesGT)

    R = cntR / (sumR + (sumR == 0))
    P = cntP / (sumP + (sumP == 0))

    return R[0], P[0]


def _match_segmentation(pred, gt):
    total_gt = np.max(gt)
    cnt = 0
    matches = np.zeros((total_gt, np.max(pred)))

    num1 = np.max(gt) + 1
    num2 = np.max(pred) + 1
    confcounts = np.zeros((num1, num2))

    # joint histogram
    sumim = 1 + gt + pred * num1

    hs, _ = np.histogram(sumim.flatten(), bins=np.linspace(1, num1*num2+1, num=num1*num2+1))
    hs = hs.reshape(confcounts.shape[1], confcounts.shape[0]).T

    confcounts = confcounts + hs
    accuracies = np.zeros((num1, num2))

    for j in range(0, num1):
        for i in range(0, num2):
            gtj = np.sum(confcounts[j, :])
            resj = np.sum(confcounts[:, i])
            gtjresj = confcounts[j, i]
            if gtj + resj - gtjresj:
                value = gtjresj / (gtj + resj - gtjresj)
                accuracies[j, i] = value
    matches[cnt:cnt + np.max(gt), :] = accuracies[1:, 1:]

    return matches.T


def _PRIandVI(pred, gt):
    (tx, ty) = pred.shape

    num1 = np.max(pred)
    num2 = np.max(gt)
    confcounts = np.zeros((int(num1) + 1, num2 + 1))

    for i in range(tx):
        for j in range(ty):
            u = pred[i, j]
            v = gt[i, j]

            confcounts[u, v] = confcounts[u, v] + 1

    RI = _rand_index(confcounts)
    VI = _variation_of_information(confcounts)

    return RI, VI


def _rand_index(n):
    N = np.sum(n)
    n_u = np.sum(n, axis=1)
    n_v = np.sum(n, axis=0)

    N_choose_2 = N * (N - 1) / 2

    ri = 1 - (np.sum(n_u * n_u) / 2 + np.sum(n_v * n_v) / 2 - np.sum(n * n)) / N_choose_2

    return ri


def _variation_of_information(n):
    N = np.sum(n)

    joint = n / N

    marginal_2 = np.sum(joint, axis=0)
    marginal_1 = np.sum(joint, axis=1)

    H1 = - np.sum(marginal_1 * np.log2(marginal_1 + (marginal_1 == 0)))
    H2 = - np.sum(marginal_2 * np.log2(marginal_2 + (marginal_2 == 0)))

    MI = np.sum(joint * _log2_quotient(joint, np.dot(marginal_1.reshape(-1, 1), marginal_2.reshape(1, -1))))

    vi = H1 + H2 - 2 * MI

    return vi


def _log2_quotient(A, B):
    lq = np.log2((A + ((A == 0) * B) + (B == 0)) / (B + (B == 0)))

    return lq
