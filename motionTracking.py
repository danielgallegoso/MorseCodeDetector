import numpy as np
from sklearn.cluster import KMeans

ELEMENT_NUM = 0
MAX_MOVEMENT_RATIO = 1.5
MAX_SIZE_RATIO = .5
ALPHA = .3
OFF_LENGTH = 10

class Element(object):
    def __init__(self, coord, rad, n):
        self.coord = coord
        self.rad = rad
        self.signal = np.append(np.zeros(n), 1).astype(int)
        self.plot = False
        global ELEMENT_NUM
        self.num = ELEMENT_NUM
        ELEMENT_NUM = ELEMENT_NUM + 1

    def similarity(self, elem):
        multiplier = 1
        # This could be a bad idea basically if it is off it will enlarge the
        # acceptance region because the light might have moved. It might be better
        # to calculate some sort of velocity vector and create acceptance region
        # elongated in that direction.
        multiplier += np.log(len(self.signal) - np.max(np.where(self.signal == 1)) + 1)
        if np.linalg.norm(self.coord - elem.coord) / self.rad <= MAX_MOVEMENT_RATIO*multiplier:
            if abs(self.rad - elem.rad) / self.rad <= MAX_SIZE_RATIO:
                return 1
        return 0

    def merge(self, elem):
        self.coord = ALPHA * elem.coord + (1 - ALPHA) * self.coord
        self.rad = ALPHA * elem.rad + (1 - ALPHA) * self.rad


def track(elements, keypoints):
    n = len(elements[0].signal) if len(elements) != 0 else 0
    sentinel = n > 256
    for keypoint in keypoints:
        (pt, rad) = keypoint
        newElem = Element(np.array(pt), rad / 2, n)
        isMerged = False
        for elem in elements:
            if len(elem.signal) == n and elem.similarity(newElem) == 1:
                elem.merge(newElem)
                elem.signal = np.append(elem.signal, 1)
                isMerged = True
                break
        if not isMerged:
            elements.append(newElem)
    for elem in elements:
        if len(elem.signal) == n:
            elem.signal = np.append(elem.signal, 0)
    elements = sorted(elements, key=lambda x: -np.sum(x.signal))
    return elements


# Could also attempt to denoise using Meanshift and delte the large cluster and
# small cluster and keep the middle ones. I don't know which approach might work
# best
def eliminate_noise(elements, minLength, force):
    n = len(elements[0].signal)
    averages = np.empty(0, dtype=float)
    indexes = np.empty(0, dtype=int)
    # print '-'*80
    for i in range(len(elements)):
        elem = elements[i]
        start = np.min(np.where(elem.signal == 1))
        if n - start >= minLength:
            elem.plot = True
            indexes = np.append(indexes, i)
            mean = np.mean(elem.signal[start:])
            # Attempt at filtering out lights that are permanently on
            # If it doesn't work then just use the commented out metric
            # It works well on data without permanent lights
            metric = 32 * (mean * (1 - mean))**5# + np.sum(elem.signal)/float(n)
            # metric = mean + np.sum(elem.signal)/float(n)
            averages = np.append(averages, metric)
            # print np.sum(elem.signal[start:]), metric
    # print '****************************************************', len(averages)
    if len(averages) >= 10 or (force and len(averages) > 2):
        cuttoff = np.mean(KMeans(n_clusters=2).fit(averages.reshape(-1,1)).cluster_centers_)
        # print indexes[averages < cuttoff], cuttoff
        for i in np.flipud(indexes[averages < cuttoff]):
            del elements[i]
    return elements


# Merges two signals if they are turning on and off at very similar times.
def merge_similar(elements, minLength, force):
    prob = np.random.rand()
    if prob > .05 and not force:
        return elements
    n = len(elements[0].signal)
    toDelete = []
    # print '-'*80
    for i in range(len(elements)):
        elem = elements[i]
        start = np.min(np.where(elem.signal == 1))
        for j in range(len(elements)):
            x = elements[j]
            startx = np.min(np.where(x.signal == 1))
            if startx >= start and (n - startx) >= minLength and i != j and i not in toDelete:
                equal = np.equal(elem.signal[startx:], x.signal[startx:]).all()
                corr = 0
                if not equal:
                    corr = np.corrcoef(elem.signal[startx:], x.signal[startx:])[0,1]
                # print np.corrcoef(elem.signal[startx:], x.signal[startx:])
                # print elem.signal[startx:].shape, x.signal[startx:].shape
                # print corr, i, j
                if corr > .8 or equal:
                    # Instead of combining both might want to simply delete one to reduce noise
                    temp = elem.signal[startx:] + x.signal[startx:] + np.random.rand() - .5
                    elem.signal[startx:] = (temp > 1).astype(int)
                    toDelete.append(j)
    for i in sorted(toDelete, reverse=True):
        if len(elements) > i:
            del elements[i]
    return elements


def prune_noise(elements, minLength=75, force=False):
    if len(elements) == 0:
        return elements
    if force:
        elements = merge_similar(elements, minLength, force)
    elements = eliminate_noise(elements, minLength, force)
    return elements
