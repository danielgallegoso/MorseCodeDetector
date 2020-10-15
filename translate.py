# -*- coding: utf-8 -*-
# Standard imports
import cv2
import numpy as np;
import matplotlib.pyplot as plt
import cvxpy as cvx
from translator import morse_to_english, dic
from sklearn.cluster import KMeans

LAMBDA = .18

def plot_signal(nums, raw, smoothed):
    plt.subplot(211)
    plt.plot(nums, label='original')
    plt.plot(raw, label='raw')
    plt.legend()
    plt.title('Original Signal with Raw Smoothed Function')
    plt.subplot(212)
    plt.plot(smoothed)
    plt.title('Smoothed Signal')
    plt.show()

# trims leading and trailing 0's and adds fixed number of 0's to front and back
def trim(nums, n=10):
    index = np.where(nums == 1)
    start = np.min(index)
    end = np.max(index)
    return np.concatenate((-np.ones(n), nums[start:end+1], -np.ones(n)))


# Currently only parses individual words still haven't implemented capability to parse
# Haven't added the capability of parsing sentences.
def parse(nums, word_breaks=False):
    nums = trim(nums)
    # Smoothing the function using total variation
    # tune LAMBDA to a value that works empirically for data
    # increasing LAMBDA smooths the signal more
    # LAMBDA might be a function of the average length of dots and dashes
    # becuase faster message means shorter errors
    x = cvx.Variable(len(nums))
    obj = cvx.Minimize(cvx.norm(nums - x) + LAMBDA * sum(cvx.abs(x[:-1] - x[1:])))
    constraints = [x >= -1, x <= 1]
    prob = cvx.Problem(obj, constraints)
    prob.solve()
    smoothed = (np.array(x.value).flatten() >= 0).astype(int) * 2 - 1

    # Comment out to show visualization of smoothing
    # plot_signal(nums, np.array(x.value).flatten(), smoothed)

    # Generating dots and dashes
    loc = np.hstack((0,np.where(smoothed[:-1] != smoothed[1:])[0])) + 1
    loc[0] = 0
    intervals = np.add.reduceat(smoothed, loc)
    on = intervals[intervals > 0].reshape(-1, 1)
    off = intervals[intervals < 0].reshape(-1, 1)
    if on.shape[0] < 2 or off.shape[0] < 2:
        return None, False

    clusterOn = KMeans(n_clusters=2).fit(on).cluster_centers_
    mean_on = np.mean(clusterOn.flatten())
    mean_off_words = -2**63
    mean_off_letters = 0

    if word_breaks:
        clusterOff = KMeans(n_clusters=3).fit(off).cluster_centers_
        a, b, c = np.sort(clusterOff, axis=None)
        mean_off_words = np.mean([a, b])
        mean_off_letters = np.mean([b, c])
    else:
        clusterOff = KMeans(n_clusters=2).fit(off).cluster_centers_
        mean_off_letters = np.mean(clusterOff.flatten())

    # Creating the actual array
    intervals = list(intervals)
    new_intervals = []
    for q in intervals:
        if q < mean_off_words:
            new_intervals.append(-1)
        elif q < mean_off_letters:
            new_intervals.append(0)
        elif q < 0:
            continue
        elif q < mean_on:
            new_intervals.append(1)
        else:
            new_intervals.append(2)

    intervals = np.array(new_intervals)

    """np.place(intervals, intervals < mean_off_letters and intervals > mean_off_words, 0)
    intervals = intervals[intervals >= -1]
    np.place(intervals, np.logical_and(intervals>0, intervals<=mean_on), 1)
    np.place(intervals, intervals>1, 2)"""
    print intervals
    return intervals, True



def decode(nums):
    parsed, success = parse(nums)
    if not success: return
    result = morse_to_english(parsed, dic)
    if "$" in result:
        print "re-parsing"
        parsed, success = parse(nums, True)
        assert(success)
        result = morse_to_english(parsed, dic)

    return result



def main():
    nums = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    nums2 = np.array([-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    parsed, success = parse(nums2)
    msg = morse_to_english(parsed, dic)
    print "secret message was {0}".format(msg)

if __name__ == "__main__":
    main()
