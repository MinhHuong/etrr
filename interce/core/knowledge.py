"""
THE KNOWLEDGE COMPONENT OF INTERCE
"""

# python's native
from typing import List, Tuple, Union
import time

# time series clustering
from tslearn.clustering.kmeans import TimeSeriesKMeans

# data manipulation
import numpy as np

# my own module
from tools.utils import FeedbackItem, CandidateByExtractor, pad_sequences


def _distance(a: np.ndarray, b: np.ndarray):
    """
    Compute the distance between two multivariate time series

    Parameters
    ----------
    a
    b

    Returns
    -------

    """
    x, y = a.copy(), b.copy()
    D = a.shape[1]

    # first, pre-pad so that they have the same length
    if len(x) != len(y):
        maxlen = max(len(x), len(y))
        # if b is shorter, pad b (append 0 to the beginning)
        if len(x) > len(y):
            delta = maxlen - len(y)
            pad = np.zeros((delta, D))
            y = np.concatenate((pad, y), axis=0)
        # otherwise, pad a
        else:
            delta = maxlen - len(a)
            pad = np.zeros((delta, D))
            x = np.concatenate((pad, x), axis=0)

    # compute the distance between the padded arrays
    return np.sqrt(sum([(i[k] - j[k]) * (i[k] - j[k]) for i, j in zip(x, y) for k in range(D)]))


class LabelTracker:
    """
    The label tracker for each cluster, which holds
    the cycle label and the number of counts of each label
    """

    def __init__(self):
        """
        Initialize the tracker
        """
        self.counts = {}  # key = cycle label, value = label count

    def update(self, label, count=1):
        """
        Update the count of a label

        Parameters
        ----------
        label
        count

        Returns
        -------

        """
        if label in self.counts.keys():
            self.counts[label] += count
        else:
            self.counts[label] = count

    def get_major_label(self):
        """
        Return the label with the most count

        Returns
        -------

        """
        if len(self.counts) == 0:
            return ''

        return max(self.counts, key=self.counts.get)


class Knowledge:
    """
    Implement the knowledge of InterCE:
    - select and label cycles automatically
    """
    label_trackers: list[LabelTracker]

    def __init__(self, n_clusters: int, atts: List):
        """
        Initialize
        """
        self.k = n_clusters     # number of clusters
        self.atts = atts        # list of attributes
        self.clusters = {}      # clusters organized by label, given by the human expert

        # self.clusters = []  # hold the clusters learned from the feedback
        # self.label_trackers = []  # hold the label trackers of each cluster

        # time-related measurement
        self.time_select = 0
        self.time_update = 0

    #region new approach (simple dictionary, no learning per se)
    def select_cycles(self, candidates: dict[str, CandidateByExtractor]) \
            -> Tuple[str, Union[None, CandidateByExtractor]]:
        """
        Select and label cycles from an input X

        Parameters
        ----------
        candidates: dict
            Indexed by extractor name, each value is a CandidatePerExtractor

        Returns
        -------

        """
        # FIXED start the timer
        start = time.perf_counter()

        # if no clusters yet, do nothing, return empty list
        if len(self.clusters) == 0:
            return '', None

        # if there are no candidates, do nothing and return None
        if sum([len(cand.markers) for ext, cand in candidates.items()]) == 0:
            return '', None

        # compute the distance between the candidate of each extractor to the feedback clusters
        distances = {}
        for ext, candidate in candidates.items():
            if len(candidate.markers) == 0:
                distances[ext] = None
            else:
                distances[ext] = 0
                for mk in candidate.markers:
                    distances[ext] += self._distance_from_closest_cluster(mk.data)

        # the extractor that has the smallest sum of distance is the best one
        # best_ext: str = min(distances, key=distances.get)
        # a bit cumbersome but more robust
        best_ext = ''
        min_dist = None
        for ext, dist in distances.items():
            if dist is None:
                continue
            if min_dist is None:
                min_dist = dist
                best_ext = ext
            elif dist < min_dist:
                min_dist = dist
                best_ext = ext
        if best_ext == '':
            return '', None

        # label the chosen cycles
        for mk in candidates[best_ext].markers:
            mlabel = self._find_closest_cluster(mk.data)
            mk.label = mlabel

        # FIXED stop the timer
        end = time.perf_counter()
        self.time_select += (end - start)

        # return the best cycles and their label
        return best_ext, candidates[best_ext]

    def _distance_from_closest_cluster(self, cdata) -> float:
        """
        Find the cluster closest to a cycle

        Parameters
        ----------
        cdata
            Data in the cycle

        Returns
        -------

        """
        if len(self.clusters) == 0:
            return -1.0

        return min([_distance(cent, cdata) for _, cent in self.clusters.items()])

    def _find_closest_cluster(self, cdata) -> str:
        """
        Find the closest cluster to a cycle, return the label of this cluster

        Parameters
        ----------
        cdata

        Returns
        -------
        str
        """
        if len(self.clusters) == 0:
            return ''

        distances = {cent_label: _distance(cent, cdata) for cent_label, cent in self.clusters.items()}
        return min(distances, key=distances.get)

    def update(self, data: List[FeedbackItem]):
        """
        Update the knowledge on new feedback

        Parameters
        ----------
        data

        Returns
        -------

        """
        # FIXED start the timer
        start = time.perf_counter()

        # organize the feedback into dictionary indexed by feedback label
        _clusters = {}
        for fb in data:
            if fb.mlabel not in _clusters.keys():
                _clusters[fb.mlabel] = [fb.mdata]
            else:
                _clusters[fb.mlabel].append(fb.mdata)

        # add the existing clusters in as well
        for clabel, cdata in self.clusters.items():
            if clabel not in _clusters.keys():
                _clusters[clabel] = [cdata]
            else:
                _clusters[clabel].append(cdata)

        # pad the cycles and compute just the centroid
        for clabel, cdata in _clusters.items():
            padded = pad_sequences(cdata, dim=len(self.atts))
            cent = np.mean(padded, axis=0)  # compute the averaged centroid
            nzero = [i for i, r in enumerate(cent) if (r != 0).any()]  # cut out trailing 0's
            self.clusters[clabel] = cent[nzero[0]:nzero[-1] + 1]

        # FIXED stop the timer
        end = time.perf_counter()
        self.time_update += (end - start)
    #endregion
