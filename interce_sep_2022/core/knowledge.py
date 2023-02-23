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
        # FIXME buggy if an ext has not candidate then its distance is automatically 0!!!
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
        # print('\tIn KNOWLEDGE: distances =', distances, '| best:', best_ext)
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

    #region old approach (using clustering)
    def select_cycles_old(self, candidates: dict[str, CandidateByExtractor]) \
            -> Tuple[str, Union[None, CandidateByExtractor]]:
        """
        ### OLD VERSION ###
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
        distances: dict[str, float] = {}
        for ext, candidate in candidates.items():
            distances[ext] = 0
            for mk in candidate.markers:
                distances[ext] += self._distance_from_closest_cluster(mk.data)

        # the extractor that has the smallest sum of distance is the best one
        best_ext: str = min(distances, key=distances.get)
        
        # label the chosen cycles
        for mk in candidates[best_ext].markers:
            c = self._find_closest_cluster(mk.data)  # return the INDEX of the cluster in self.clusters
            # get the predicted label of the cycle
            label = self.label_trackers[c].get_major_label()
            if label != '':
                mk.label = label

        # FIXED stop the timer
        end = time.perf_counter()
        self.time_select += (end - start)

        return best_ext, candidates[best_ext]

    def _distance_from_closest_cluster_old(self, cdata) -> float:
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

        return min([_distance(c, cdata) for c in self.clusters])

    def _find_closest_cluster_old(self, cdata) -> int:
        """
        Find the closest cluster to a cycle, return the index of this cluster in self.clusters

        Parameters
        ----------
        cdata

        Returns
        -------

        """
        if len(self.clusters) == 0:
            return -1

        distances = [_distance(c, cdata) for c in self.clusters]
        return min(range(len(distances)), key=lambda x: distances[x])

    def _update_old(self, data: List[FeedbackItem]):
        """
        ### OLD VERSION ###
        Update itself on new feedback data

        Parameters
        ----------
        data: List[FeedbackItem]

        Returns
        -------

        """
        # FIXED start the timer
        start = time.perf_counter()

        # re-discover the clusters on all the feedback
        cyc_data = [fb.mdata for fb in data]
        cyc_data = pad_sequences(cyc_data, dim=len(self.atts))
        # can change to 'dtw' to be more accurate
        model = TimeSeriesKMeans(n_clusters=self.k, metric='euclidean', random_state=420)
        model.fit(cyc_data)

        # update the label tracker in each cluster
        self.label_trackers = []  # reset the tracker
        for y in np.unique(model.labels_):
            tracker = LabelTracker()
            # get the indices of the cycles in this cluster
            indices = np.where(model.labels_ == y)[0]
            # update the count of each label in the label tracker
            for idx in indices:
                if data[idx].mlabel != '':
                    tracker.update(label=data[idx].mlabel)
            self.label_trackers.append(tracker)

        # trim the trailing 0's at the beginning and end of each cluster centroid
        self.clusters = []
        for cent in model.cluster_centers_:
            nzero = [i for i, r in enumerate(cent) if (r != 0).any()]
            self.clusters.append(cent[nzero[0]:nzero[-1]+1])

        # FIXED stop the timer
        end = time.perf_counter()
        self.time_update += (end - start)
    #endregion

# if __name__ == '__main__':
#     from database.dbmanager import DBConnection
#
#     dbconn = DBConnection(dbname='interce_r2b', user='postgres', pwd='123456')
#     knowledge = Knowledge(n_clusters=3)
#     fb_data = dbconn.get_feedback_data()
#     knowledge.update(fb_data)
