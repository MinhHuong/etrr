"""
MEMORY OF INTERCE
"""
import sys
from typing import List, Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import time

from tools.utils import Query


def _motif_distance(a, b) -> float:
    """
    Compute the distance between 2 motifs using fast Dynamic Time Warping (https://github.com/slaypni/fastdtw)

    Parameters
    ----------
    a: np.ndarray
        A univariate time series
    b: np.ndarray
        A univaraite time series

    Returns
    -------
    float
        The distance between two time series
    """
    # scale the data before computing the distance (use min-max scaling)
    a_ = (a - np.min(a)) / (np.max(a) - np.min(a))
    b_ = (b - np.min(b)) / (np.max(b) - np.min(b))
    # compute the distance (DTW)
    dist, _ = fastdtw(a_, b_, dist=euclidean)
    return dist


class Memory:
    """
    Manage the memory of InterCE:
    - whether an input has been seen
    - whether an input has its query processed
    - whether a query is redundant
    """
    intra_distances: list[list]

    def __init__(self, sim_thres, dbconn):
        """Initialize the memory"""
        self.sim_thres = sim_thres      # similarity threshold for data motifs
        self.motifs = []                # list of all motifs (stored in memory for now)
        self.motif_ids = []             # list of the ID's associated to each motif
        self.intra_distances = []       # the intra-distances of one motif to all the others
        self.dbconn = dbconn            # connection to the database
        self.length_ratio_thres = 0.9   # tolerance threshold on the length ratio of two motifs
        self.dim = 0                    # dimension of the motif
        self.n_motifs = 0               # number of motifs (sync issue when metric thread is running)

        # timer related measurement
        self.time_motifs = 0      # time to find similar motifs (accumulated)

    def has_seen_data(self, X: pd.DataFrame, X_id):
        """
        Check whether an input X has been seen

        Parameters
        ----------
        X
        X_id

        Returns
        -------
        (bool, str)
            True if a motif similar to X has been processed before, False otherwise
            Also return the ID of the most similar motif to X, if any
        """
        # FIXED record the time
        start = time.perf_counter()

        # convert to numpy array
        x = X.to_numpy(copy=True)

        # dimension of the data
        self.dim = x.shape[1]

        # cut out trailing 0's at the beginning and end of the series
        non0 = [i for i, step in enumerate(x) if (step != 0).any()]
        x = x[non0[0]:non0[-1]+1]

        # if no motifs yet, add x as a new one
        if len(self.motifs) == 0:
            self._add_new_motif(x, X_id)
            has_seen, motif_id = False, ''
        # if only one motif, compare the length ratio, if the length is close, just...deduce it's the same motif
        elif len(self.motifs) == 1:
            length_ratio = len(x) / len(self.motifs[0]) if len(x) < len(self.motifs[0]) \
                else len(self.motifs[0]) / len(x)
            # if two series have quite dissimilar length, consider x a new motif
            if length_ratio < self.length_ratio_thres:
                self._add_new_motif(x, X_id)
                has_seen, motif_id = False, ''
            else:
                # otherwise, X is a redundant motif
                has_seen, motif_id = True, self.motif_ids[0]
        # if more than 1 motif
        else:
            # compute the distance only if the two series have quite similar length (based on length_ratio_threshold)
            # if the length is too dissimilar, just assign it np.nan and ignore it
            distances = [np.nan] * len(self.motifs)
            for im, m in enumerate(self.motifs):
                length_ratio = len(x) / len(m) if len(x) < len(m) else len(m) / len(x)
                # only compute the distance if two series are potentially close
                if length_ratio >= self.length_ratio_thres:
                    distances[im] = np.mean([_motif_distance(m[:, j], x[:, j]) for j in range(self.dim)])
            # if all distances are Nan, it means all motifs are dissimilar to this one --> add new motif
            if all(np.isnan(distances)):
                self._add_new_motif(x, X_id)
                has_seen, motif_id = False, ''
            else:
                # check if x is a new motif or if it's already been seen
                idx = np.nanargmin(distances)  # index of the motif that is closest to X
                intra_dist = self.intra_distances[idx]
                # only add new motif if the distance from X to the closest motif is superior to all the intra-distances
                if all(distances[idx] > intra_dist):
                    self._add_new_motif(x, X_id)
                    has_seen, motif_id = False, ''
                else:
                    has_seen, motif_id = True, self.motif_ids[idx]

        # FIXED end the timer
        end = time.perf_counter()
        self.time_motifs += (end - start)

        # return the final result
        return has_seen, motif_id

    def _add_new_motif(self, motif, motif_id):
        """
        Add a new motif

        Parameters
        ----------
        motif
        motif_id

        Returns
        -------

        """
        # add the motif in the list
        self.motifs.append(motif)
        self.motif_ids.append(motif_id)

        # update the intra-distance of existing motifs (add one more element in) and of the new one
        new_intradist = []
        for im in range(len(self.motifs[:-1])):
            motif_dist = _motif_distance(self.motifs[im], motif)
            self.intra_distances[im].append(motif_dist)
            new_intradist.append(motif_dist)

        # add the new intra-distance entry for the new motif in self.intra_distances
        self.intra_distances.append(new_intradist)

        # save the motif in the database
        self.dbconn.save_motif(motif_id=motif_id, motif_data=motif.tolist())

        # update the number of motif
        self.n_motifs += 1

    def has_processed_query_of(self, X_id):
        """
        Check whether an input X_id has its query processed before

        Parameters
        ----------
        X_id

        Returns
        -------

        """
        return self.dbconn.is_query_processed(X_id)

    def is_redundant_query(self, query: Query):
        """
        Check whether a query is redundant

        Returns
        -------

        """
        return False

    def create_query(self, X_id, X, candidates):
        """
        Create a query for an input X

        Parameters
        ----------
        X_id
        X
        candidates

        Returns
        -------

        """
        return Query(X_id, X, candidates)
