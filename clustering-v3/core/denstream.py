"""
DenStream: Density-based clustering algorithm on evolving data streams.

Implementation of the algorithm proposed by Cao et al. (2006) with some modifications
"""

# python's native libraries
import itertools
import math
from enum import Enum, IntEnum
from time import perf_counter

# data manipulation
from typing import Union, List, Callable
import numpy as np
from scipy.stats import t as student_t

# custom modules
from utils import TIMEUNIT
from feeder.data_point import Instance

# database connector
from database.mongo_conn import MongoDBConnector


def euclidean_distance(p1, p2):
    """
    Compute the euclidean distance between 2 points

    Parameters
    ----------
    p1
    p2

    Returns
    -------

    """
    dist = [(a - b) * (a - b) for a, b in zip(p1, p2)]
    return math.sqrt(sum(dist))


def manhattan_distance(p1, p2):
    """
    Compute the manhattan distance between 2 points

    Parameters
    ----------
    p1
    p2

    Returns
    -------

    """
    dist = sum([abs(a - b) for a, b in zip(p1, p2)])
    return dist


def _is_close_to_0(n):
    """
    Check whether a positive number is very close to 0 (1e-6)

    Parameters
    ----------
    n
        A positive number

    Returns
    -------

    """
    return n < 1e-5


class MCType(IntEnum):
    """Type of the micro cluster"""
    PMC = 1     # potential micro cluster
    OMC = 2     # outlier micro cluster


class MCLabel(Enum):
    """Label of the micro cluster"""
    GOOD = 'good'           # reference cluster i.e. cluster of good health
    ANOMALY = 'anomaly'     # cluster of an anomaly


class MicroCluster:
    """A micro-cluster that maintain the summary statistics for each system"""
    id_mc = itertools.count()

    def __init__(self, dim: int, mc_type: IntEnum, t0: float, time_unit: TIMEUNIT, distance: Callable,
                 min_points_system: int, decay_factor: float):
        """
        Initialize a micro cluster

        Parameters
        ----------
        dim: int
            Dimension of the data
        mc_type: IntEnum
            Whether this MC is an outlier MC or a potential MC
        t0: float
            Creation time
        time_unit
            Unit of time
        distance: Callable
            A distance function
        min_points_system: int
            Minimum points a system must have to be considered official
        decay_factor: float
            Decay factor (lambda)
        """
        # unique ID of this micro-cluster (auto-increment)
        self.id = next(MicroCluster.id_mc)

        # ID of the clusters that have been merged in this one
        self.id_merged = []

        # linear sum
        self.cf1 = {}  # linear sum, separated by system, i.e., cf1[sys_id] = np.zeros(dim)
        self.CF1 = np.zeros(dim)  # total linear sum, computed from the cf1 of all systems

        # squared sum
        self.cf2 = {}  # squared sum, separated by system, i.e., cf2[sys_id] = np.zeros(dim)
        self.CF2 = np.zeros(dim)  # total squared sum, computed from the cf2 of all systems

        # weight (decayed from the number of points)
        self.weight = {}  # weight of the MC, separated by system, i.e., w[sys_id] = a number
        self.W = 0  # total weight of the MC, computed from the weight of all systems

        # time-related info
        self.t0 = t0  # creation time
        self.last_upd = {}  # last update time, separated by system, i.e., last_upd[sys_id] = a number
        self.time_unit = time_unit

        # size
        self.n = {}  # number of data points by system, i.e., n[sys_id] = a number
        self.N = 0  # total number of data points from all systems

        # local statistic (center, radius, variance)
        self.dim = dim  # the dimension of the points
        self.center = np.zeros(dim)  # center of the MC = CF1 / N, where CF1 = sum of all cf1's, N = sum of all n's
        self.radius = 0  # radius of the MC, CF2 = sum of all cf2's, CF1 = sum of all cf1's, W = sum of all w's

        # FIXED maximal distance as a kind of "soft" radius
        self.max_dist = -1
        self.sys_max_dist = ''  # the ID of the system that is currently holding the maximal distance
        self.distance = distance

        # FIXED mask for systems that have yet to be considered "official" (list of system ID's)
        self.masked_systems = []
        self.min_points_system = min_points_system

        # FIXED add decayed linear sum of timestamps, allow to compute anomaly score
        self.decayed_ls_tsp = {}  # linear sum of timestmaps with decay factor 2^(-lambda * t) x T_j
        self.decay_factor = decay_factor

        # type of the MC (OMC or PMC)
        self.mc_type = mc_type  # type of this MC

        # member points indexed by system ID (members[sys_id] = list of tuples: (instance ID, instance tsp))
        self.members = {}

        # label of this cluster ('good' or 'anomaly')
        self.label = ''

        # anomaly degree (different from the decaying weight of a micro cluster)
        self.anomaly_degree = 0

        # used to compute the standard deviation of the radius incrementally
        self.m2 = 0
        self.mean_radius = 0
        self.std_radius = 0

    def add(self, inst: Instance):
        """
        Add a new point in this cluster and recompute the statistics

        Parameters
        ----------
        inst: Instance
            A new data instance

        Returns
        -------
        None
        """
        inst_ = inst.copy()

        # update the members
        self._update_members(inst_)

        # decide whether to mask or unmask this system
        # case 1: if this system has recently produced a few points in this cluster and is not yet masked -> mask it
        if len(self.members[inst_.sys_id]) < self.min_points_system \
                and inst_.sys_id not in self.masked_systems:
            self.masked_systems.append(inst_.sys_id)
        # case 2: if this system has just exceeded the limit and is still masked -> unmask it
        if len(self.members[inst_.sys_id]) >= self.min_points_system \
                and inst_.sys_id in self.masked_systems:
            self.masked_systems.remove(inst_.sys_id)
        # do nothing if the system produces a few points under the limit and is already masked

        # update the summary statistics
        self._update_weight(inst_)
        self._update_cf1(inst_)
        self._update_cf2(inst_)
        self._update_size(inst_)
        self._update_center()
        self._update_radius()

        # FIXED update the maximal distance
        self._update_max_distance(inst_)

        # FIXED update decayed linear sum of timestamp
        # FIXME result overflow because it's too large...
        # FIXME solution: instead of taking the absolute timestamp, substract T_o some landmark point
        # FIXME e.g., the time where a maintenance has taken place
        # FIXME t_m = T_m - T_0, t_j = T_j - T_0 => t_m - t_j = (T_m - T_0) - (T_j - T_0) = (T_m - T_j)
        # FIXME but this wont fix the problem completely as T_m and T_j -> inf (the choice is to reset T0 regularly)
        # self._update_linear_sum_timestamp(inst_)

        # FIXED update the timestamp per system
        # self.last_upd[inst_.sys_id] = inst_.tsp
        self._update_last_timestamp(inst_)

    def _update_cf1(self, inst: Instance):
        """
        Update the linear sum when a new point is added

        Parameters
        ----------
        inst: Instance

        Returns
        -------
        None
        """
        if inst.sys_id in self.cf1.keys():
            self.cf1[inst.sys_id] += inst.data
        else:
            self.cf1[inst.sys_id] = inst.data
        # only update if the system is not masked
        if inst.sys_id not in self.masked_systems:
            self.CF1 += inst.data

    def _update_cf2(self, inst: Instance):
        """
         Update the squared sum when a new point is added

        Parameters
        ----------
        inst: Instance

        Returns
        -------
        None
        """
        if inst.sys_id in self.cf2.keys():
            self.cf2[inst.sys_id] += (inst.data * inst.data)
        else:
            self.cf2[inst.sys_id] = inst.data * inst.data
        # only update if the system is not masked
        if inst.sys_id not in self.masked_systems:
            self.CF2 += (inst.data * inst.data)

    def _update_weight(self, inst: Instance):
        """
        Update the weight

        Parameters
        ----------
        inst: Instance

        Returns
        -------
        None
        """
        if inst.sys_id in self.weight.keys():
            self.weight[inst.sys_id] += 1
        else:
            self.weight[inst.sys_id] = 1
        # only update if the system is not masked
        if inst.sys_id not in self.masked_systems:
            self.W += 1

    def _update_size(self, inst: Instance):
        """
        Update the number of points in this micro cluster

        Parameters
        ----------
        inst: Instance

        Returns
        -------
        None
        """
        if inst.sys_id in self.n.keys():
            self.n[inst.sys_id] += 1
        else:
            self.n[inst.sys_id] = 1
        # only update if the system is not masked
        if inst.sys_id not in self.masked_systems:
            self.N += 1

    def _update_members(self, inst: Instance):
        """
        Update the members of the concerned system

        Parameters
        ----------
        inst: Instance

        Returns
        -------
        None
        """
        if inst.sys_id in self.members.keys():
            # self.members[inst.sys_id].append((inst.id, inst.tsp))
            prev_step = self.members[inst.sys_id][-1][1]  # get the previous step of this system
            self.members[inst.sys_id].append((inst.id, prev_step + 1))
        else:
            # self.members[inst.sys_id] = [(inst.id, inst.tsp)]
            self.members[inst.sys_id] = [(inst.id, self.t0)]

    def _update_center(self):
        """
        Update the center of this micro-cluster

        Returns
        -------

        """
        if _is_close_to_0(self.W):  # prevent division by 0 which wrecks the computation
            self.center = np.zeros(self.dim)
        else:
            self.center = self.CF1 / self.W

    def _update_radius(self):
        """
        Update the radius of this micro-cluster

        Returns
        -------

        """
        # TODO do we need to change it? A radius of 0 is just really weird
        if _is_close_to_0(self.W):   # prevent division by 0 which wrecks the computation
            self.radius = 0
        else:
            self.radius = self._compute_radius(ls=self.CF1, ss=self.CF2, n=self.W)
            # update the mean and std of radius values as seen so far
            self._update_radius_mean_std()

    def recompute_radius(self, X):
        """
        Recompute the radius of this MC as if a new point X is added in

        Parameters
        ----------
        X: np.ndarray
            The new data point

        Returns
        -------

        """
        CF1_ = self.CF1 + X
        CF2_ = self.CF2 + (X * X)
        W_ = self.W + 1
        radius = self._compute_radius(ls=CF1_, ss=CF2_, n=W_)
        return radius

    def _compute_radius(self, ls, ss, n) -> float:
        """

        Parameters
        ----------
        ls
        ss
        n

        Returns
        -------

        """
        # IDEA my old method
        # a = abs(ss) / n
        # b = abs(ls) / n
        # result = math.sqrt(np.abs(np.mean(a - b*b)))

        # IDEA with n-1 as working on a sample, get the max std at the end (should it be the mean instead?)
        # actually this is the same as in Putina's paper (according to his code)
        if n > 1:
            a = ss / (n-1)
            b = (ls * ls) / (n * (n - 1))
        else:
            a = ss / n
            b = (ls * ls) / (n * n)
        # result = math.sqrt(np.max(a - b))
        result = np.nanmax(np.sqrt(a - b))

        # IDEA do it strictly as instructed in the paper, but get negative number so can't do the squared root
        # a = LA.norm(ss, ord=1) / n  # |CF2| / (W - 1)
        # b = LA.norm(ls, ord=1) / n  # |CF1| / (W(W-1))
        # result = math.sqrt(a - b*b)

        return result

    def _update_radius_mean_std(self):
        """
        Compute the standard deviation of the radius incrementally

        Returns
        -------

        """
        # first, updat the mean of radius values
        self.mean_radius = self.mean_radius + (self.radius - self.mean_radius) / self.N

        # then, update the standard deviation of radius values
        delta = (self.radius - self.mean_radius) * (self.radius - self.mean_radius)    # delta = (x - mean)^2
        self.m2 += (self.N - 1) * delta / self.N   # M2 = M2 + (N-1) * (x - mean)^2 / N
        if self.N > 1:  # only do the update if there are more than 1 points (to avoid NaN values)
            self.std_radius = math.sqrt(self.m2 / (self.N - 1))  # std = square(m2 / (n-1))

    def _update_max_distance(self, inst):
        """
        Update the maximum distance

        Returns
        -------

        """
        # only update the max distance if the system is not masked
        if inst.sys_id in self.masked_systems:
            return

        # N.B.: distance can be 0 if the cluster has only one data point
        dist = self.distance(inst.data, self.center)
        if dist > self.max_dist:
            self.max_dist = dist
            self.sys_max_dist = inst.sys_id

    def _update_linear_sum_timestamp(self, inst):
        """
        Update the linear sum of timestamp with decay factor

        Parameters
        ----------
        inst

        Returns
        -------

        """
        # FIXME to review, if timestamp = timestep then this is simply the decayed linear sum of number of points
        # do the update whether the system is masked or not because this is used to compute the anomaly score
        # and NOT to define clusters
        T = inst.tsp / float(self.time_unit)
        if inst.sys_id in self.decayed_ls_tsp.keys():
            self.decayed_ls_tsp[inst.sys_id] += 2 ** (self.decay_factor * T)
        else:
            self.decayed_ls_tsp[inst.sys_id] = 2 ** (self.decay_factor * T)

    def _update_last_timestamp(self, inst):
        """
        Update the last timestamp a system has been updated in this cluster

        Parameters
        ----------
        inst

        Returns
        -------

        """
        # update disregard whether the system is masked or non-masked
        if inst.sys_id in self.last_upd.keys():
            self.last_upd[inst.sys_id] += 1
        else:
            # if this is the 1st time sys_id is added in, set its first timestep to the creation time of this MC
            self.last_upd[inst.sys_id] = self.t0

    def decay(self, last_tsp_per_system):
        """
        Decay the micro cluster after a while

        Parameters
        ----------
        last_tsp_per_system: dict
            The most recent tsp at which a system creates its last cycle

        Returns
        -------
        None
        """
        # update CF1, CF2, and the weight by system
        for sys_id in self.cf1.keys():
            # interval = the last tsp a system produces a cycle - the last tsp a system produces a cycle in THIS mc
            # last_upd[sys_id] = last LOCAL tsp of sys_id in this mc
            # FIXED get the global last tsp of the system
            interval = (last_tsp_per_system[sys_id] - self.last_upd[sys_id]) / float(self.time_unit)
            decay = 2 ** (-self.decay_factor * interval)
            # update the linear sum CF1
            self.cf1[sys_id] *= decay
            self.CF1 = np.sum([val for val in self.cf1.values()], axis=0)
            # update the squared sum CF2
            self.cf2[sys_id] *= decay
            self.CF2 = np.sum([val for val in self.cf2.values()], axis=0)
            # update the weight W
            self.weight[sys_id] *= decay
            self.W = np.sum([val for val in self.weight.values()])
            # FIXED no need to reset last upd time (only reset it if a new point is added)
            # self.last_upd[sys_id] = tc

        # update the center and the radius
        self._update_center()
        self._update_radius()

        # FIXED decay the maximal distance, current tsp = max tsp of last tsp per system and max last up of each sys?
        if self.sys_max_dist != '':
            interval = (last_tsp_per_system[self.sys_max_dist] - self.last_upd[self.sys_max_dist]) / float(self.time_unit)
            decay = 2 ** (-self.decay_factor * interval)
            self.max_dist *= decay

    def change_type(self, new_type):
        """
        Change the type of this micro cluster

        Parameters
        ----------
        new_type

        Returns
        -------

        """
        self.mc_type = new_type

    def get_decayed_number_of_instances(self, sys_id, t_m) -> float:
        """
        Return the number of instances of a given system, decayed by the fading factor.
        The time interval is calculated wrt the latest timestamp of the specified system.

        Parameters
        ----------
        sys_id: str
            ID of the system
        t_m
            The timestamp of the system's last data generation (globally)

        Returns
        -------
        float
        """
        if sys_id not in self.members.keys():
            return 0.0

        # IDEA new method, only use decayed linear sum of timestamp
        # result = self.decayed_ls_tsp[sys_id] / (2 ** (self.decay_factor * t_m))

        # IDEA old method
        # for each instance of sys_id in this cluster, compute its decayed weight
        result = 0.0
        for inst in self.members[sys_id]:  # inst -> [0] the instance ID, [1] the instance timestamp
            interval = (t_m - inst[1]) / self.time_unit
            result += 2 ** (-self.decay_factor * interval)

        return result

    def get_number_of_instances(self, sys_id) -> int:
        """
        Get the number of instances of a given system that are in this cluster

        Parameters
        ----------
        sys_id
            System's ID

        Returns
        -------
        int
            The number of instances of sys_id that are in this cluster
        """
        if sys_id in self.n.keys():
            return self.n[sys_id]
        else:
            return 0

    def merge(self, mc):
        """
        Merge another cluster into this cluster

        Parameters
        ----------
        mc: MicroCluster
        t_merge: float
            Timestamp at which the update occurs

        Returns
        -------
        None
        """
        # update cf1 and CF1
        for sid, sval in mc.cf1.items():
            if sid not in self.cf1.keys():
                self.cf1[sid] = sval
            else:
                self.cf1[sid] += sval
        self.CF1 += mc.CF1

        # update cf2 and CF2
        for sid, sval in mc.cf2.items():
            if sid not in self.cf2.keys():
                self.cf2[sid] = sval
            else:
                self.cf2[sid] += sval
        self.CF2 += mc.CF2

        #  update weight and W
        for sid, sval in mc.weight.items():
            if sid not in self.weight.keys():
                self.weight[sid] = sval
            else:
                self.weight[sid] += sval
        self.W += mc.W

        # update n and N
        for sid, sval in mc.n.items():
            if sid not in self.n.keys():
                self.n[sid] = sval
            else:
                self.n[sid] += sval
        self.N += mc.N

        # update center and radius
        self._update_center()
        self._update_radius()

        # update members
        for sid, sval in mc.members.items():
            if sid not in self.members.keys():
                self.members[sid] = sval
            else:
                self.members[sid] += sval

        # update last_upd
        for sid, sval in mc.last_upd.items():
            # self.last_upd[sid] = t_merge
            if sid in self.last_upd.keys():
                self.last_upd[sid] = max(self.last_upd[mc.last_upd[sid]], self.last_upd[sid])
            else:
                self.last_upd[sid] = mc.last_upd[sid]

        # update the ID's of merged clusters and also their merged clusters
        self.id_merged = self.id_merged + [mc.id] + mc.id_merged

    def to_document(self):
        """
        Return a document to store in MongoDB

        Returns
        -------

        """
        result = {
            '_id': self.id,
            'id_merged': self.id_merged,
            'discarded': False,
            'mc_type': 'pmc' if self.mc_type == MCType.PMC else 'omc',
            'creation_tsp': self.t0,
            'label': 'good' if self.label == MCLabel.GOOD else 'anomaly',
            'center': self.center.tolist(),
            'radius': float(self.radius),
            # FIXED more info related to the radius
            'mean_radius': float(self.mean_radius),
            'std_radius': float(self.std_radius),
            # FIXED more info about the maximal distance
            'max_dist': float(self.max_dist),
            'sys_max_dist': self.sys_max_dist,
            # FIXED masked system
            'masked_systems': self.masked_systems,
            # FIXED decayed linear sum of timestamps per system
            'decayed_ls_tsp': self.decayed_ls_tsp,
            'n_points': int(self.N),
            'weight': float(self.W),
            'anomaly_degree': float(self.anomaly_degree),
            'cf1': self.CF1.tolist(),
            'cf2': self.CF2.tolist(),
            'last_updated': self.last_upd,
            'members': self.members,
            'by_system': {
                'cf1': {sys_id: _cf1.tolist() for sys_id, _cf1 in self.cf1.items()},
                'cf2': {sys_id: _cf2.tolist() for sys_id, _cf2 in self.cf2.items()},
                'weight': self.weight,
                'n_points': self.n
            }
        }

        # return the dict
        return result


def _is_intersect(distance: Callable, g1: MicroCluster, g2: MicroCluster):
    """
    Check if two clusters intersect

    Parameters
    ----------
    g1: MicroCluster
        A cluster
    g2: MicroCluster
        Another cluster

    Returns
    -------
    bool
        True if g1 and g2 intersect, False otherwise
    """
    # if the distance between the two centroids is <= the sum of their radius, these clusters intersect
    dist = distance(g1.center, g2.center)
    return dist <= g1.max_dist + g2.max_dist


def _is_approx_similar(g1: MicroCluster, g2: MicroCluster, confidence=0.95):
    """
    Check if two clusters are approximately similar, by checking if their confidence interval intersects

    Parameters
    ----------
    g1: MicroCluster
        A cluster
    g2: MicroCluster
        Another cluster fella

    Returns
    -------

    """
    # confidence interval in size of G1
    t1 = np.abs(student_t.ppf((1 - confidence) / 2, g1.N - 1))
    err1 = t1 * g1.radius / np.sqrt(g1.N)
    # IDEA size or weight? should be weight because size only increases...
    ci1 = (g1.W - err1, g1.W + err1)

    # confidence interval in size of G2
    t2 = np.abs(student_t.ppf((1 - confidence) / 2, g2.N - 1))
    err2 = t2 * g2.radius / np.sqrt(g2.N)
    # IDEA size or weight? should be weight because size only increases...
    ci2 = (g2.W - err2, g2.W + err2)

    # check if the 2 intervals intersect
    case1 = ci2[1] > ci1[1] > ci2[0] > ci1[0]
    case2 = ci1[1] > ci2[1] > ci1[0] > ci2[0]
    case3 = ci1[0] <= ci2[0] and ci1[1] >= ci2[1]
    case4 = ci1[0] >= ci2[0] and ci1[1] <= ci2[1]
    return case1 or case2 or case3 or case4


class DenStream:
    """
    This class implements the DenStream algorithm, proposed in [1].

    [1] F. Cao, M. Ester, W. Qian, and A. Zhou, “Density-Based Clustering over an Evolving Data Stream with Noise,”
    in Proceedings of the Sixth SIAM International Conference on Data Mining, April 20-22, 2006,
    Bethesda, MD, USA, 2006, vol. 2006. doi: 10.1137/1.9781611972764.29.
    """

    def __init__(self, eps: float, mu: float, beta: float, lamb: float, W: float, time_unit: TIMEUNIT,
                 dist_metric: str, min_points_system: int, db_conn: MongoDBConnector):
        """
        Initialize DenStream

        Parameters
        ----------
        eps
        mu
        beta
        lamb
        W
        time_unit
        dist_metric
        db_conn
        """
        # primary hyperparameters
        self.eps = eps
        self.mu = mu
        self.beta = beta
        self.lamb = lamb

        # FIXED OMC/PMC relative weight threshold
        # original version is beta * mu
        self.outlier_threshold = self.beta / (1 - 2 ** (-self.lamb))

        # FIXED to decide the membership eligibility of a system
        self.min_points_system = min_points_system

        # budget-related constraints (not used for now)
        self.W = W
        self.v = -1
        self.n_mc = self.W / (self.mu * self.beta)

        # inner data structures
        self.pmc_buffer: List[MicroCluster] = []
        self.omc_buffer: List[MicroCluster] = []

        # mark the reference cluster
        # [0] OMC or PMC, [1] = index in the PMC or OMC buffer, [2] = cluster ID
        self.ref_cluster = None

        # time-related
        self.T_p = self._compute_pruning_interval()
        self.T_now = 1
        self.last_pruning = 0
        self.time_unit = time_unit

        # warm start state
        self.has_warm_start = False

        # connection to database
        self.db_conn = db_conn

        # distance metrics
        self.distance = euclidean_distance if dist_metric == 'euclidean' else manhattan_distance

        # cheat: store the global last tsp per system (key = system ID, value = timestamp)
        self.global_last_tsp = {}

        # time-related measurements
        self.pruning_time = 0
        self.decay_time = 0
        self.merging_time = 0

    def reset_timer(self):
        """
        Reset all the time-related measurements

        Returns
        -------

        """
        self.pruning_time = 0
        self.decay_time = 0
        self.merging_time = 0

    def _compute_pruning_interval(self) -> float:
        """
        Compute the minimal time span for pruning operation

        Returns
        -------

        """
        # original version
        # return math.ceil((1 / self.lamb) * np.log((self.beta * self.mu) / (self.beta * self.mu - 1)))

        # FIXED if we lump u to u+ -> minital timespan also changes
        return math.ceil((1 / self.lamb) * np.log2(1 / self.beta))

    def update_current_time(self):
        """
        Update the current time as requested (called by CHMOC)

        Returns
        -------

        """
        self.T_now += 1

    def update_parameters(self, new_v):
        """
        Update the weight, lambda, and the minimal time span given the new stream speed

        Parameters
        ----------
        new_v
            New speed of the stream

        Returns
        -------
        None
        """
        # update the weight (budget) according to the new speed
        if self.W <= new_v:
            self.W = new_v + np.log(new_v / self.v) * self.W

        # update the maximum number of micro clusters
        self.n_mc = self.W / (self.beta * self.mu)

        # update the decay factor lambda
        self.lamb = math.log2(self.W / (self.W - new_v))

        # update the minimal time span
        self.T_p = self._compute_pruning_interval()

        # update epsilon with the average non-0 radius of all the micro clusters
        self._reassess_epsilon()

        # finally, set the new speed
        self.v = new_v

    def _reassess_epsilon(self):
        """
        Recompute epsilon by taking the average radius of all the clusters

        Returns
        -------

        """
        # IDEA my own method (mean of all radiuses)
        # all_mc_radius = [mc.radius for mc in self.omc_buffer + self.pmc_buffer if not _is_close_to_0(mc.radius)]
        # if len(all_mc_radius) != 0:
        #     self.eps = np.mean(all_mc_radius)

        # IDEA from Putina's work (k = 3 by default)
        candidates = [mc.mean_radius + mc.std_radius * 3 for mc in self.pmc_buffer]
        if len(candidates) != 0:
            self.eps = np.nanmax(candidates)

    def warm_start(self, batch: List[Instance]):
        """
        Warm start on a batch of data to estimte the initial value of epsilon
        (as instructed in Cao et al. (2006))

        Parameters
        ----------
        batch: List[Instance]

        Returns
        -------

        """
        # find the initial set of clusters
        N, D = len(batch), batch[0].dim   # size and dimension of the warm start batch
        membership = np.zeros(N)  # quick membership access
        idx_pmc = 0   # index of any new pmc
        visited = [False] * N  # true if a point has been processed, false otherwise
        for i in range(N):
            if not visited[i]:
                ngb = [j for j in range(N) if i != j and self.distance(batch[i].data, batch[j].data) <= self.eps]

                # if the neighborhood is dense enough
                if len(ngb) >= self.outlier_threshold:
                    # create a new miro-cluster
                    # pmc = MicroCluster(dim=D, mc_type=MCType.PMC, t0=batch[i].tsp, time_unit=self.time_unit,
                    #                    distance=self.distance, min_points_system=self.min_points_system,
                    #                    decay_factor=self.lamb)
                    pmc = MicroCluster(dim=D, mc_type=MCType.PMC, t0=self.T_now, time_unit=self.time_unit,
                                       distance=self.distance, min_points_system=self.min_points_system,
                                       decay_factor=self.lamb)
                    self.db_conn.add_cluster(pmc.to_document())

                    # add points to this cluster
                    pmc.add(batch[i])
                    for j in ngb:
                        if not visited[j]:  # if j not already visited and nor is added to any cluster
                            pmc.add(batch[j])
                            visited[j] = True
                            membership[j] = idx_pmc

                    # update it in the database & add it to the buffer
                    self.pmc_buffer.append(pmc)
                    self.db_conn.update_cluster(pmc.to_document())

                    # update the membership in the cluster
                    membership[i] = idx_pmc
                    idx_pmc += 1
                visited[i] = True

        # save the data points in the database as well
        for inst in batch:
            # FIXED update the global last tsp per system (ordering guaranteed from the feeder)
            # self.global_last_tsp[inst.sys_id] = inst.tsp
            # first time update the global last timestamp
            if inst.sys_id in self.global_last_tsp.keys():
                self.global_last_tsp[inst.sys_id] += 1
            else:
                self.global_last_tsp[inst.sys_id] = self.T_now

            # FIXME have to change this in the database as well?...
            self.db_conn.add_system(sys_id=inst.sys_id, t=inst.tsp, timestep=self.global_last_tsp[inst.sys_id])
            self.db_conn.add_instance(inst.to_document())

        # estimate the initial value of eps
        # self._reassess_epsilon()

        # reasses the reference cluster
        self.reassess_clusters(t_upd=batch[-1].tsp)

        # prevent that a pruning has been done at the warm start to avoid losing too many clusters in the next batch
        # self.last_pruning = batch[-1].tsp

        # check has warm start
        self.has_warm_start = True

    def process(self, instance: Instance):
        """
        Update DenStream on a new data point

        Parameters
        ----------
        instance: Instance

        Returns
        -------

        """
        # FIXED update the global last timestamp of this system
        # self.global_last_tsp[instance.sys_id] = instance.tsp
        if instance.sys_id not in self.global_last_tsp.keys():
            self.global_last_tsp[instance.sys_id] = self.T_now
        else:
            self.global_last_tsp[instance.sys_id] += 1

        # add the new point and system to the database
        self.db_conn.add_instance(instance.to_document())
        self.db_conn.add_system(sys_id=instance.sys_id, t=instance.tsp, timestep=self.global_last_tsp[instance.sys_id])

        # merge the new point in a cluster
        self._merge(instance)

    def _merge(self, inst):
        """
        Merge a new data point in a micro cluster

        Parameters
        ----------
        inst: Instance
            The data point

        Returns
        -------

        """
        c_p = self._find_nearest_pmc(inst.data)

        # if the point can be in the eps-neighborhood of this PMC -> add it in the PMC
        if c_p is not None and c_p.recompute_radius(inst.data) <= self.eps:
            c_p.add(inst)
        else:
            c_o = self._find_nearest_omc(inst.data)
            # if the point can be in the eps-neighborhood of this OMC -> add it in the OMC
            if c_o is not None and c_o.recompute_radius(inst.data) <= self.eps:
                c_o.add(inst)
                # if the point makes the OMC denser, it becomes a PMC
                if c_o.W > self.outlier_threshold:
                    c_o.change_type(MCType.PMC)  # this OMC becomes an PMC
                    self.pmc_buffer.append(c_o)  # add it to the PMC buffer
                    self.omc_buffer.remove(c_o)  # remove it from the OMC buffer
            # if it can't be added in the nearest PMC nor the OMC -> create a new OMC
            else:
                # new_omc = MicroCluster(dim=inst.dim, mc_type=MCType.OMC, t0=inst.tsp, time_unit=self.time_unit,
                #                        distance=self.distance, min_points_system=self.min_points_system,
                #                        decay_factor=self.lamb)
                # FIXME how to make a consistent creation time?
                new_omc = MicroCluster(dim=inst.dim, mc_type=MCType.OMC, t0=self.T_now, time_unit=self.time_unit,
                                       distance=self.distance, min_points_system=self.min_points_system,
                                       decay_factor=self.lamb)
                new_omc.add(inst)
                self.omc_buffer.append(new_omc)

    def _find_nearest_pmc(self, X) -> Union[MicroCluster, None]:
        """
        Find the PMC nearest to this point

        Parameters
        ----------
        X

        Returns
        -------

        """
        # if no PMC yet, return None
        if len(self.pmc_buffer) == 0:
            return None

        # else, find the nearest PMC
        dist = [self.distance(X, c.center) for c in self.pmc_buffer]
        i_closest = np.argmin(dist)
        return self.pmc_buffer[i_closest]

    def _find_nearest_omc(self, X) -> Union[MicroCluster, None]:
        """
        Find the OMC nearest to this point

        Parameters
        ----------
        X

        Returns
        -------

        """
        # if no OMC yet, return None
        if len(self.omc_buffer) == 0:
            return None

        # else, find the nearest OMC
        dist = [self.distance(X, c.center) for c in self.omc_buffer]
        i_closest = np.argmin(dist)
        return self.omc_buffer[i_closest]

    def prune(self, t_upd: float):
        """
        At pruning time, do the following:
        - decay PMC to OMC (do NOT promote OMC to PMC - it has already been done when merging new instances)
        - discard very outdated OMC

        Parameters
        ----------
        t_upd: float
            Timestamp at which the decay occurs

        Returns
        -------

        """
        # demote pmc to omc
        to_demote = []
        for i, mc in enumerate(self.pmc_buffer):
            mc.decay(last_tsp_per_system=self.global_last_tsp)
            # demote it if its weight goes below beta * mu
            if mc.W < self.outlier_threshold:
                # TODO demote to OMC or discard it completely?
                to_demote.append(i)
                mc.change_type(MCType.OMC)
                self.omc_buffer.append(mc)

        # remove demoted PMC from the buffer
        self.pmc_buffer = [mc for i, mc in enumerate(self.pmc_buffer) if i not in to_demote]

        # decay and discard very outdate OMC
        to_discard = []
        for i, mc in enumerate(self.omc_buffer):
            mc.decay(last_tsp_per_system=self.global_last_tsp)
            # discard it if its weight goes below the minimum acceptable threshold
            # TODO even the low limit must be separated by system???
            interval = (t_upd - mc.t0 + self.T_p) / float(self.time_unit)
            low_limit = (2 ** (-self.lamb * interval) - 1) / (2 ** (-self.lamb * self.T_p) - 1)
            if mc.W < low_limit:
                to_discard.append(i)
                self.db_conn.discard_cluster(mc.id)

        # discard these OMC
        self.omc_buffer = [mc for i, mc in enumerate(self.omc_buffer) if i not in to_discard]

    def reassess_clusters(self, t_upd: float):
        """
        Find the new reference cluster and recompute the anomaly degree of each cluster.

        Parameters
        ----------
        t_upd: float
            Timestamp of the update

        Returns
        -------

        """
        # if no micro-clusters -> do nothing, reset the good cluster info
        if len(self.pmc_buffer) == 0 and len(self.omc_buffer) == 0:
            self.ref_cluster = None

        # # refine the clusters (only if it is not during the warm start)
        # if self.has_warm_start:
        #     self.refine_clusters(t_upd=t_upd)

        # after refining, reassess epsilon
        self._reassess_epsilon()

        # if only omc without pmc -> no reference cluster either
        if len(self.pmc_buffer) == 0:
            self.ref_cluster = None

        # if some pmc -> search for good cluster among the pmc
        # IDEA what if there are many OMC but no PMC? does it make sense to not have a reference cluster?
        if len(self.pmc_buffer) != 0:
            # first, label all as anomaly
            for mc in self.pmc_buffer:
                mc.label = MCLabel.ANOMALY

            # find the dominant cluster
            # IDEA reassess by size or weight?
            idx_good = np.argmax([pmc.W for pmc in self.pmc_buffer])  # FIXED: reassess by weight
            # [0] = omc or pmc, [1] = index in the buffer, [2] cluster id
            self.ref_cluster = (MCType.PMC, idx_good, self.pmc_buffer[idx_good].id)
            self.pmc_buffer[idx_good].label = MCLabel.GOOD      # label -> 'good'
            self.pmc_buffer[idx_good].anomaly_degree = 0.0      # anomaly weight of good cluster = 0.0

            # a refinement step to merge approximately close clusters together
            # self._refine_clusters(t_upd)

            # update the clusters' anomaly degree as well
            max_weight = 0.0
            for mc in self.pmc_buffer + self.omc_buffer:
                # avoid the reference cluster
                if mc != self.pmc_buffer[idx_good]:
                    mc.anomaly_degree = np.linalg.norm(mc.center - self.pmc_buffer[idx_good].center)
                    if mc.anomaly_degree > max_weight:
                        max_weight = mc.anomaly_degree
            # normalize the anomaly degree of all clusters
            for mc in self.pmc_buffer + self.omc_buffer:
                # avoid the reference cluster
                if mc != self.pmc_buffer[idx_good]:
                    mc.anomaly_degree = mc.anomaly_degree / max_weight

    def refine_clusters(self, t_upd: float):
        """
        Refine the clusters after having identified a new refenrece cluster.
        Condition to merge:
        - two clusters intersect
        - two clusters have approximately the same weight (confidence interval)

        First, only check in the vicinity of the new reference cluster

        Parameters
        ----------
        t_upd: float
            Update time

        Returns
        -------

        """
        # for now, just decay them
        for mc in self.pmc_buffer + self.omc_buffer:
            mc.decay(last_tsp_per_system=self.global_last_tsp)

    def update_clusters_in_db(self):
        """
        Update all the clusters at once in mongodb
        (to be called at the end of each batch)

        Returns
        -------

        """
        for mc in self.pmc_buffer + self.omc_buffer:
            self.db_conn.update_cluster(mc.to_document())

    def _get_reference_cluster(self) -> Union[MicroCluster, None]:
        """
        Get the reference cluster as a MicroCluster object

        Returns
        -------

        """
        # if no good cluster has been identified yet
        if self.ref_cluster is None:
            return None

        # else, retrieve it
        ref_type, ref_idx = self.ref_cluster[0], self.ref_cluster[1]
        buffer_ = self.pmc_buffer if ref_type == MCType.PMC else self.omc_buffer
        if len(buffer_) != 0:
            return buffer_[ref_idx]
        else:
            return None

    def update_health(self, sys_id: str, t_upd: float):
        """
        Update the health score of a system

        Parameters
        ----------
        sys_id: str
            ID of the system to update
        t_upd: float
            Timestamp of the update

        Returns
        -------

        """
        # get the reference cluster (if None, do nothing))
        g_ref = self._get_reference_cluster()
        if g_ref is None:
            return

        # allow debugging
        ano_scores_ = {}

        # recompute the health score from all the micro-clusters (only from the PMC)
        uw_health_score = 0
        w_health_score = 0
        uw_activated = 0
        w_activated = 0
        for g_k in self.pmc_buffer:
            # if it is not the reference cluster
            if g_k != g_ref:
                # compute the anomaly score
                # IDEA non-weighted version
                uw_score_k = g_k.get_number_of_instances(sys_id)
                uw_score_ref = g_ref.get_number_of_instances(sys_id)
                uw_ano_score = uw_score_k / (uw_score_k + uw_score_ref) if uw_score_k + uw_score_ref != 0 else 0
                if uw_ano_score > 0:
                    uw_activated += 1

                # IDEA weighted/decayed version
                w_score_k = g_k.get_decayed_number_of_instances(sys_id, self.global_last_tsp[sys_id])
                w_score_ref = g_ref.get_decayed_number_of_instances(sys_id, self.global_last_tsp[sys_id])
                w_ano_score = w_score_k / (w_score_k + w_score_ref) if w_score_k + w_score_ref != 0 else 0
                if w_ano_score > 0:
                    w_activated += 1

                # log them
                ano_scores_[str(g_k.id)] = {
                    'uw_score_k': uw_score_k, 'uw_score_ref': uw_score_ref, 'uw_total': uw_ano_score,
                    'w_score_k': w_score_k, 'w_score_ref': w_score_ref, 'w_total': w_ano_score
                }

                # sum in the health score
                uw_health_score += (g_k.anomaly_degree * uw_ano_score)
                w_health_score += (g_k.anomaly_degree * w_ano_score)

        # compute the final average
        if len(self.pmc_buffer) > 1:
            # uw_health_score = uw_health_score / (len(self.pmc_buffer) - 1)
            # w_health_score = w_health_score / (len(self.pmc_buffer) - 1)
            # FIXED if average by all clusters the health may be less accurate thus cause false negative
            uw_health_score = uw_health_score / uw_activated if uw_activated != 0 else uw_health_score
            w_health_score = w_health_score / w_activated if w_activated != 0 else w_health_score

        # finally, update in the database
        self.db_conn.add_health_score(sys_id=sys_id,
                                      health_score={
                                          'weighted': {'score': w_health_score, 'activated': w_activated},
                                          'unweighted': {'score': uw_health_score, 'activated': uw_activated}
                                      },
                                      t=t_upd,
                                      timestep=self.global_last_tsp[sys_id], ano_scores=ano_scores_)

    def to_document(self) -> dict:
        """
        Export all hyperparameters in form of dictionary

        Returns
        -------
        dict
        """
        return {
            'eps': float(self.eps),
            'mu': float(self.mu),
            'beta': float(self.beta),
            'lamb': float(self.lamb),
            'stream_weight': float(self.W),
            'stream_speed': float(self.v),
            'timespan': float(self.T_p),
            'mc_budget': int(self.n_mc),
            'has_warm_start': bool(self.has_warm_start),
            'last_pruning': float(self.last_pruning),
            'current_time': float(self.T_now),
            'n_pmc': len(self.pmc_buffer),
            'n_omc': len(self.omc_buffer),
            'T_now': self.T_now,
            'T_p': self.T_p,
            'outlier_threshold': self.outlier_threshold,
            'ref_cluster': None if self.ref_cluster is None else
            {
                'mctype': str(self.ref_cluster[0]),
                'mc_index': int(self.ref_cluster[1]),
                'mc_id': int(self.ref_cluster[2])
            }
        }

    def get_cluster_snapshot(self) -> List[dict]:
        """
        Get a summary of all clusters (lightweight, not including the details of all the points)

        Returns
        -------

        """
        res = []
        for mc in self.pmc_buffer + self.omc_buffer:
            res.append({
                'cluster_id': mc.id,
                'mctype': str(mc.mc_type),
                'CF1': mc.CF1.tolist(),
                'CF2': mc.CF2.tolist(),
                'center': mc.center.tolist(),
                'radius': mc.radius,
                # FIXED more info about the radius
                'mean_radius': float(mc.mean_radius),
                'std_radius': float(mc.std_radius),
                # FIXED more info about the max distance
                'max_dist': float(mc.max_dist),
                'sys_max_dist': mc.sys_max_dist,
                # FIXED masked systems
                'masked_systems': mc.masked_systems,
                # FIXED decayed linear sum of timestamps
                # 'decayed_ls_tsp': mc.decayed_ls_tsp,
                'size': int(mc.N),
                'weight': float(mc.W),
                'anomaly_degree': float(mc.anomaly_degree),
                'label': str(mc.label),
                'creation_tsp': mc.t0,
                'id_merged': mc.id_merged,
                # will be very heavy to store these kinds of snapshots...
                # 'n_points_per_system': mc.n,
                # 'cf1_per_system': {sid: sval.tolist() for sid, sval in mc.cf1.items()},
                # 'cf2_per_system': {sid: sval.tolist() for sid, sval in mc.cf2.items()},
                # 'points_per_system': {sid: [m[0] for m in mb] for sid, mb in mc.members.items()}
                'points_per_system': {sid: [(m[0], m[1]) for m in mb] for sid, mb in mc.members.items()}
            })
        return res
