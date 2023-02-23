"""
CHMOC: Continuous Health Monitoring using Online Clustering
"""

# custom modules
import time
from typing import List

from core.denstream import DenStream
from database.mongo_conn import MongoDBConnector
from feeder.data_point import Instance


class CheMoc:
    """
    CheMoc: Continuous Health Monitoring using Online Clustering.

    CheMoc uses an online clustering algorithm as the core processor.
    The choice of the core online clus. algorithm is arbitrary. Any algorithm
    that produces online clusters can be used.
    """

    def __init__(self, config, db_conn):
        """
        Initialize CheMoc

        Parameters
        ----------
        config: dict
            A dictionary containing the config of the test
        db_conn: MongoDBConnector
            A connection to a MongoDB
        """
        # initial weight of the stream
        W = config['clustering']['n_pmc'] / (config['clustering']['beta'] * config['clustering']['mu'])

        # initialize the online clusterer
        self.clusterer = DenStream(eps=config['clustering']['eps'],
                                   mu=config['clustering']['mu'],
                                   beta=config['clustering']['beta'],
                                   lamb=config['clustering']['lamb'],
                                   W=W,
                                   dist_metric=config['clustering']['distance'],
                                   min_points_system=config['clustering']['min_points_system'],
                                   db_conn=db_conn)

        # connection to database
        self.db_conn = db_conn

        # time-related measurement
        self.ws_time = 0  # warm start time
        self.denstream_time = 0  # time spent in denstream
        self.prune_time = 0  # time for pruning
        self.reassess_time = 0  # time to reassess clusters
        self.updhealth_time = 0  # time to reupdate system health
        self.updcluster_time = 0  # time to update all the clusters at the end of each batch

    def process(self, batch: List[Instance]):
        """
        Process a new batch of data

        Parameters
        ----------
        batch
            A batch of N data points, each data point contains: the data, the ID, the system ID, the timestamp

        Returns
        -------

        """
        # reset the timer in DenStream
        self.clusterer.reset_timer()

        # if DenStream has not been warm started, do it on the first batch
        if not self.clusterer.has_warm_start:
            start = time.perf_counter()
            self.clusterer.warm_start(batch)
            end = time.perf_counter()
            self.ws_time = end - start
            print(f'\tWarm start: found {len(self.clusterer.pmc_buffer) + len(self.clusterer.omc_buffer)} clusters')
            return

        # update the current timestamp on this new batch (timestamp of last data point)
        self.clusterer.update_current_time()

        # (1) update DenStream on every point in this batch
        start = time.perf_counter()
        for idx, inst in enumerate(batch):
            self.clusterer.process(inst)
        end = time.perf_counter()
        self.denstream_time = end - start

        # (3) reassess the cluster: find the new reference cluster and recompute the weight
        start = time.perf_counter()
        self.clusterer.reassess_clusters(t_upd=self.clusterer.T_now)
        end = time.perf_counter()
        self.reassess_time = end - start

        # (4) update all the clusters at once
        start = time.perf_counter()
        self.clusterer.update_clusters_in_db()
        end = time.perf_counter()
        self.updcluster_time = end - start

        # (5) update health once every batch
        start = time.perf_counter()
        sys_ids = set([inst.sys_id for inst in batch])
        for sys_id in sys_ids:
            self.clusterer.update_health(sys_id=sys_id, t_upd=self.clusterer.T_now)
        end = time.perf_counter()
        self.updhealth_time = end - start

    def get_clusterer_snapshot(self) -> dict:
        """
        Get the current hyperparameter values of the inner clusterer (DenStream)

        Returns
        -------

        """
        return self.clusterer.to_document()

    def get_cluster_snapshot(self) -> List[dict]:
        """
        Get a snapshot of all the clusters maintained by the clusterer

        Returns
        -------

        """
        return self.clusterer.get_cluster_snapshot()

    def get_system_snapshot(self) -> List[dict]:
        """
        Get a snapshot of all the systems' health

        Returns
        -------

        """
        return self.db_conn.get_all_systems()
