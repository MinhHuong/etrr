"""
CONNECTOR TO A MONGO DATBASE
"""

# mongodb driver for python
from typing import List

import pymongo as mg

# python's native libraries
import time
from datetime import datetime as dt


class MongoDBConnector:
    """
    Connector to the MongoDB to write the experiment metrics
    """

    # connection info
    HOST = '[host]'
    PORT = 27017
    DBNAME = '[database name]'

    # name of collections
    EXPERIMENTS = 'experiments'
    CYCLES = 'cycles'
    CLUSTERS = 'clusters'
    SYSTEMS = 'systems'
    SYSTEM_HEALTH = 'system_health'

    def __init__(self, dataset):
        """
        Initialize the connector & open the connection
        """
        self.client = mg.MongoClient(host=self.HOST, port=self.PORT)
        self.db = self.client[self.DBNAME[dataset]]
        print('Connection to MongoDB successfully opened.')

    def close(self):
        """
        Close the connection to MongoDB

        Returns
        -------

        """
        self.client.close()
        print('Connection to MongoDB successfully closed.')

    def add_experiment(self, expe_name: str, expe_note: str, expe_config: dict):
        """
        Add a new experiment entry in the meta-expe collection (store general info of an experiment)
        The ID and timestamp of the experiment will be set automatically.

        Parameters
        ----------
        expe_name: str
            Name of the experiment
        expe_note: str
            Notes about this expe
        expe_config: dict
            Configuration of the experiment

        Returns
        -------
        ObjectID
        """
        expe = {
            'expe_name': expe_name,
            'expe_tsp': time.time(),
            'expe_date': dt.now(),
            'expe_note': expe_note,
            'config': expe_config
        }
        self.db[self.EXPERIMENTS].insert_one(expe)

    def update_experiment(self, expe_name: str, expe_info: dict):
        """
        Update the current experiment by adding in some info

        Parameters
        ----------
        expe_name
        expe_info

        Returns
        -------

        """
        expe_collection = f'{self.EXPERIMENTS}_{expe_name}'
        return self.db[expe_collection].insert_one(expe_info).inserted_id

    def add_cluster_snapshot(self, expe_name: str, batch_id, clusters: List[dict]):
        """
        Add a snapshot of all clusters in a collection of the name
        "clusterlog_[expe_name]" (each cluster its own document in mongodb)

        Parameters
        ----------
        expe_name: str
            Name of the experiment
        batch_id
            ID of the batch-level entry in the experiment log
        clusters: List[dict]
            List of cluster snapshot in form of dictionary

        Returns
        -------

        """
        snapshot_collection = f'clusterlog_{expe_name}'
        for clus in clusters:
            doc_clus = clus.copy()
            doc_clus['batch_id'] = batch_id
            self.db[snapshot_collection].insert_one(doc_clus)

    def add_system(self, sys_id: str, t: float, timestep: int, meta_info=''):
        """

        Parameters
        ----------
        sys_id: str
            ID of the system
        t: float
            The timestamp where this sytem created the last point in the current batch
        timestep: int
            The timestep starting from 1 (equivalent to the number of cycles this system has generated so far)
        meta_info: str
            Additional information about the system

        Returns
        -------

        """
        sys_ = self.db[self.SYSTEMS].find_one({'_id': sys_id})

        # insert new if this system has not been added in the database
        # _id = ID of the system as a string (vehicule_scnf + dcu), last_timestamp = most recent tsp of data from sys_id
        if sys_ is None:
            system_doc = {
                '_id': sys_id,
                'last_timestamp': t,
                'last_timestep': timestep
            }
            self.db[self.SYSTEMS].insert_one(system_doc)
        # otherwise, update its last timestamp AND timestep
        else:
            self.db[self.SYSTEMS].update_one(filter={'_id': sys_id},
                                             update={'$set': {'last_timestamp': t, 'last_timestep': timestep}})

    def get_all_systems(self) -> List[dict]:
        """
        Take a snapshot of all the systems

        Returns
        -------
        List[dict]
        """
        res = []
        systems = self.db[self.SYSTEMS].find()
        for syst in systems:
            res.append({
                'sys_id': syst['_id'],
                'last_timestamp': syst['last_timestamp'],
                'last_timestep': syst['last_timestep'],
                'heatlh': syst['health']
            })
        return res

    def add_instance(self, instance: dict):
        """

        Parameters
        ----------
        instance

        Returns
        -------

        """
        self.db[self.CYCLES].insert_one(instance)

    def add_cluster(self, cluster: dict):
        """

        Parameters
        ----------
        cluster

        Returns
        -------

        """
        self.db[self.CLUSTERS].insert_one(cluster)

    def update_cluster(self, cluster: dict):
        """
        Update an existing cluster, or add new if it hasn't been added yet

        Parameters
        ----------
        cluster

        Returns
        -------

        """
        clus_ = self.db[self.CLUSTERS].find_one({'_id': cluster['_id']})
        # if new cluster --> add it
        if clus_ is None:
            self.add_cluster(cluster)
        # else, update the existing cluster
        else:
            self.db[self.CLUSTERS].replace_one(filter={'_id': cluster['_id']}, replacement=cluster)

    def discard_cluster(self, cluster_id):
        """
        Discard a given cluster

        Parameters
        ----------
        cluster_id

        Returns
        -------

        """
        # (1) instead of dropping it definitely from the DB, just set its status from MAINTAINED to DISCARDED
        # self.db[self.CLUSTERS].update_one(filter={'_id': cluster_id},
        #                                   update={'$set': {'discarded': True}})

        # (2) just delete it from the DB permanently
        self.db[self.CLUSTERS].delete_one(filter={'_id': cluster_id})

    def add_health_score(self, sys_id: str, health_score: dict, t: float, timestep: int, ano_scores: dict):
        """
        Add a new record of the health of a given system

        Parameters
        ----------
        sys_id
        health_score
        t
        timestep
        ano_scores

        Returns
        -------

        """
        health_record = {
            'sys_id': sys_id,
            'health': health_score,
            'timestamp': t,
            'timestep': timestep,
            'ano_scores': ano_scores
        }
        self.db[self.SYSTEM_HEALTH].insert_one(health_record)

    def clean(self, expe_name=None):
        """
        Clean the whole database

        Parameters
        ----------
        expe_name
            Name of the experiment to erase all metrics of (if empty, leave unscathed)

        Returns
        -------

        """
        self.db[self.CLUSTERS].delete_many({})
        self.db[self.CYCLES].delete_many({})
        self.db[self.SYSTEM_HEALTH].delete_many({})
        self.db[self.SYSTEMS].delete_many({})

        # remember to NOT to delete experiment logs
        if expe_name is not None:
            self.db[f'{self.EXPERIMENTS}_{expe_name}'].delete_many({})  # delete log per batch
            self.db[f'clusterlog_{expe_name}'].delete_many({})  # delete cluster snapshots as well
