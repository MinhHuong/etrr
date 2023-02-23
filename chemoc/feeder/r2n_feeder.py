"""
CONNECTOR TO A POSTGRES DATABASE
"""

# python's native libraries
import time
from typing import List

# postgres driver in python
import psycopg2 as pg

# data manipulation
import numpy as np

# custom modules
from feeder.data_point import Instance


class Feeder:
    """
    Connect to a postgres database (to fetch data - R2N case)
    """

    DBNAME = '[database of feature vectors]'
    USER = '[database user]'
    PASSWORD = '[database password]'

    def __init__(self, atts: List, excluded_cyc: str):
        """
        Init the connector
        """
        # init a connection
        self.conn = pg.connect(dbname=self.DBNAME, user=self.USER, password=self.PASSWORD)

        # get all the week numbers
        self.weeks = self._get_all_weeks()

        # current week to retrieve data
        self.current = 0

        # list of indicators to retrieve at request time
        self.atts = atts

        # mean, std, min, max of all attributes
        self._mean = np.zeros(len(self.atts))
        self._std = np.zeros(len(self.atts))
        self._min = np.zeros(len(self.atts))
        self._max = np.zeros(len(self.atts))
        self._get_mean_std_min_max()

    def close(self):
        """
        Shut down the connector

        Returns
        -------

        """
        self.conn.close()

    def _get_all_weeks(self):
        """
        Retrieve all the weeks that have data

        Returns
        -------
        List of week numbers
        """
        with self.conn.cursor() as cur:
            cur.execute("""[query to feed feature vectors from database]""")
            res = [row[0] for row in cur.fetchall()]
            return res

    def has_data(self):
        """
        Check if there are more data

        Returns
        -------
        bool
            True if there are more data in 2021, False otherwise
        """
        return self.current < len(self.weeks)

    def _get_mean_std_min_max(self):
        """
        Compute the mean and standard deviation of ALL the data in the test batch for normalization

        Returns
        -------

        """
        # query filler
        query_atts = ', '.join([f'avg(ind.{att}), stddev(ind.{att}), min(ind.{att}), max(ind.{att})'
                                for att in self.atts])

        with self.conn.cursor() as cur:
            cur.execute("""[query to get the min, max, avg, std of each feature""")
            # retrieve the mean and std of each indicator
            res = cur.fetchone()
            for i in range(len(self.atts)):
                self._mean[i] = float(res[4 * i])
                self._std[i] = float(res[4 * i + 1])
                self._min[i] = float(res[4 * i + 2])
                self._max[i] = float(res[4 * i + 3])

    def fetch_data(self) -> List[Instance]:
        """
        Get new data (weekly)

        Returns
        -------

        """
        # if all data have been queried
        if self.current >= len(self.weeks):
            return []

        # else, fetch the indicators during the week
        query_filler = ', '.join([f'ind.{att}' for att in self.atts])
        with self.conn.cursor() as cur:
            cur.execute("""[query to fetch data]""")

            # increment the iterator index
            self.current += 1

            # each data point: X (data), X_id, sys_id, X_timestamp
            res = []
            for item in cur.fetchall():
                X = np.array([float(indi) if indi is not None else np.nan for indi in item[7:]])
                X = (X - self._mean) / self._std
                # avoid rows with NaN values
                if not np.isnan(sum(X)):
                    syst_id = f'{item[1]}_{item[2]}'
                    ctxt = {'temperature': item[4], 'station_id': item[5], 'mission_code': item[6]}
                    instance = Instance(data=X, data_id=item[0], sys_id=syst_id, tsp=item[3].timestamp(), ctxt=ctxt)
                    res.append(instance)

            return res
