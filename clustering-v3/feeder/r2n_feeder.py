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
from sklearn.decomposition import PCA

# custom modules
from feeder.data_point import Instance


class R2NFeeder:
    """
    Connect to a postgres database (to fetch data - R2N case)
    """

    DBNAME = 'r2n_indicators'
    USER = 'cbmr2n_admin'
    PASSWORD = '123456'

    def __init__(self, atts: List, excluded_cyc: str):
        """
        Init the connector
        """
        # ID's of the cycles to be excluded
        self.excluded = []
        with open(excluded_cyc, mode='r') as f:
            content = f.read()
            self.excluded = tuple([int(part) for part in content.split(',')])

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

        # dimension reduction model
        # self.dimension = dim
        # self.dim_reduc = PCA(n_components=dim, random_state=420)
        # self._fit_pca()

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
            cur.execute("""
            select distinct(date_part('week', ctxt.date_record)) as no_week
            from cbm_av.ind_por_f as ind, cbm_av.cycle_por_f as cyc, cbm_av.ctxt_file as ctxt, cbm_av.station as stt
            where date_part('year', ctxt.date_record) = 2021  AND ctxt.pvs_stationid is not null AND stt.valid_p = 1
            and ind.id_cycle not in %s
            and ind.id_cycle = cyc.id_cycle and cyc.id_ctxt = ctxt.id_ctxt 
            and ctxt.pvs_stationid = stt.station_short_name
            order by no_week;
            """, (self.excluded,))
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
            cur.execute(f"""
            SELECT {query_atts}
            FROM cbm_av.ind_por_f as ind, cbm_av.cycle_por_f as cyc, cbm_av.ctxt_file as ctxt, cbm_av.station as stt
            WHERE date_part('year', ctxt.date_record) = 2021 
            AND ctxt.pvs_stationid is not null 
            AND stt.valid_p = 1
            AND ind.id_cycle NOT IN %s
            AND ind.id_cycle = cyc.id_cycle
            AND cyc.id_ctxt = ctxt.id_ctxt
            AND ctxt.pvs_stationid = stt.station_short_name
            """, (self.excluded,))
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
            cur.execute(f"""
            SELECT 
                ind.id_ind as X_id,
                ctxt.vehicule_sncf as veh_id,
                ctxt.dcu as dcu_id, 
                ctxt.date_record as X_tsp,
                ctxt.temperature,
                ctxt.pvs_stationid,
                ctxt.mission_code,
                {query_filler}
            FROM cbm_av.ind_por_f as ind, cbm_av.cycle_por_f as cyc, cbm_av.ctxt_file as ctxt, cbm_av.station AS stt
            WHERE date_part('year', ctxt.date_record) = 2021 
            AND date_part('week', ctxt.date_record) = {self.weeks[self.current]} 
            AND ctxt.pvs_stationid is not null 
            AND stt.valid_p = 1
            AND ind.id_cycle NOT IN %s
            AND ind.id_cycle = cyc.id_cycle 
            AND cyc.id_ctxt = ctxt.id_ctxt
            AND ctxt.pvs_stationid = stt.station_short_name
            ORDER BY date_record
            """, (self.excluded,))

            # increment the iterator index
            self.current += 1

            # each data point: X (data), X_id, sys_id, X_timestamp
            res = []
            for item in cur.fetchall():
                X = np.array([float(indi) if indi is not None else np.nan for indi in item[7:]])
                # standardization (cheat: used precomputed mean and standard deviation...)
                X = (X - self._mean) / self._std
                # min-max scaling to fit in range [0, 1]
                # X = (X - self._min) / (self._max - self._min)
                # avoid rows with NaN values
                if not np.isnan(sum(X)):
                    syst_id = f'{item[1]}_{item[2]}'
                    ctxt = {'temperature': item[4], 'station_id': item[5], 'mission_code': item[6]}
                    instance = Instance(data=X, data_id=item[0], sys_id=syst_id, tsp=item[3].timestamp(), ctxt=ctxt)
                    res.append(instance)

            return res


class R2NFeatureFeeder:
    pass
