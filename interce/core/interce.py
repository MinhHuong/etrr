"""
CORE FUNCTIONALITY OF INTERCE
"""
import time
from typing import List

import numpy as np
import pandas as pd
from core.memory import Memory
from core.knowledge import Knowledge
from core.extractors import Ensemble
from tools.utils import CandidateByExtractor, FeedbackItem
import psutil as ps
import os


class InterCE:
    """
    Implement the core functionalities of InterCE
    """

    def __init__(self, config, dbconn, expe_id):
        """Initialize the framework"""

        # the memory M
        self.memory = Memory(sim_thres=config["memory"]['similarity_thresold'],
                             query_thres=config["memory"]['query_threshold'],
                             dbconn=dbconn)

        # the knowledge K
        self.knowledge = Knowledge(n_clusters=config["knowledge"]['n_clusters'], atts=config['atts'])

        # the ensemble of extractors
        self.ensemble = Ensemble(config)

        # the database connection
        self.dbconn = dbconn

        # the attributes
        self.atts = config['atts']

        # experiment-related, not concerning the core of InterCE
        self.expe_id = expe_id
        self.n_processed_files = 0
        self.n_queries = 0
        self.n_buffered_queries = 0
        self.n_official_queries = 0
        self.total_time = 0
        self.process = ps.Process(os.getpid())
        self.n_cycles = 0
        self.n_auto_cycles = 0
        self.n_human_cycles = 0
        self.n_auto = 0     # number of times InterCE works in auto mode
        self.n_human = 0    # number of times InterCE works in human mode
        self.n_feedback = 0
        self.time_ensemble = 0

    def process_input(self, X: pd.DataFrame, X_id):
        """
        Process a new input X. Three cases may happen:
        1. InterCE has enough knowldge to extract cycles from X -> return the results directly
        2. InterCE has seen X but hasn't received the feedback to X -> create a buffered query Qr(X)
        3. InterCE has never seen X -> create an official query Qr(X)

        Parameters
        ----------
        X: np.ndarray
            The input data
        X_id
            The ID associated to this input

        Returns
        -------

        """
        # FIXED start the timer
        start = time.perf_counter()

        # extract the candidates using the ensemble, the results are separated by extractors
        candidates: dict[str, CandidateByExtractor] = self.ensemble.predict(X, X_id)
        self.time_ensemble += (time.perf_counter() - start)

        # from this point onward, keep the most important attributes only
        x = X[self.atts]

        # if InterCE has seen X
        has_seen, motif_id = self.memory.has_seen_data(x, X_id)   # check and retrieve the motif ID at the same time
        if has_seen:
            # if the query for X has been processd --> automatically select cycles
            if self.memory.has_processed_query_of(motif_id):
                ext, results = self.knowledge.select_cycles(candidates)
                if results is not None:
                    self._save_results(ext, X_id, results)
                    self.n_cycles += len(results.markers)
                    self.n_auto_cycles += len(results.markers)
                    self.n_auto += 1
            else:
                query = self.memory.create_query(X_id, x, candidates)
                self._mark_buffered(query, motif_id)
        # if X is new
        else:
            query = self.memory.create_query(X_id, x, candidates)
            self._mark_official(query)

        # end the timer
        end = time.perf_counter()
        self.total_time += (end - start)

        # file done processed
        self.n_processed_files += 1

    def save_metrics(self):
        """
        Save the metrics/time measurement in the database

        Returns
        -------

        """
        metrics = {
            # global measurement
            'total_time': self.total_time,
            'mem_usage': self.process.memory_info().rss,
            'n_cycles': self.n_cycles,
            'n_auto_cycles': self.n_auto_cycles,
            'n_human_cycles': self.n_human_cycles,
            'n_auto': self.n_auto,
            'n_human': self.n_human,
            'n_files': self.n_processed_files,
            # motif-related metrics
            'time_m_motifs': self.memory.time_motifs,
            'n_motifs': self.memory.n_motifs,
            # ensemble-related metrics
            'time_ensemble': self.time_ensemble,
            # query-related metrics
            'n_total_queries': self.n_queries,
            'n_buffered_queries': self.n_buffered_queries,
            'n_official_queries': self.n_official_queries,
            # feedback-related metrics
            'n_feedback': self.n_feedback,
            # knowledge-related metrics
            'time_k_select': self.knowledge.time_select,
            'time_k_update': self.knowledge.time_update,
        }

        # persist all in the database
        self.dbconn.save_metrics(expe_id=self.expe_id, metrics=metrics)

    def _mark_buffered(self, query, id_query_ref):
        """
        Mark a query as buffered

        Parameters
        ----------
        query

        Returns
        -------

        """
        self.dbconn.mark_buffered(query, id_query_ref)
        self.n_buffered_queries += 1
        self.n_queries += 1

    def _mark_official(self, query):
        """
        Mark a query as official

        Parameters
        ----------
        query

        Returns
        -------

        """
        self.dbconn.mark_official(query)
        self.n_official_queries += 1
        self.n_queries += 1

    def _save_results(self, ext, X_id, results: CandidateByExtractor):
        """
        Save the final cycles

        Parameters
        ----------
        ext
        X_id
        results: CandidateByExtractor

        Returns
        -------

        """
        self.dbconn.save_results(ext=ext, X_id=X_id, results=results)

    def _save_results_from_buffered_queries(self, X_id, results: CandidateByExtractor):
        """
        Save the cycles selected from buffered queries.
        Instead of adding new markers, update the existing ones, as if receiving feedback from a human

        Parameters
        ----------
        X_id
        results

        Returns
        -------

        """
        self.dbconn.solve_buffered_queries(query_fname=X_id, results=results)

    def process_feedback(self):
        """
        The module to process the feedback is to be launched in its own thread

        Returns
        -------

        """
        # check if there are unsolved feedback in the database, if there are, update the knowledge
        n_new_feedback = self.dbconn.get_nb_unprocessed_feedback()
        if n_new_feedback > 0:
            # update the knowledge on the feedback
            feedback, feedback_ids = self.dbconn.get_feedback_data()

            # only update if the feedback data are not empty
            if len(feedback) != 0:
                self.knowledge.update(feedback)
                # FIXED add feedback log
                for clabel, centroid in self.knowledge.clusters.items():
                    self.dbconn.add_feedback_log(label=clabel, data=centroid.tolist())

            # FIXED record the number of cycles extracted (human mode)
            self.n_cycles += len(feedback)
            self.n_human_cycles += len(feedback)
            self.n_human += len(feedback_ids)
            self.n_feedback += n_new_feedback

            # solve the buffered queries associated to the query of this feedback as well
            for fid in feedback_ids:
                buf_queries = self.dbconn.get_buffered_queries_of_feedback(fid)
                for X_id, candidates in buf_queries.items():
                    ext, results = self.knowledge.select_cycles(candidates)
                    self.n_auto += 1
                    if results is not None:
                        self.dbconn.solve_buffered_queries(query_fname=X_id, results=results)
                        self.n_cycles += len(results.markers)
                        self.n_auto_cycles += len(results.markers)

            # mark the feedback as solved (only at the end where everything is done)
            self.dbconn.mark_feedback_solved(feedback_ids)
