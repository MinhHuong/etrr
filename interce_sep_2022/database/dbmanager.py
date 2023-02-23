"""
MANAGE CONNECTION TO THE POSTGRES DATABASE
"""
from typing import List, Tuple

import psycopg as pg
from tools.utils import SelectedMarker, FeedbackItem, CandidateByExtractor, Marker
import time
import numpy as np


class DBConnection:
    """
    Manage connection to a database
    """

    #region Creating-closing-resetting
    def __init__(self, dbname: str, user: str, pwd: str):
        """

        Parameters
        ----------
        dbname: str
            Database name
        user: str
            Username
        pwd: str
            Password
        """
        self.dbname = dbname
        self.user = user
        self.pwd = pwd
        self.conn = pg.connect(dbname=self.dbname, user=self.user, password=self.pwd)

    def exit(self):
        """
        Close the database connection

        Returns
        -------

        """
        self.conn.commit()
        self.conn.close()

    def reset(self):
        """
        Clean the data in all tables (except from the metrics and experiment logs)

        Returns
        -------

        """
        with self.conn.cursor() as cur:
            # marker table
            cur.execute("""DELETE FROM marker;""")
            cur.execute("""ALTER SEQUENCE marker_id_seq RESTART WITH 1;""")
            # feedback table
            cur.execute("""DELETE FROM feedback;""")
            cur.execute("""ALTER SEQUENCE feedback_id_seq RESTART WITH 1;""")
            # feedback log
            cur.execute("""DELETE FROM feedback_log;""")
            # query table
            cur.execute("""DELETE FROM ce_query;""")
            cur.execute("""ALTER SEQUENCE ce_query_id_seq RESTART WITH 1;""")
            # motif table
            cur.execute("""DELETE FROM motif;""")
            cur.execute("""ALTER SEQUENCE motif_id_seq RESTART WITH 1;""")
            # commit everything
            self.conn.commit()
    #endregion

    #region Query-related
    def mark_buffered(self, query, id_query_ref):
        """
        Create a new buffered query

        Parameters
        ----------
        query
        id_query_ref

        Returns
        -------

        """
        with self.conn.cursor() as cur:
            # find the correct serial ID of id_query_ref
            cur.execute("""SELECT id FROM ce_query WHERE fname = %s""", (id_query_ref,))
            id_ref = cur.fetchone()
            # if no query with such ID exist, which should not happen, some bullshit did happen, so we skip this case
            if id_ref is None:
                return
            else:
                id_ref = id_ref[0]

            # insert the query
            cur.execute(
                """
                INSERT INTO ce_query(fname, is_answered, is_buffered, id_ref)
                VALUES (%(fname)s, %(is_answered)s, %(is_buffered)s, %(id_ref)s)
                """,
                {'fname': query.query_id, 'is_answered': False, 'is_buffered': True, 'id_ref': id_ref}
            )

            # get the ID of the newly inserted query
            cur.execute("""SELECT id FROM ce_query WHERE fname = %s""", (query.query_id,))
            id_qr = cur.fetchone()[0]

            # insert the markers associated to this query
            # markers = query.to_db_records()
            # for mk in markers:
            #     self.add_marker(X_id=mk['query_id'], mstart=mk['mstart'], mend=mk['mend'], no_cyc=mk['no_cyc'],
            #                     input_len=mk['input_len'], label='', data=mk['data'], mode='',
            #                     extractor=mk['extractor'], id_query=id_qr, id_feedback=None, is_official=False)

            candidate: CandidateByExtractor
            for ext, candidate in query.candidates.items():
                for mk in candidate.markers:
                    self.add_marker(X_id=query.query_id, mstart=mk.mstart, mend=mk.mend, no_cyc=mk.no_cyc,
                                    input_len=mk.input_len, label=mk.label, data=mk.data.tolist(), mode='',
                                    extractor=ext, id_query=id_qr, id_feedback=None, is_official=False)

            # commit all
            self.conn.commit()

    def mark_official(self, query):
        """
        Save a query to the database and mark it official

        Parameters
        ----------
        query

        Returns
        -------

        """
        with self.conn.cursor() as cur:
            # insert the query
            cur.execute(
                """
                INSERT INTO ce_query(fname, is_answered, is_buffered, id_ref)
                VALUES (%(fname)s, %(is_answered)s, %(is_buffered)s, %(id_ref)s)
                """,
                {'fname': query.query_id, 'is_answered': False, 'is_buffered': False, 'id_ref': None}
            )

            # get the ID of the newly inserted query
            cur.execute("""SELECT id FROM ce_query WHERE fname = %s""", (query.query_id,))
            id_qr = cur.fetchone()[0]

            # insert the markers associated to this query
            # markers = query.to_db_records()
            # for mk in markers:
            #     self.add_marker(X_id=mk['query_id'], mstart=mk['mstart'], mend=mk['mend'], no_cyc=mk['no_cyc'],
            #                     input_len=mk['input_len'], label='', data=mk['data'], mode='',
            #                     extractor=mk['extractor'], id_query=id_qr, id_feedback=None, is_official=False)

            candidate: CandidateByExtractor
            for ext, candidate in query.candidates.items():
                for mk in candidate.markers:
                    self.add_marker(X_id=query.query_id, mstart=mk.mstart, mend=mk.mend, no_cyc=mk.no_cyc,
                                    input_len=mk.input_len, label=mk.label, data=mk.data.tolist(), mode='',
                                    extractor=ext, id_query=id_qr, id_feedback=None, is_official=False)

            # commit all
            self.conn.commit()

    def is_query_processed(self, X_id):
        """
        Check if the query associated to this X_id has been processed

        Parameters
        ----------
        X_id

        Returns
        -------

        """
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT is_answered FROM ce_query WHERE fname = %s""",
                (X_id,)
            )
            res = cur.fetchone()
            return res[0] if res is not None else False

    def get_nb_unanswered_queries(self) -> int:
        """
        Retrieve the OFFICIAL queries that have not yet been answered

        Returns
        -------
        int
            Number of official queries that have not yet been answered
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT COUNT(*) FROM ce_query WHERE is_answered IS FALSE AND is_buffered IS FALSE"""
            )
            res = cur.fetchone()
            return res[0] if res is not None else 0

    def get_total_unanswered_queries(self) -> int:
        """
        Get the total number of unanswered queries

        Returns
        -------

        """
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT COUNT(*) FROM ce_query WHERE is_answered IS FALSE"""
            )
            res = cur.fetchone()
            return res[0] if res is not None else 0

    def get_one_official_query(self):
        """
        Retrieve one unanswered query randomly
        (in the future, we may retrieve queries in the other of time)

        Returns
        -------

        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT fname, mstart, mend, no_cyc, input_len, extractor, m.id
                FROM marker AS m, 
                    (SELECT id FROM ce_query WHERE is_buffered IS FALSE and is_answered IS FALSE LIMIT 1) AS q
                WHERE m.id_query = q.id AND m.is_official IS FALSE"""
            )
            res = cur.fetchall()
            return res if res is not None else []
    #endregion

    #region Memory-related
    def save_motif(self, motif_id, motif_data):
        """
        Save a new motif in

        Parameters
        ----------
        motif_id
            The ID of the motif (the filename)
        motif_data
            The data of the motif (the whole series)

        Returns
        -------

        """
        with self.conn.cursor() as cur:
            # add the new motif in
            cur.execute(
                """INSERT INTO motif(fname, data) VALUES (%(fname)s, %(data)s);""",
                {'fname': motif_id, 'data': motif_data}
            )
            # commit
            self.conn.commit()
    #endregion

    #region Feedback-related
    def add_feedback(self, id_query, markers: List[SelectedMarker]):
        """
        Add a new feedback

        Parameters
        ----------
        id_query
            ID of the query this feedback is for
            (note that this is the filename, instead of the true serial ID of the query in the DB)
        markers
            List of selected markers (ID and labels)

        Returns
        -------

        """
        with self.conn.cursor() as cur:
            # first, get the correct serial ID of the query
            cur.execute("""SELECT id FROM ce_query WHERE fname = %s""", (id_query,))
            res = cur.fetchone()
            if res is None:
                return
            id_qr = res[0]

            # also, mark this query as answered
            cur.execute("""UPDATE ce_query SET is_answered = True WHERE id = %s""", (id_qr,))

            # insert a new feedback
            cur.execute(
                """
                INSERT INTO feedback(id_query, is_solved, is_empty) 
                VALUES (%(id_query)s, %(is_solved)s, %(is_empty)s)
                """,
                {'id_query': id_qr, 'is_solved': False, 'is_empty': True if len(markers) == 0 else False}
            )

            # get the ID of the newly inserted feedback
            cur.execute("""SELECT id FROM feedback WHERE id_query = %s""", (id_qr,))
            res = cur.fetchone()
            if res is None:
                return
            id_fb = res[0]

            # modify the markers
            for mk in markers:
                cur.execute(
                    """
                    UPDATE marker 
                    SET id_feedback = %(id_fb)s, cyc_label = %(label)s, ext_mode = %(mode)s, is_official = %(is_off)s
                    WHERE id = %(id_mk)s
                    """,
                    {'id_fb': id_fb, 'id_mk': mk.mid, 'label': mk.mlabel, 'mode': 'human', 'is_off': True}
                )

            # commit all
            self.conn.commit()

    def add_feedback_log(self, label, data):
        """
        Update the data in the feedback after an update

        Parameters
        ----------
        label
        data

        Returns
        -------

        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback_log(label, data) VALUES (%(label)s, %(data)s)
                ON CONFLICT (label) DO UPDATE SET data = %(data_u)s
                """,
                ({'label': label, 'data': data, 'data_u': data})
            )
            # commit
            self.conn.commit()

    def has_unprocessed_feedback(self) -> bool:
        """
        Check if there are any unprocessed NON-EMPTY feedbacks

        Returns
        -------

        """
        with self.conn.cursor() as cur:
            cur.execute("""SELECT COUNT(*) FROM feedback WHERE is_solved IS FALSE AND is_empty IS FALSE""")
            res = cur.fetchone()
            return res[0] > 0 if res is not None else False

    def get_nb_unprocessed_feedback(self) -> int:
        """
        Get the number of unprocessed NON-EMPTY feedback

        Returns
        -------

        """
        with self.conn.cursor() as cur:
            # include empty feedback as well
            cur.execute("""SELECT COUNT(*) FROM feedback WHERE is_solved IS FALSE""")
            res = cur.fetchone()
            return res[0] if res is not None else 0

    def get_feedback_data(self) -> Tuple[List[FeedbackItem], List]:
        """
        Retrieve all the cycles confirmed by the human expert and has not yet been solved
        (the markers with id_feedback not null)


        Returns
        -------

        """
        with self.conn.cursor() as cur:
            # cur.execute("""SELECT cyc_data, cyc_label, id_feedback FROM marker WHERE id_feedback IS NOT NULL""")

            # first, get the data of the unsolved feedback
            fb_data = []
            cur.execute(
                """
                SELECT cyc_data, cyc_label, id_feedback 
                FROM marker m, feedback f 
                WHERE f.is_solved IS FALSE AND id_feedback IS NOT NULL 
                AND m.id_feedback = f.id 
                """
            )
            res = cur.fetchall()
            fb_data = [] if res is None else [FeedbackItem(mdata=r[0], mlabel=r[1], fid=r[2]) for r in res]

            # then, get the separate list of feedback ID's (will make a difference if there are empty feedback)
            cur.execute("""SELECT id FROM feedback WHERE is_solved IS FALSE""")
            res = cur.fetchall()
            fb_ids = [] if res is None else [r[0] for r in res]

            # return both the data and the ID
            return fb_data, fb_ids

    def mark_feedback_solved(self, fids):
        """
        Mark the given feedback ID's as solved

        Parameters
        ----------
        fids

        Returns
        -------

        """
        with self.conn.cursor() as cur:
            # mark the indicated feedback as solved
            cur.execute(
                """UPDATE feedback SET is_solved = TRUE WHERE id = ANY(%s)""",
                (fids,)
            )
            self.conn.commit()

    def get_buffered_queries_of_feedback(self, id_feedback) \
            -> dict[str, dict[str, CandidateByExtractor]]:
        """
        Get the buffered queries associated to a query of a given feedback

        Parameters
        ----------
        id_feedback

        Returns
        -------
        dict
            Indexed by X_id (filename), value is the candidates for each query (file)
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.fname, m.mstart, m.mend, m.no_cyc, m.input_len, m.extractor, m.id, m.cyc_data
                FROM ce_query q, marker m, feedback f
                WHERE f.id = %s AND q.id_ref = f.id_query AND m.id_query = q.id 
                """,
                (id_feedback,)
            )
            markers = cur.fetchall()
            if markers is None or len(markers) == 0:
                return {}
            else:
                # create the candidates in its correct form
                result = {}
                for mk in markers:
                    X_id, mstart, mend, no_cyc, input_len, ext, mid, cdata = \
                        mk[0], mk[1], mk[2], mk[3], mk[4], mk[5], mk[6], mk[7]
                    if X_id not in result.keys():
                        result[X_id] = {}
                    if ext not in result[X_id].keys():
                        result[X_id][ext] = CandidateByExtractor(flattened=np.zeros(input_len), markers=[])
                    # recreate the flattened array
                    result[X_id][ext].flattened[mstart:mend+1] = no_cyc
                    mk_item = Marker(mid=mid, mstart=mstart, mend=mend, no_cyc=no_cyc, label='', ext=ext, data=cdata,
                                     input_len=input_len)
                    result[X_id][ext].markers.append(mk_item)

                # return the result
                return result

    def solve_buffered_queries(self, query_fname, results: CandidateByExtractor):
        """
        Solve a buffered query, mark all the chosen markers as official

        Parameters
        ----------
        query_fname
        results

        Returns
        -------

        """
        # set all the selected markers as official
        # mids = [mk.mid for mk in results.markers]
        # print('IN DBCONN:', mids)

        with self.conn.cursor() as cur:
            # set the markers as official
            # cur.execute("""UPDATE marker SET is_official = True AND ext_mode = 'auto' WHERE id = ANY(%s)""", ([mids]))

            # set the marker as official and update its label
            for mk in results.markers:
                cur.execute(
                    """
                    UPDATE marker 
                    SET is_official = True, ext_mode = 'auto', cyc_label = %(clabel)s
                    WHERE id = %(mid)s
                    """,
                    {'clabel': mk.label, 'mid': mk.mid}
                )

            # update the query as solved
            cur.execute("""UPDATE ce_query SET is_answered = True WHERE fname = %s""", (query_fname,))

            # commit all
            self.conn.commit()
    #endregion

    def add_marker(self, X_id, mstart, mend, no_cyc, input_len, label, data, extractor, mode, id_query, id_feedback,
                   is_official):
        """
        Insert a new marker (i.e., a new cycle)

        Parameters
        ----------
        X_id
        mstart
        mend
        no_cyc
        input_len
        label
        data
        extractor
        mode
        id_query
        id_feedback
        is_official

        Returns
        -------

        """
        with self.conn.cursor() as cur:
            # convert to python list, if cycle data are in form of np.ndarray
            if isinstance(data, np.ndarray):
                data = data.tolist()
            # insert the marker
            cur.execute(
                """
                INSERT INTO marker(fname, mstart, mend, no_cyc, input_len, cyc_label, cyc_data, extractor, ext_mode, 
                                    id_query, id_feedback, is_official)
                VALUES 
                (%(fname)s, %(mstart)s, %(mend)s, %(no_cyc)s, %(input_len)s, %(label)s, %(data)s, %(ext)s, %(mode)s, 
                %(id_qr)s, %(id_fb)s, %(is_off)s)
                """,
                {'fname': X_id, 'mstart': mstart, 'mend': mend, 'no_cyc': no_cyc, 'input_len': input_len,
                 'label': label, 'data': data, 'ext': extractor, 'mode': mode,
                 'id_qr': id_query, 'id_fb': id_feedback, 'is_off': is_official}
            )

            # commit
            self.conn.commit()

    def save_results(self, ext, X_id, results: CandidateByExtractor):
        """
        Save the chosen candidates by inserting new markers in the database

        Parameters
        ----------
        ext
            Name of the chosen extractor
        X_id
            ID of the input (filename)
        results

        Returns
        -------

        """
        for mk in results.markers:
            self.add_marker(X_id=X_id, mstart=mk.mstart, mend=mk.mend, no_cyc=mk.no_cyc, input_len=mk.input_len,
                            label=mk.label, data=mk.data, extractor=ext, mode='auto', id_query=None, id_feedback=None,
                            is_official=True)

    def add_experiment(self, expe_name, expe_desc) -> int:
        """
        Add a new experiment in the database, return the ID of the newly added records

        Parameters
        ----------
        expe_name
        expe_desc

        Returns
        -------

        """
        with self.conn.cursor() as cur:
            cur_time = time.time()
            # add the new experiment in
            cur.execute(
                """INSERT INTO experiment(expe_name, expe_date, expe_desc) VALUES (%(name)s, %(date)s, %(desc)s)""",
                {'name': expe_name, 'date': cur_time, 'desc': expe_desc}
            )
            # retrieve the ID of the newly added experiment
            cur.execute(
                """SELECT id FROM experiment WHERE expe_name = %(name)s AND expe_date = %(date)s""",
                {'name': expe_name, 'date': cur_time}
            )
            res = cur.fetchone()
            # commit all
            self.conn.commit()
            return res[0] if res is not None else -1

    def save_metrics(self, expe_id, metrics: dict):
        """
        Save the measurement (time/memory usage) in metrics

        Parameters
        ----------
        expe_id
            ID of the experiment in the database
        metrics: dict
            Metrics to save in form of dictionary

        Returns
        -------

        """
        with self.conn.cursor() as cur:
            # check: if no row with (id_expe, n_files) exists, add new
            # add a new metrics in
            cur.execute(
                """
                INSERT INTO metrics(id_expe, n_files, total_time, mem_usage, n_cycles, n_auto_cycles, n_human_cycles,
                n_auto, n_human, is_training,
                n_motifs, n_total_queries, n_buffered_queries, n_official_queries, n_feedback, time_m_motifs, 
                time_k_select, time_k_update, time_ensemble)
                VALUES (%(id_expe)s, %(n_files)s, %(total_time)s, %(mem_usage)s, %(n_cycles)s, %(n_auto_cycles)s, 
                %(n_human_cycles)s, %(n_auto)s, %(n_human)s, %(is_training)s,
                %(n_motifs)s, %(n_total_queries)s, %(n_buffered_queries)s, %(n_official_queries)s,
                %(n_feedback)s, %(time_m_motifs)s, %(time_k_select)s, %(time_k_update)s, %(time_ensemble)s)
                ON CONFLICT (id_expe, n_files)
                DO UPDATE SET
                total_time = %(total_time_u)s, mem_usage = %(mem_usage_u)s, n_cycles = %(n_cycles_u)s, 
                n_auto_cycles = %(n_auto_cycles_u)s, n_human_cycles = %(n_human_cycles_u)s, 
                n_auto = %(n_auto_u)s, n_human = %(n_human_u)s, is_training = %(is_training_u)s, 
                n_motifs = %(n_motifs_u)s, 
                n_total_queries = %(n_total_queries_u)s, n_buffered_queries = %(n_buffered_queries_u)s, 
                n_official_queries = %(n_official_queries_u)s, n_feedback = %(n_feedback_u)s, 
                time_m_motifs = %(time_m_motifs_u)s, time_k_select = %(time_k_select_u)s, 
                time_k_update = %(time_k_update_u)s, time_ensemble = %(time_ensemble_u)s
                """,
                {
                    # INSERT NEW
                    'id_expe': expe_id, 'n_files': metrics['n_files'], 'total_time': metrics['total_time'],
                    'mem_usage': metrics['mem_usage'], 'n_cycles': metrics['n_cycles'],
                    'n_auto_cycles': metrics['n_auto_cycles'], 'n_human_cycles': metrics['n_human_cycles'],
                    'n_auto': metrics['n_auto'], 'n_human': metrics['n_human'], 'is_training': metrics['is_training'],
                    'n_motifs': metrics['n_motifs'], 'n_total_queries': metrics['n_total_queries'],
                    'n_buffered_queries': metrics['n_buffered_queries'],
                    'n_official_queries': metrics['n_official_queries'], 'n_feedback': metrics['n_feedback'],
                    'time_m_motifs': metrics['time_m_motifs'], 'time_k_select': metrics['time_k_select'],
                    'time_k_update': metrics['time_k_update'], 'time_ensemble': metrics['time_ensemble'],
                    # UPDATE
                    'total_time_u': metrics['total_time'], 'mem_usage_u': metrics['mem_usage'],
                    'n_cycles_u': metrics['n_cycles'], 'n_auto_cycles_u': metrics['n_auto_cycles'],
                    'n_human_cycles_u': metrics['n_human_cycles'], 'n_auto_u': metrics['n_auto'],
                    'n_human_u': metrics['n_human'], 'is_training_u': metrics['is_training'],
                    'n_motifs_u': metrics['n_motifs'],
                    'n_total_queries_u': metrics['n_total_queries'],
                    'n_buffered_queries_u': metrics['n_buffered_queries'],
                    'n_official_queries_u': metrics['n_official_queries'], 'n_feedback_u': metrics['n_feedback'],
                    'time_m_motifs_u': metrics['time_m_motifs'], 'time_k_select_u': metrics['time_k_select'],
                    'time_k_update_u': metrics['time_k_update'], 'time_ensemble_u': metrics['time_ensemble'],
                 }
            )
            # commit everything
            self.conn.commit()

    def has_finished(self):
        """
        Check if all the queries have been answered
        and all the feedback has been processed

        Returns
        -------

        """
        with self.conn.cursor() as cur:
            # get the number of unanswered queries
            cur.execute("""SELECT COUNT(*) FROM ce_query WHERE is_answered IS FALSE""")
            tmp = cur.fetchone()
            n_unanswered = tmp[0] if tmp is not None else 0

            # get the number of unsolved NON-EMPTY feedback
            cur.execute("""SELECT COUNT(*) FROM feedback WHERE is_solved IS FALSE AND is_empty IS FALSE""")
            tmp = cur.fetchone()
            n_unsolved = tmp[0] if tmp is not None else 0

            # true if both are 0, false otherwise
            return n_unanswered == 0 and n_unsolved == 0
