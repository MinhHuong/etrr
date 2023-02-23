"""
IMPLEMENT THE GUI ON THE CLIENT'S SIDE
"""

# PyQT libraries
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QTimer

# python's native
import json
import os

# data manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# my own module
from database.dbmanager import DBConnection
from gui.gui_annotation import GUIAnnotation
from tools.utils import preprocess, Query, Marker, CandidateByExtractor


class GUIUser(QApplication):
    """This class implements a simple GUI that updates the number of the queries in waiting,
    and allows the user to enter the annotation mode"""

    def __init__(self, conf, dbconn_):
        """

        Parameters
        ----------
        conf
        dbconn_
        """
        super().__init__([])

        # config
        self.config = conf

        # connection to the database
        self.dbconn = dbconn_

        # create the window
        self.window = QWidget()
        self.window.setWindowTitle('Interactive Cycle Extraction (Client)')
        self.window.setFixedWidth(400)
        self.window.setFixedHeight(150)

        # signal to end the annotation process
        self.stop = False

        # set up the layout
        layout = QVBoxLayout()

        # create the text that counts the number of queries in waiting
        self.lb_query_count = QLabel(f'{self.dbconn.get_nb_unanswered_queries()} queries in waiting')
        self.lb_query_count.setFont(QFont('Arial', 18))
        self.lb_query_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lb_query_count)

        # create the button that allows the users to enter the annotation
        self.btn_start = QPushButton('Start annotating')
        self.btn_start.setFont(QFont('Arial', 14))
        self.btn_start.clicked.connect(self._click_start_anno)
        layout.addWidget(self.btn_start)

        # create the button that allows the users to stop the annotation
        self.btn_stop = QPushButton('Stop annotating')
        self.btn_stop.setFont(QFont('Arial', 14))
        self.btn_stop.clicked.connect(self._click_stop_anno)
        layout.addWidget(self.btn_stop)

        # the timer to signal new queries available
        self.timer_query = QTimer()  # the timer of PyQT
        self.timer_query.timeout.connect(self._timeout_new_queries)
        self.timer_query.start(self.config['gui']['interval_new_queries'])  # periodically check for new queries

        # the timer to fetch a new query to the GUI annotation
        self.timer_process = QTimer()
        self.timer_process.timeout.connect(self._timeout_process_query)

        # the GUI for the annotation
        n_exts = len(config['ensemble'].keys())
        ext_names = list(config['ensemble'].keys())
        self.gui_anno = GUIAnnotation(n_exts=n_exts, ext_names=ext_names, atts=self.config['atts'])

        # finish things up
        self.window.setLayout(layout)
        self.window.show()
        self.exec()

    #region button event
    def _click_start_anno(self):
        """
        Start the annotation if the button "Start annotating" is clicked
        Returns
        -------

        """
        print('Start the annotation!')
        # afterwards, disable the START button and enable the STOP button
        self.btn_start.setDisabled(True)
        self.btn_stop.setEnabled(True)
        self.timer_process.start(self.config['gui']['interval_process'])

    def _click_stop_anno(self):
        """
        Stop the annotation if the button "Stop annotating" is clicked

        Returns
        -------

        """
        print('Stop the annotation!')
        # afterwards, disable the STOP button and enable the START button
        self.btn_stop.setDisabled(True)
        self.btn_start.setEnabled(True)
        self.timer_process.stop()
        plt.close('gui')
    #endregion

    #region timer events
    def _timeout_new_queries(self):
        """
        Check for new queries periodically to update the number of queries on the welcome interface

        Returns
        -------

        """
        self._update_nb_queries()

    def _update_nb_queries(self):
        """
        Update the number of waiting queries

        Returns
        -------

        """
        self.lb_query_count.setText(f'{self.dbconn.get_nb_unanswered_queries()} queries in waiting')
        self.lb_query_count.update()

    def _timeout_process_query(self):
        """
        Retrieve an unanswered official query from the database and fetch it to the GUI

        Returns
        -------

        """
        # get all cycles of one query from the database
        markers = self.dbconn.get_one_official_query()
        if len(markers) == 0:
            return

        # stop the timer temporarily while processing a new cycle
        self.timer_process.stop()

        fname = markers[0][0]  # the filename is the same for all markers in the same query
        input_len = markers[0][4]    # the input length is the same for all markers in the same query

        # read the full input data in and preprocess it
        x = pd.read_pickle(os.path.join(self.config['inpath'], fname))
        x = preprocess(X=x, case=self.config['dataset'])
        if x is None:
            return
        x = x[['time'] + self.config['atts']]  # keep the relevant attributes only (add time!!!)

        # transform it to a form compatible for the GUIAnnotation
        qr_markers = {
            'activity': CandidateByExtractor(flattened=np.zeros(input_len), markers=[]),
            'autoencoder': CandidateByExtractor(flattened=np.zeros(input_len), markers=[]),
            'expert': CandidateByExtractor(flattened=np.zeros(input_len), markers=[])}
        # one line = [fname, mstart, mend, no_cyc, input_len, extractor, m_id]
        for mk in markers:
            mstart, mend, no_cyc, input_len, ext, mid = mk[1], mk[2], mk[3], mk[4], mk[5], mk[6]
            # recreate the flattened array
            qr_markers[ext].flattened[mstart:mend+1] = no_cyc
            # make one marker item
            cyc_data = x.iloc[mstart:mend]
            mk_item = Marker(mid=mid, mstart=mstart, mend=mend, no_cyc=no_cyc, label='', ext=ext, data=cyc_data,
                             input_len=input_len)
            qr_markers[ext].markers.append(mk_item)

        # make the candidates and create the query
        query = Query(X_id=fname, X=x, candidates=qr_markers)

        # invoke the GUI annotation
        selected = self.gui_anno.show_new_query(query)
        for s in selected:
            print(s)
        print()

        # update the number of queries processed
        self._update_nb_queries()

        # save the markers as official ONLY IF the expert selected at least one cycle
        self.dbconn.add_feedback(id_query=query.query_id, markers=selected)

        # once it's done, start the timer again
        self.timer_process.start(self.config['gui']['interval_process'])
    #endregion


if __name__ == '__main__':
    # read the config file
    with open('config/config.json', 'r') as f:
        config = json.load(f)

    # create a connection to the database (/!\ DO NOT RESET ANYTHING)
    dbconn = DBConnection(dbname=config['database']['dbname'],
                          user=config['database']['user'],
                          pwd=config['database']['pwd'])

    # start the GUI
    gui = GUIUser(config, dbconn)
