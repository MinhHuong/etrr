"""
ALL EXTRACTORS IN INTERCE
"""
import os

import keras.backend

from tools.clean_cycles import clean_cycles

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import abc
import numpy as np
import pandas as pd
from more_itertools import consecutive_groups
from tools.nat_marker import NATMarker
from keras.models import load_model
import tensorflow as tf
from tools.utils import pad_sequences
from scipy.linalg import norm
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks
from tools.utils import CandidateByExtractor, Marker, get_cycles
from tools.compute_gradients import compute_gradients
from tools.remove_spikes import remove_spikes


def _distance(a, b):
    """Normalized euclidean distance"""
    # eucl = math.sqrt(sum([(i - j) * (i - j) for i, j in zip(a, b)]))
    return euclidean(a, b) / np.sqrt(norm(a) * norm(b))


def find_index_in_data(data, tsp) -> int:
    """Finds the index of a timestamp in the original data"""
    for idx, inst in reversed(list(enumerate(data))):
        if inst[0] == tsp:
            return idx
    return -1


class Extractor(abc.ABC):
    """
    Abstract class providing an interface that subclass must abide to
    """

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame, X_id):
        """
        Given an input, extract cycles from it and return:
        - the list of change points (cycle boundaries)
        - the marker as a multiclass array
        - the actual cycle data

        Parameters
        ----------
        X
        X_id

        Returns
        -------

        """
        N = X.shape[0]
        return [], np.zeros(N), []


class ActivityBasedExtractor(Extractor):

    def __init__(self, delta: float, atts):
        """
        Initialize the activity-based extractor

        Parameters
        ----------
        delta
            Threshold of variability of a segment
        atts
            List of variables
        """
        self.delta = delta
        self.atts = atts

    def predict(self, X: pd.DataFrame, X_id):
        """
        Extract cycles

        Parameters
        ----------
        X
        X_id

        Returns
        -------

        """
        # transform it to a numpy array
        x = X[self.atts].to_numpy(copy=True)

        # initialize
        N = len(x)
        cpts = []  # list of change points
        y = np.zeros(N)  # cycle in terms of multiclass array (0 = no cycle, i = i-th cycle)
        cyc_data = []

        # indices of rows that contain non-0 values
        idx_non0 = [i for i in range(N) if any(x[i] != 0)]

        # get groups of consecutive indices whose length > 1
        all_indices = [g for g in [list(_g) for _g in consecutive_groups(idx_non0)] if len(g) > 1]

        # start numbering the cycles (starting from 1)
        num_cyc = 1
        std_data = np.std(x)
        for indices in all_indices:
            segm = x[indices]
            # check if there are activities in this cycle segment (std of the segment / std of the whole data)
            if np.std(segm) / std_data > self.delta:
                y[indices] = num_cyc
                num_cyc += 1
                cyc_data.append(segm)
                cpts += [indices[0], indices[-1] + 1]

        # return the change points (cpts), the numbered array (y), the actual cycle data (cyc_data)
        return cpts, y, cyc_data


class NATExpertSystemExtractor(Extractor):

    def __init__(self, marker_config, atts):
        """
        Initialize the expert system marker
        """
        self.marker = NATMarker(marker_config)
        self.atts = atts

    def predict(self, X: pd.DataFrame, X_id):
        """
        Extract the cycles

        Parameters
        ----------
        X
        X_id

        Returns
        -------

        """
        # get the context from the data
        ctxt = X.loc[0, 'Context']
        ctxt['file_name'] = ctxt['file_name'].replace('.dat', '.pkl')
        ctxt = pd.DataFrame(ctxt, index=[0])
        ctxt.fillna(0)  # replace all NaN by 0
        self.marker.copy_dat_cycle_context(ctxt)

        # extract the cycles
        self.marker.extract_markers(X.copy(), X_id)
        marks = self.marker.get_markers_as_list()

        """
        FORMAT OF MARKERS:
        List of 8 elements: 
        [0] = step open start, [1] = end step open, [2] = door open start, [3] = door open end,
        [4] = door close start, [5] = door close end, [6] = step close start, [7] = step close end
        """
        # format it correspondingly (kinda manually...)
        y, cyc_data, cpts = np.zeros(X.shape[0]), [], []
        x = X[self.atts].values

        # ouverture porte
        if marks[2] != marks[3]:
            y[marks[2]:marks[3] + 1] = 1
            cyc_data += [x[marks[2]:marks[3] + 1]]
            cpts += [marks[2], marks[3] + 1]
        # fermeture porte
        if marks[4] != marks[5]:
            y[marks[4]:marks[5] + 1] = 2
            cyc_data += [x[marks[4]:marks[5] + 1]]
            cpts += [marks[4], marks[5] + 1]

        # return the results
        return cpts, y, cyc_data


class R2NExpertSystemExtractor(Extractor):

    def __init__(self, marker_config, atts):
        """
        Initialize the expert system marker
        """
        self.atts = atts

    def predict(self, X: pd.DataFrame, X_id):
        """
        Extract the cycles

        Parameters
        ----------
        X
        X_id

        Returns
        -------

        """
        # --------------- #
        # Code of Sofiane #
        # --------------- #
        if len(X) > 0:
            # name of the system
            system = 'POR'

            # Convert time to datetime
            X['time'] = pd.to_timedelta(X['time'])

            # Define opening/closing
            df = pd.DataFrame()

            # marker, ctxt_cycle, output_data = [], [], []
            markers = []

            if sum(X['position_moteur_de_la_porte'].dropna()) == 0:
                X['mvt_por'] = 0
            else:
                X['mvt_por'] = 1

                df = compute_gradients(X, df, system)

                # Boolean variation
                boolean_var_thresholds = {'POR': 50, 'MM': 75}
                a = df['var'].map(lambda x: 1 if x > boolean_var_thresholds[system] else 0)
                a_filtered = remove_spikes(a)

                # Cycle starts with rising edge and ends with falling edge
                edges = a_filtered.diff().shift(-1).dropna()
                start_index, end_index = edges[edges == 1].index, edges[edges == -1].index + 1

                # Check if first and last cycles are complete
                if start_index[0] < end_index[0] and start_index[-1] < end_index[-1]:
                    number_cycles = len(start_index)
                elif start_index[0] >= end_index[0] and start_index[-1] >= end_index[-1]:
                    number_cycles = len(start_index) - 2
                else:
                    number_cycles = len(start_index) - 1

                # Identify cycle
                for nb_cycle in range(number_cycles):
                    marker = {'cycle': None, 'mstart': None, 'mend': None, 'sys': None, 'type': None, 'tps': None}
                    ctxt_cycle = {'id_type': None, 'type_label': None, 'temps_cycle': None}
                    # If first cycle is complete, check if duration between start index
                    # and next index is more than 100 ms
                    marker['cycle'] = nb_cycle + 1
                    if start_index[0] < end_index[0]:
                        if df.iloc[start_index[0] + 1].timestamp - df.iloc[start_index[0]].timestamp > 100:
                            start_cycle_index = start_index[nb_cycle] + 1
                        else:
                            start_cycle_index = start_index[nb_cycle]
                        end_cycle_index = end_index[nb_cycle]
                    else:
                        if df.iloc[start_index[0] + 1].time - df.iloc[start_index[0]].time > 100:
                            start_cycle_index = start_index[nb_cycle + 1] + 1
                        else:
                            start_cycle_index = start_index[nb_cycle + 1]
                        end_cycle_index = end_index[nb_cycle + 1]
                    # Cycle d'ouverture avec fermeture temporisée
                    cycle = X[start_cycle_index:end_cycle_index]
                    cycle = cycle[cycle.sleepmode == 0]

                    marker['mstart'] = start_cycle_index
                    marker['mend'] = end_cycle_index

                    marker_sys = {'POR': ['P', 6, 4, 200], 'MM': ['MM', 7, 5, 100]}

                    if cycle.empty:
                        marker['sys'] = marker_sys[system][0]
                        marker['type'] = 'E'
                        marker['tps'] = 0
                        ctxt_cycle['id_type'] = marker_sys[system][1]
                        ctxt_cycle['type_label'] = 'Cycle Vide'
                        ctxt_cycle['temps_cycle'] = 0
                    else:
                        system = X.columns[-1].split('_')[-1].upper()
                        if (system == 'POR' and abs(
                                cycle['position_moteur_de_la_porte'].iloc[0] -
                                cycle['position_moteur_de_la_porte'].iloc[-1]) < 200) or \
                                (system == 'MM' and
                                    (cycle.PMM.isna().all()
                                     or abs(cycle[f'position_moteur_de_la_marche_mobile'].iloc[0] -
                                            cycle['position_moteur_de_la_marche_mobile'].iloc[-1]) < 20)):
                            marker['sys'] = marker_sys[system][0]
                            marker['type'] = 'E'
                            ctxt_cycle['id_type'] = marker_sys[system][2]
                            ctxt_cycle['type_label'] = 'Error deltaP=0'
                        else:
                            if max(cycle['courant_moteur_de_la_porte']) < marker_sys[system][3]:
                                marker['sys'] = marker_sys[system][0]
                                marker['type'] = 'E'
                                ctxt_cycle['id_type'] = marker_sys[system][1]
                                ctxt_cycle['type_label'] = 'Error bruit capteur I'
                            else:
                                if system == 'POR':
                                    if cycle['tension_moteur_de_la_porte'].mean() > 0:  # door opens
                                        marker['sys'] = 'P'
                                        marker['type'] = 'O'
                                        ctxt_cycle['id_type'] = 1
                                        ctxt_cycle['type_label'] = 'ouverture'
                                    else:
                                        # normal closing
                                        if cycle.lt_cf_est_active.sum() + cycle.lt_cf_est_active.sum() > 0:
                                            marker['sys'] = 'P'
                                            marker['type'] = 'F'
                                            ctxt_cycle['id_type'] = 2
                                            ctxt_cycle['type_label'] = 'fermeture'
                                        else:  # slow closing
                                            marker['sys'] = 'P'
                                            marker['type'] = 'FL'
                                            ctxt_cycle['id_type'] = 3
                                            ctxt_cycle['type_label'] = 'fermeture lente'
                                if system == 'MM':
                                    if cycle.lt_cf_est_active.sum() == 0:  # opening
                                        if max(cycle['position_moteur_de_la_marche_mobile']) > 200:
                                            marker['sys'] = 'MM'
                                            marker['type'] = 'O'
                                            ctxt_cycle['id_type'] = 1
                                            ctxt_cycle['type_label'] = 'ouverture'
                                        else:
                                            marker['sys'] = 'MM'
                                            marker['type'] = 'OC'
                                            ctxt_cycle['id_type'] = 2
                                            ctxt_cycle['type_label'] = 'ouverture courte'
                                    else:  # closing
                                        if max(cycle['position_moteur_de_la_marche_mobile']) > 180:
                                            marker['sys'] = 'MM'
                                            marker['type'] = 'F'
                                            ctxt_cycle['id_type'] = 3
                                            ctxt_cycle['type_label'] = 'fermeture'
                                        else:
                                            marker['sys'] = 'MM'
                                            marker['type'] = 'FC'
                                            ctxt_cycle['id_type'] = 4
                                            ctxt_cycle['type_label'] = 'fermeture courte'

                        marker['tps'] = cycle.time.iloc[-1] - cycle.time.iloc[0]
                        ctxt_cycle['temps_cycle'] = 0.001 * (cycle.timestamp.iloc[-1] - cycle.timestamp.iloc[0])
                        cycle = cycle.reset_index(drop=True)
                        marker, ctxt, cycle = clean_cycles(marker, ctxt_cycle, cycle)
                        marker.update(ctxt)

                    # add a new marker
                    markers.append(marker)

        # turn the markers into the correct format
        cpts, y, cycles = [], np.zeros(len(X)), []
        for mk in markers:
            cpts.append(mk['mstart'])
            cpts.append(mk['mend'])
            y[mk['mstart']:mk['mend']] = mk['cycle']
            cycles.append(X.loc[mk['mstart']:mk['mend'], self.atts].values)
        # return the result
        return cpts, y, cycles


class AutoencoderBasedExtractor(Extractor):
    """
    Implement the autoencoder-based extractor (pretrained models)

    [1] W.-H. Lee, J. Ortiz, B. Ko, and R. Lee, "Time Series Segmentation through Automatic Feature Learning,"
    arXiv:1801.05394 [cs, stat], Jan. 2018, Accessed: Nov. 23, 2020. [Online].
    Available: https://arxiv.org/abs/1801.05394
    """

    def __init__(self, ae_path, enc_path, winsize, atts, delta):
        """
        Load the pretrained models

        Parameters
        ----------
        ae_path
            Path to the autoencoder
        enc_path
            Path to the encoder
        winsize
            Window size
        """
        # load the full autoencoder
        self.autoencoder = load_model(ae_path)

        # load the encoder
        self.encoder = load_model(enc_path)
        self.encoder.compile(optimizer='adam', loss='mse')

        # hyperparameters
        self.winsize = winsize
        self.atts = atts
        self.delta = delta
        # deduce the maximum length from the input shape of the encoder
        self.maxlen = int(self.encoder.layers[0].input_shape[0][1] / len(self.atts))

    def predict(self, X: pd.DataFrame, X_id):
        """
        Extract cycles

        Parameters
        ----------
        X
        X_id

        Returns
        -------

        """
        # transform the data to a format comptatible with tensorflow
        x, windows, data = self._preprocess(X.copy())
        if x is None and windows is None:
            return [], np.zeros(len(X)), []

        # row = number of windows, col = dimension
        x = x.T

        # predict cycles
        # /!\ Calling predict in a loop causes memory leak!!!
        # encoded_ft = [self.encoder.predict(np.atleast_2d(inst), verbose=0) for inst in x]
        ts_x = tf.convert_to_tensor(x, dtype=float)
        encoded_ft = np.array(self.encoder.predict_on_batch(ts_x))
        del ts_x
        keras.backend.clear_session()

        distances = [0] + [_distance(encoded_ft[i - 1], encoded_ft[i]) for i in range(1, x.shape[0])] + [0]
        cpts = [data[data['time'] == windows[pk]['time'].iloc[0]].index[0] for pk in find_peaks(distances)[0]]

        # do some postprocessing: get the array label and list of cycle data
        N = len(X)
        y = np.zeros(N)
        cyc_data = []
        std_data = np.std(X[self.atts].values)
        num_cyc = 1
        for i in range(len(cpts) - 1):
            segm = X[self.atts].values[cpts[i]:cpts[i + 1]]
            if np.count_nonzero(segm) != 0 and np.std(segm) / std_data >= self.delta:
                y[cpts[i]:cpts[i + 1]] = num_cyc
                cyc_data.append(segm)
                num_cyc += 1

        # return the list of change points, the ID array, and the actual cycle data
        return cpts, y, cyc_data

    def _preprocess(self, X: pd.DataFrame):
        """
        Transform the input data so that it's compatible to the autoencoder
        Parameters
        ----------
        X

        Returns
        -------

        """
        # the name of the time column
        time_col = 'time'

        # X[time_col] = X[time_col] * 3600 * 1000  # convert to milisecond  # TODO uncomment if NAT data
        X[time_col] = X[time_col] * 1000   # TODO uncomment if R2N data

        X[time_col] = X[time_col].astype(int)

        # cut into windows & add delta time in for each window
        windows_, start = [], 0
        for i_row, row in X.iterrows():
            time_diff = row[time_col] - X.loc[start, time_col]
            if time_diff > self.winsize or (i_row == len(X) - 1 and start != i_row):
                w_data = X.loc[start:i_row]  # get the data in this window, keep all attributes
                start = i_row
                windows_.append(w_data)

        # if there are windows with less than 10 points, just ignore them
        windows = [w for i, w in enumerate(windows_) if len(w) > 1]

        # pad the windows (only keep the 3 main attributes)
        wins = [w[self.atts].values for w in windows]
        padded = pad_sequences(wins, dim=len(self.atts), maxlen=self.maxlen)

        # make column-stacked vector for each window
        X_ = np.hstack(
            [np.vstack([np.atleast_2d(win[j]).T for j in range(win.shape[0])]) for win in padded])

        # return the preprocessed data, the windows, the original data
        return X_, windows, X


class Ensemble:

    def __init__(self, config):
        """
        Initialize the ensemble as a dictionary, key = extractor name, value = the extractpr
        """
        self.extractors = {
            'activity': ActivityBasedExtractor(delta=config['ensemble']['activity']['delta'], atts=config['atts']),
            'autoencoder': AutoencoderBasedExtractor(
                ae_path=os.path.join(config['wd'], config['ensemble']['autoencoder']['ae_path']),
                enc_path=os.path.join(config['wd'], config['ensemble']['autoencoder']['enc_path']),
                winsize=config['ensemble']['autoencoder']['winsize'],
                atts=config['atts'],
                delta=config['ensemble']['autoencoder']['delta']
            ),
            'expert':
                NATExpertSystemExtractor(marker_config=config['ensemble']['expert'], atts=config['atts'])
                if config['dataset'] == 'nat'
                else R2NExpertSystemExtractor(marker_config=config['ensemble']['expert'], atts=config['atts'])
        }

        # number of extractors
        self.J = len(self.extractors.keys())

    def predict(self, X: pd.DataFrame, X_id) -> dict[str, CandidateByExtractor]:
        """
        Get the individual extraction from each extractor

        Parameters
        ----------
        X
        X_id

        Returns
        -------

        """
        candidates = {}
        for ext_label, ext_model in self.extractors.items():
            # get the result from each extractor
            cpts, y, cyc_data = ext_model.predict(X, X_id)
            # transform it to the correct form (CandidatePerExtractor)
            indices = get_cycles(y)
            markers = [Marker(mid=None, mstart=idx[0], mend=idx[-1]+1, no_cyc=i+1, label='', ext=ext_label,
                              data=cyc_data[i], input_len=len(y))
                       for i, idx in enumerate(indices)]
            candidates[ext_label] = CandidateByExtractor(flattened=y, markers=markers)
        return candidates


# # ---------------------------- #
# # QUICK TEST OF EACH EXTRACTOR #
# # ---------------------------- #
# if __name__ == '__main__':
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     from tools.utils import preprocess
#     import json
#
#     # read and preprocess the input
#     # fname_ = 'z50001__________zr_506_001______dcu2____________dcuconddata_____210216_190336.pkl'
#     fname_ = 'z5500503________zr_55_21503_____dcu1____________dcuconddata______210712_041346.pkl'
#     inpath = 'E:\\PhD\\Data\\clean\\2021\\test'
#     # atts_ = ['imot_p', 'umot_p', 'pmot_p']
#     atts_ = ['courant_moteur_de_la_porte', 'tension_moteur_de_la_porte', 'position_moteur_de_la_porte']
#     data_ = pd.read_pickle(inpath + '\\' + fname_)
#     data_ = preprocess(data_, 'r2n')
#
#     with open('../config/config_r2n.json', 'r') as f:
#         conf_ = json.load(f)
#
#     # extract cycles
#     # ext = ActivityBasedExtractor(delta=conf_['ensemble']['activity']['delta'], atts=conf_['atts'])
#     # ext2 = ExpertSystemExtractor(marker_config=conf_['ensemble']['expert'], atts=conf_['atts'])
#     # ext = AutoencoderBasedExtractor(ae_path=conf_['ensemble']['autoencoder']['ae_path'],
#     #                                 enc_path=conf_['ensemble']['autoencoder']['enc_path'],
#     #                                 winsize=conf_['ensemble']['autoencoder']['winsize'],
#     #                                 atts=conf_['atts'],
#     #                                 delta=conf_['ensemble']['autoencoder']['delta'])
#     # ext = R2NExpertSystemExtractor(marker_config={}, atts=atts_)
#     # cpts_, y_, cycles_ = ext.predict(data_, fname_)
#     # # plot it
#     # fig = plt.figure()
#     # plt.plot(data_[atts_])
#     # for cp in cpts_:
#     #     plt.axvline(cp, c='k', linestyle='--')
#     # plt.show()
#
#     # test the whole ensemble
#     ensemble = Ensemble(conf_)
#     res = ensemble.predict(X=data_, X_id=fname_)
#
#     fig = plt.figure(figsize=(18, 3.5))
#     i = 0
#     for ext_name in res.keys():
#         ax = fig.add_subplot(1, 3, i + 1)
#         ax.plot(data_[atts_])
#         markers = res[ext_name].markers
#         for mk in markers:
#             ax.axvline(mk.mstart, c='k', linestyle='--')
#             ax.axvline(mk.mend, c='k', linestyle='--')
#         ax.set_title(ext_name.upper(), fontsize=14)
#         i += 1
#     plt.show()
