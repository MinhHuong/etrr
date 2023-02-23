"""
THE EXPERT SYSTEM THAT EXTRACTS CYCLES FROM NAT DATA
"""

import json
import os.path
from abc import ABCMeta
from dataclasses import field, dataclass
from enum import Enum
from typing import Dict, List, Tuple
import logging
from uuid import uuid1

# data manipulation
import pandas as pd
import numpy as np


# Define aliases for signal names
_LTAO = 'LTAO'
_DOOR = 'DOOR'
_STEP = 'STEP'
_OPEN = 'OPEN'
_CLOSE = 'CLOSE'
_UFR = 'UFR'
_PMR = 'PMR'


class DCUFlags(Enum):

    # Signal values mapping of last DCU column ("ID")
    LTAO = 16  # LTAO signal value
    DOOR_OPEN = 32  # Door opening (if between LTAO edges)
    DOOR_CLOSE = 33  # Door closing (if between LTCF edges)
    PMR_OPEN = 34  # "PMR march" opening (if between LTAO edges)
    PMR_CLOSE = 35  # "PMR march" closing (if between LTCF edges)
    UFR_OPEN = 36  # "UFR march" opening (if between LTAO edges)
    UFR_CLOSE = 37  # "UFR march" closing (if between LTCF edges)


# Utility dictionary for accessing signal values ("ID" of last DCU column) which are later used in code
DCU_ID_MAPPING = {

    # LTAO signal value
    _LTAO: DCUFlags.LTAO.value,

    # Use cases
    _DOOR: {
        # Door opening (if between LTAO edges)
        _OPEN: DCUFlags.DOOR_OPEN.value,
        # Door closing (if between LTCF edges)
        _CLOSE: DCUFlags.DOOR_CLOSE.value
    },
    _PMR: {
        # "PMR march" opening (if between LTAO edges)
        _OPEN: DCUFlags.PMR_OPEN.value,
        # "PMR march" closing (if between LTCF edges)
        _CLOSE: DCUFlags.PMR_CLOSE.value
    },
    _UFR: {
        # "UFR march" opening (if between LTAO edges)
        _OPEN: DCUFlags.UFR_OPEN.value,
        # "UFR march" opening, alternative signal (FIXME Data-scientists agree that this seems wrong)
        'OPEN_W37': 37,
        # "UFR march" closing (if between LTCF edges)
        _CLOSE: DCUFlags.UFR_CLOSE.value
    }
}


@dataclass
class Mark:
    start: int = 0
    end: int = 0

    def set_mark(self, candidate_idxs, default_start, force_start=None):
        """
        Set Marker "start" and "end"
        - "start" at first index found between edges (or default_start <=> "end" of previous use case most of the time)
        - "end" at last index found between edges (or at "start" index by default)
        """
        self.start = candidate_idxs[0] if candidate_idxs.size != 0 else default_start
        self.end = candidate_idxs[-1] if candidate_idxs.size != 0 else self.start
        if force_start is not None:
            self.start = force_start


@dataclass
class OpenCloseMark:
    """
    """
    open: Mark = field(default_factory=Mark)
    close: Mark = field(default_factory=Mark)

    @property
    def op_ini(self):
        return self.open.start

    @property
    def op_end(self):
        return self.open.end

    @property
    def cl_ini(self):
        return self.close.start

    @property
    def cl_end(self):
        return self.close.end

    def get_open(self):
        return [self.op_ini, self.op_end]

    def get_close(self):
        return [self.cl_ini, self.cl_end]

    def get_open_marks(self):
        return self.op_ini, self.op_end

    def get_close_marks(self):
        return self.cl_ini, self.cl_end

    @classmethod
    def generate_from_df(cls, markers_df):
        if markers_df.shape != (1, 8) and markers_df.shape != (1, 9):
            # A 9th field is allowed to be present (base dat name) but is not used
            raise ValueError('Marker interfaces have been modified')

        markers = markers_df.loc[0]

        door_marker = cls()
        door_marker.open.start = markers.door_open_ini
        door_marker.open.end = markers.door_open_end
        door_marker.close.start = markers.door_close_ini
        door_marker.close.end = markers.door_close_end

        step_marker = cls()
        step_marker.open.start = markers.step_open_ini
        step_marker.open.end = markers.step_open_end
        step_marker.close.start = markers.step_close_ini
        step_marker.close.end = markers.step_close_end

        return door_marker, step_marker


class MultipleRisingError(Exception):
    def __init__(self, first_edge):
        self.first_edge = first_edge


class MultipleFallingError(Exception):
    def __init__(self, first_edge):
        self.first_edge = first_edge


class NoRisingError(Exception):
    pass


class NoFallingError(Exception):
    pass


class NoLtaoError(Exception):
    pass


class DoorProcessor(metaclass=ABCMeta):

    def __init__(self, config):
        super().__init__()

        # Tgz data
        self.tgz_cycles_context = pd.DataFrame()  # Cycle context data
        self.tgz_file_context = pd.DataFrame()  # File context data

        # Current cycle file to process
        self.dat_file_in_process = None

        # some configs
        self.relevant_variables = None
        self.config = config

    def get_row(self):
        indexes = self.tgz_cycles_context.index[self.tgz_cycles_context['file_name'] == self.dat_file_in_process]\
            .tolist()
        # Numpy equivalent: np.where(self.tgz_cycles_context['file_name'] == dat_file_in_process)
        nb_found = len(indexes)
        # Check uniqueness (could maybe be removed in prod)
        if nb_found == 0:
            raise ValueError(f'No context found with this name {self.dat_file_in_process}')
        elif nb_found > 1:
            raise ValueError(f'Multiple existing contexts, parquet file is corrupted for {self.dat_file_in_process}')
        # Update the corresponding dat context in tgz context table
        else:  # nb_found == 1:
            index = indexes[0]
        return index

    def get_dat_cycle_context(self, key):
        """
        Getter for the dat/cycle context value
        """
        index = self.get_row()
        return self.tgz_cycles_context.loc[index, key]

    def copy_dat_cycle_context(self, cycle_context: pd.DataFrame) -> None:
        self.tgz_cycles_context = cycle_context.copy()

    def update_dat_cycle_context(self, new_partial_context: Dict) -> None:
        # For each context value
        for key in new_partial_context.keys():
            # Get dat context index...
            index = self.get_row()
            # ...update
            # self.tgz_cycles_context.iloc[index] = partial_tgz_cycles_context.iloc[0]
            self.tgz_cycles_context.loc[index, key] = new_partial_context[key]
            self.tgz_cycles_context.loc[index, 'last_update_by'] = type(self).__base__.__name__.lower()
            # FIXME See with data-scientists to remove this flag, which has no added-value inside the processing
            if self.tgz_cycles_context.loc[index, 'anomaly_id'] > 0:
                tmp = self.tgz_cycles_context.loc[index, 'nb_dcu_active_codes']
                abnormal = 2 if self.tgz_cycles_context.loc[index, 'nb_dcu_active_codes'] == 0 else 1
                if self.tgz_cycles_context.loc[index, 'abnormal'] != abnormal:
                    self.tgz_cycles_context.loc[index, 'abnormal'] = abnormal

    def _init_dat_cycle_context(self, fname):
        # set the context
        cycle_context_id = str(uuid1())
        context_file_id = self.config.get('tgz_file_context_id', '')  # FIXME this will be empty all the time...
        init_cycle_context = {
            'file_name': fname,  # str: Decoder
            'last_update_by': '',  # str: BaseMicroService
            'anomaly_id': 0,  # UInt32: Decoder, Sampling...
            'vehicle': None,  # UInt16: Decoder
            'door_side': '',  # str: Decoder
            'id': cycle_context_id,  # str: Decoder
            'context_file_id': context_file_id,  # str: Decoder
            'mission_code': '',  # str: Decoder
            'station_id': None,  # UInt32: Decoder
            'cycle_date': None,  # datetime64[ns, UTC]: Decoder
            'end_date': None,  # datetime64[ns, UTC]: Decoder
            'step_type': '',  # str: Marker
            'door_has_moved': None,  # boolean: Sampling
            'step_has_moved': None,  # boolean: Sampling
            'is_valid_door': None,  # boolean: Decoder
            'is_valid_pmr': None,  # boolean: Decoder
            'is_valid_ufr': None,  # boolean: Decoder
            'door_instance': None,  # UInt8: Decoder
            'soft_release': None,  # UInt8: Decoder
            'soft_update': None,  # UInt8: Decoder
            'soft_version': None,  # UInt8: Decoder
            'offset_open': None,  # Int32: Sampling
            'offset_close': None,  # Int32: Sampling
            'outside_temp': None,  # float32: Decoder
            'abnormal': 0,  # UInt16: Decoder, Sampling...
            'nb_dcu_active_codes': None,  # UInt8: Decoder, Marker...
            'dcu_default_codes': '',  # str: Decoder
            'grany_test': False,  # boolean
            'z_ufr': None  # boolean: Decoder
        }

        # Set types once for all
        with open(os.path.join(self.config['wd'], self.config['context_var_type']), 'r') as f:
            var_mapping = json.load(f)['portes']
            dat_cycle_context_types = {var['name']: var['type'] for var in var_mapping}

        # Convert to dataframe
        init_cycle_context_df = pd.DataFrame({
            key: [val]
            for key, val in init_cycle_context.items()
        }).astype(dat_cycle_context_types, copy=False)

        # Create a tgz cycles context container if it is the first time we add/update a dat cycle context
        if self.tgz_cycles_context.empty:
            self.tgz_cycles_context = init_cycle_context_df
        # Otherwise, append dat cycle to tgz cycles dataframe
        else:
            prev_context_ind = np.where(self.tgz_cycles_context['file_name'] == fname)
            if prev_context_ind[0].size == 0:
                self.tgz_cycles_context = self.tgz_cycles_context.append(init_cycle_context_df).reset_index(drop=True)
            else:
                raise ValueError(f'Already existant contexts at initialization for {fname}')

    def set_relevant_variables(self, use_case: str):
        """
        Set the relevant variables depending on the use case: "decoder", "marker", "sampler"

        Parameters
        ----------
        use_case

        Returns
        -------

        """
        self.relevant_variables = self.config['relevant_variables'][use_case]


class NATMarker(DoorProcessor):

    def __init__(self, config) -> None:
        # Decoded DCU data (input)
        super().__init__(config)
        self.dcu_data = None
        # Type of step if applicable
        self.step_type = None
        # Marker data (the actual markers)
        self.door_mark = OpenCloseMark()
        self.step_mark = OpenCloseMark()

        self.fdcv_step = False
        self.fdcv_door = False

        # Number of noisy spikes in analog signals associated to opening authorisation and closing commands
        self.ltao_spikes = 0
        self.ltcf_spikes = 0
        # Falling edge and rising edge indexes of these signals
        self.ltcf_rise = 0  # This one is needed to default to zero in case of exception (=> used in contexts)
        self.ltcf_fall = None
        self.ltao_rise = None
        self.ltao_fall = None

        # get threshold info from file JSON
        with open(os.path.join(self.config['wd'], self.config['thresholds']), 'r') as f:
            obj = json.load(f)
            self.thresholds = obj['marker']['default_marker']

        # set the relevant variables
        self.set_relevant_variables('marker')

        # currently processed file
        self.dat_file_in_process = None

    def reset(self, base_dat_name):
        self.dat_file_in_process = base_dat_name
        self.dcu_data = None
        self.step_type = None
        self.door_mark = OpenCloseMark()
        self.step_mark = OpenCloseMark()
        self.fdcv_step = False
        self.fdcv_door = False
        self.ltao_spikes = 0
        self.ltcf_spikes = 0
        self.ltcf_rise = 0
        self.ltcf_fall = None
        self.ltao_rise = None
        self.ltao_fall = None

    def _set_zufr(self):
        z_ufr = True if self.dcu_data['zufr'].mean() > 0.5 else False
        self.update_dat_cycle_context({
            'z_ufr': z_ufr,
        })

    def _set_step_type(self):
        df = self.dcu_data
        # Pycharm warns about bool.sum() but it's a non issue here
        pmr_openings = (df.loc[:, 'id'] == DCU_ID_MAPPING[_PMR][_OPEN]).sum()
        ufr_openings = (df.loc[:, 'id'] == DCU_ID_MAPPING[_UFR][_OPEN]).sum()
        self.step_type = _UFR if ufr_openings > pmr_openings else _PMR

    def remove_spikes(self, ltao_sig, ltcf_sig):
        """
        This function corresponds to "filtre aocf" in MatLab code
        """
        ltao_clean = ltao_sig.copy()
        ltcf_clean = ltcf_sig.copy()
        ltao_spikes = 0
        ltcf_spikes = 0

        def is_a_spike(j, sig, ref):
            return sig[j - 1] != sig[j] != sig[j + 1] and ref[j - 1] == ref[j] == ref[j + 1]

        # NOTA: this loop in MatLab version stops at signal.size-2, this corrected version has been orally accepted
        # TODO Vectorise for performances
        for i in range(1, ltao_clean.size - 1):
            # If values differs from neighbours (noise spike)
            if is_a_spike(i, ltao_sig, ltcf_sig):
                ltao_clean[i] = ltao_sig[i + 1]
                ltao_spikes = ltao_spikes + 1
            if is_a_spike(i, ltcf_sig, ltao_sig):
                ltcf_clean[i] = ltcf_sig[i + 1]
                ltcf_spikes = ltcf_spikes + 1

        self.ltao_spikes = ltao_spikes
        self.ltcf_spikes = ltcf_spikes
        return ltao_clean, ltcf_clean

    @staticmethod
    def get_rise_idx(signal):
        rising_edges = np.argwhere(signal == 1).flatten()
        if rising_edges.size == 0:
            logging.debug(f'No rising edges found')
            raise NoRisingError
        elif rising_edges.size > 1:
            logging.debug(f'{rising_edges.size} rising edges found')
            raise MultipleRisingError(rising_edges[0])
        else:
            return rising_edges[0]

    @staticmethod
    def get_fall_idx(signal):
        falling_edges = np.argwhere(signal == -1).flatten()
        if falling_edges.size == 0:
            logging.debug(f'No falling edges found')
            raise NoFallingError
        elif falling_edges.size > 1:
            logging.debug(f'{falling_edges.size} falling edges found')
            raise MultipleFallingError(falling_edges[0])
        else:
            return falling_edges[0]

    def set_falls_and_rises_aocf(self):
        # Get signals to process
        ltao_signal = self.dcu_data['ltao']
        ltcf_signal = self.dcu_data['ltcf']
        dcu_ids = self.dcu_data['id']
        data_len = self.dcu_data.shape[0]

        # Find points corresponding to a door opening authorisation (LTAO)
        ltao_rises = np.argwhere(dcu_ids.values == DCU_ID_MAPPING[_LTAO]).flatten()
        if ltao_rises.size == 0:
            logging.warning('No opening authorisation found')
            raise NoLtaoError
        else:
            ltao_rise = ltao_rises[0]

        # Remove noise from LTAO and LTCF signals...
        ltao_clean, ltcf_clean = self.remove_spikes(ltao_signal.to_numpy(), ltcf_signal.to_numpy())
        # ...compute derivatives...
        ltao_diff = np.diff(ltao_clean[ltao_rise::])
        ltcf_diff = np.diff(ltcf_clean[ltao_rise::])

        # ... set default values and get falling and rising (unique) edges...
        ltcf_rise = self.ltcf_rise
        ltao_fall = self.ltao_fall
        ltcf_fall = self.ltcf_fall

        # .../!\ starting with "LTCF rise" because it is needed upstream in case of exception (used in context)
        try:
            ltcf_rise = self.get_rise_idx(ltcf_diff)
        except NoRisingError:
            ltcf_rise = data_len - 1
        except MultipleRisingError as e:
            ltcf_rise = e.first_edge
            raise e
        finally:
            self.ltcf_rise = ltcf_rise

        # LTCF fall
        try:
            ltcf_fall = self.get_fall_idx(ltcf_diff)
        except NoFallingError:
            ltcf_fall = data_len - 1
        except MultipleFallingError as e:
            ltcf_fall = e.first_edge
            raise e
        else:
            # /!\ If falling edge of door closing is too early, it must be set to the last index
            # This section should be improved with help of data-scientists because happens 99% of the time...
            ltcf_fall_min = self.thresholds.get('ltcf_fall_min_index')
            if ltcf_fall < ltcf_fall_min:
                last_idx = data_len - 1
                logging.debug(f'LTCF fall = {ltcf_fall} (< {ltcf_fall_min}), using default ({last_idx} = last index)')
                ltcf_fall = last_idx
            else:
                logging.info(f'LTCF fall = {ltcf_fall} (>= {ltcf_fall_min})')
        finally:
            self.ltcf_fall = ltcf_fall

        # LTAO rise : this one is not computed using ltao_diff, but directly from DCU ids (above)
        try:
            _ = self.get_rise_idx(ltao_diff)
        except NoRisingError:  # Only let MultipleRisingError go through
            pass
        finally:
            self.ltao_rise = ltao_rise

        # LTAO fall
        try:
            ltao_fall = self.get_fall_idx(ltao_diff)
        except NoFallingError:
            ltao_fall = data_len - 1
        except MultipleFallingError as e:
            ltao_fall = e.first_edge
            raise e
        finally:
            self.ltao_fall = ltao_fall

    def compute_edges_w_error_handling(self):
        # Check for empty data files
        dcu_data = self.dcu_data
        total_pos = sum(dcu_data['pmot'])
        min_motor_steps = self.thresholds.get('min_motor_steps')
        if total_pos < min_motor_steps:
            self.update_dat_cycle_context({
                'anomaly_id': 2
            })
            if total_pos == 0:
                logging.warning('Anomaly no.2: no motor steps in file, cannot compute markers')
            else:
                logging.warning(
                    f'Anomaly no.2: only {total_pos} (< {min_motor_steps}) motor steps in file, cannot compute markers')
            return False

        # Get rising and falling edges for (LT)AO and (LT)CF signals
        try:
            self.set_falls_and_rises_aocf()
        except NoLtaoError:
            self.update_dat_cycle_context({
                'anomaly_id': 3
            })
            logging.warning(f'Anomaly n°3: no signal edges, cannot compute markers')
            return False
        except (MultipleRisingError, MultipleFallingError):
            # /!\ FDCV flags are true in this exception case
            self.fdcv_step = True
            self.fdcv_door = True
            self.update_dat_cycle_context({
                'anomaly_id': 4
            })
            logging.warning(f'Anomaly n°4: multiples LTAO or LTCF, cannot compute markers')
            return False
        else:
            self.fdcv_step = True
            self.fdcv_door = True
            return True

    def get_dcu_id_from_use_case(self, system, open_close_flag):
        """
        Util function: provides mapping between use cases and associated DCU 'ID' value
        system = 'DOOR' | 'STEP'
        open_close_flag = 'OPEN' | 'CLOSE'
        """
        if system == _STEP:
            system_alias = self.step_type
        else:
            system_alias = system

        if open_close_dict := DCU_ID_MAPPING.get(system_alias):
            if dcu_value := open_close_dict.get(open_close_flag):
                return dcu_value
        raise KeyError(f'Unhandled system {system!r} for use case {open_close_flag!r}')

    def set_markers(self, system, open_close_flag):

        # Get DCU value corresponding to the use case ("uc") to mark
        uc_id = self.get_dcu_id_from_use_case(system, open_close_flag)

        # Get rise/fall edges from LTAO or LTCF depending on open/close flag
        if open_close_flag == _OPEN:
            rise_idx, fall_idx = self.ltao_rise, self.ltao_fall
        elif open_close_flag == _CLOSE:
            rise_idx, fall_idx = self.ltcf_rise, self.ltcf_fall
        else:
            raise ValueError(f'[Marker] Unknown flag value {open_close_flag}')

        # Find matching DCU values between edges (use case indexes)
        uc_idxs = rise_idx + np.argwhere(self.dcu_data.loc[rise_idx: fall_idx + 1, 'id'].values == uc_id).flatten()

        # Perform actual marking depending on use case
        if system == _STEP:
            step_type = self.step_type
            if open_close_flag == _OPEN:
                self.step_mark.open.set_mark(uc_idxs, 0, force_start=0)
                # (FIXME Data-scientists agree that this seems wrong)
                if step_type == _UFR:
                    idxs_w_37 = np.argwhere(
                        self.dcu_data.loc[rise_idx: fall_idx + 1, 'id'].values == DCU_ID_MAPPING[_UFR]['OPEN_W37']
                    ).flatten()
                    max_w_37 = max(idxs_w_37) if idxs_w_37.size > 0 else 0
                    ufr_open_end_w37 = rise_idx + max_w_37
                    self.step_mark.open.end = max(self.step_mark.open.end, ufr_open_end_w37)
            elif open_close_flag == _CLOSE:
                self.step_mark.close.set_mark(uc_idxs, self.door_mark.cl_end)
            else:
                raise ValueError(f'[Marker] Unknown flag value {open_close_flag}')
        elif system == _DOOR:
            if open_close_flag == _OPEN:
                self.door_mark.open.set_mark(uc_idxs, self.step_mark.op_end)
            elif open_close_flag == _CLOSE:
                self.door_mark.close.set_mark(uc_idxs, self.door_mark.op_end)
            else:
                raise ValueError(f'[Marker] Unknown flag value {open_close_flag}')
        else:
            raise ValueError(f'[Marker] Unknown system value {system}')

    def set_fdcv_flags(self):

        """
        Closing flags
        """
        self.fdcv_step = True
        self.fdcv_door = True
        dcu_data = self.dcu_data

        # Aliasing for readability
        stp_ini = self.step_mark.close.start
        stp_end = self.step_mark.close.end
        doo_ini = self.door_mark.close.start
        doo_end = self.door_mark.close.end

        # Check for activations of FDCV during step closing
        if self.step_type == _UFR:
            if dcu_data.loc[stp_ini: stp_end + 1, 'fcvufr'].any():
                self.fdcv_step = False
        elif self.step_type == _PMR:
            if dcu_data.loc[stp_ini: stp_end + 1, 'fcfpmr'].any():
                self.fdcv_step = False
        # Check for activations of FDCV during door closing ("Fin de course condamnation")
        if dcu_data.loc[doo_ini: doo_end + 1, 'fdccondpor'].any():
            self.fdcv_door = False

    def update_anomaly_context(self):
        """
        Checks that marks inner data "start" and "end" are coherent ("end index" > "start index")
        Equivalent to MatLab script "verification_marqueur.m"
        """

        active_codes = self.get_dat_cycle_context('nb_dcu_active_codes') > 0
        ltcf_idx = self.ltcf_rise
        fdcv_step = not self.fdcv_step
        fdcv_door = not self.fdcv_door

        time_data = np.cumsum(self.dcu_data['time_step'].values)

        ltao_time = 0
        ltcf_time = 0.02 * time_data[ltcf_idx] if ltcf_idx != 0 else 0  # Conversion from 20ms increments to seconds

        step_open_ok = self.step_mark.op_end > self.step_mark.op_ini
        door_open_ok = self.door_mark.op_end > self.door_mark.op_ini
        door_close_ok = self.door_mark.cl_end > self.door_mark.cl_ini
        step_close_ok = self.step_mark.cl_end > self.step_mark.cl_ini

        anomaly = self.get_dat_cycle_context('anomaly_id')
        # "Step Open" without "Step Close" ...or all markers incoherent
        if step_open_ok and not step_close_ok or not (step_open_ok or door_open_ok or door_close_ok or step_close_ok):
            anomaly = 5
            logging.warning('Anomaly no.5: step opening without closing')
        # "Door Open" without "Door Close"
        if door_open_ok and not door_close_ok:
            logging.warning('Anomaly no.6: door opening without closing')
            anomaly = 6
        # Step closed but no lock (FDCV)
        if step_close_ok != fdcv_step:
            logging.warning('Anomaly no.7: step closed but no lock (FDCV)')
            anomaly = 7
        # Door closed but no lock (FDCV)
        if door_close_ok != fdcv_door:
            logging.warning('Anomaly no.8: door closed but no lock (FDCV)')
            anomaly = 8
        # Check than cycle is not too long
        max_duration = self.thresholds.get('max_cycle_duration_seconds')
        if (ltcf_time - ltao_time) < max_duration:
            logging.warning(f'Anomaly no.9: cycle is less than {max_duration:.1f} sec long')
            anomaly = 9
        # If cycle has an active default code
        if active_codes:
            logging.warning('Anomaly no.10: cycle has an active default code')
            anomaly = 10

        self.update_dat_cycle_context({
            'anomaly_id': anomaly}
        )

    def get_markers_as_list(self):
        """
        Output array is equivalent to MatLab array "Marqueur" of shape (1,8)
        """
        return self.step_mark.get_open() + self.door_mark.get_open() + \
            self.door_mark.get_close() + self.step_mark.get_close()

    def get_markers_as_df(self) -> pd.DataFrame:
        """
        Output is a dataframe with header

        Returns
        -------
        pd.DataFrame
            1 row, 8 columns
        """
        cols = ['step_open_ini', 'step_open_end', 'door_open_ini', 'door_open_end',
                'door_close_ini', 'door_close_end', 'step_close_ini', 'step_close_end']
        df_payload = pd.DataFrame(np.array([self.get_markers_as_list()]), columns=cols)
        return df_payload

    def perform_cycle_marking(self):
        # Set step type (UFR of PMR) and UFR flag
        self._set_zufr()
        self._set_step_type()
        self.update_dat_cycle_context({
            'step_type': self.step_type,
        })
        logging.debug(f'March type found: {self.step_type}')

        # Compute signal edges, then markers if edges are found
        continue_to_markers = self.compute_edges_w_error_handling()
        if continue_to_markers:
            # Set markers (start/end of each system action)
            self.set_markers(_STEP, _OPEN)
            self.set_markers(_DOOR, _OPEN)
            self.set_markers(_DOOR, _CLOSE)
            self.set_markers(_STEP, _CLOSE)
            # Set FDCV flags
            self.set_fdcv_flags()
            logging.debug(f'Markers successfully computed')

        # If no anomaly yet, add more anomaly handling...
        if self.get_dat_cycle_context('anomaly_id') == 0:
            self.update_anomaly_context()

        # If an anomaly has finally occurred, update these booleans (used in indicators)
        if self.get_dat_cycle_context('anomaly_id') != 0:
            # TODO See with data-scientists, this should rather be NULL instead of False
            self.update_dat_cycle_context({
                'step_has_moved': False,
                'door_has_moved': False
            })

    def extract_markers(self, dcu_data: pd.DataFrame, dat_file_path: str) -> pd.DataFrame:
        file_name = os.path.basename(dat_file_path)
        self.reset(file_name)

        # FIXED got rid of the context checking
        self.dcu_data = dcu_data.reset_index(drop=True)
        self.perform_cycle_marking()
        return self.get_markers_as_df()

        # if self.get_dat_cycle_context('anomaly_id') != 0:
        #     logging.warning(f"Anomaly no.{self.get_dat_cycle_context('anomaly_id')}, skipping markers")
        #     return pd.DataFrame()
        # else:
        #     self.dcu_data = dcu_data.reset_index(drop=True)
        #     self.perform_cycle_marking()
        #     # return self.get_markers_as_list()
        #     return self.get_markers_as_df()

    @staticmethod
    def get_cycles(dcu_data, context, marker) -> List[Tuple]:
        """
        Extract the cycles from a decoded dat file

        Parameters
        ----------
        dcu_data
        context
        marker

        Returns
        -------
        List[Tuple]
            list of all cycles, as a tuple of (cycle data, cycle type)
        """
        result = []

        # ouverture marche
        if marker[0] != marker[1]:
            cyc_data = dcu_data[marker[0]:marker[1] + 1].reset_index(drop=True)
            cyc_type = 'OM_' + context['step_type'][0]
            result += [(cyc_data, cyc_type)]

        # ouverture porte
        if marker[2] != marker[3]:
            cyc_data = dcu_data[marker[2]:marker[3] + 1].reset_index(drop=True)
            cyc_type = 'OP'
            result += [(cyc_data, cyc_type)]

        # fermeture porte (can be FP or FLP, depending on the duration)
        if marker[4] != marker[5]:
            cyc_data = dcu_data[marker[4]:marker[5] + 1].reset_index(drop=True)
            duration = (cyc_data['time'].iloc[-1] - cyc_data['time'].iloc[0]) * 3600
            cyc_type = 'FP' if duration < 5.0 else 'FLP'
            result += [(cyc_data, cyc_type)]

        # fermeture marche
        if marker[6] != marker[7]:
            cyc_data = dcu_data[marker[6]:marker[7] + 1].reset_index(drop=True)
            cyc_type = 'FM_' + context['step_type'][0]
            result += [(cyc_data, cyc_type)]

        return result

    @staticmethod
    def get_cycles_from_sampled_data(sampled_data: pd.DataFrame, context: Dict) -> List[Tuple]:
        """

        Parameters
        ----------
        sampled_data
        context

        Returns
        -------
        List[Tuple]
            each tuple is [0] cycle data (as dataframe) and [1] cycle type (str)
        """
        # get the type of the step
        step_type = context['step_type'][0]

        # get the cycles
        step_open = sampled_data.loc[sampled_data['sys_action'] == 'step_open']
        door_open = sampled_data.loc[sampled_data['sys_action'] == 'door_open']
        door_close = sampled_data.loc[sampled_data['sys_action'] == 'door_close']
        step_close = sampled_data.loc[sampled_data['sys_action'] == 'step_close']

        return [(step_open, 'OM_' + step_type), (door_open, 'OP'), (door_close, 'FP'), (step_close, 'FM_' + step_type)]
