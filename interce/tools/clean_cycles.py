import math
from datetime import timedelta

import pandas as pd

my_dict = {'cycle_too_short': {'1': (5, 'MARK', 'tps cycle court', 2.5),
                               '2': (3, 'MARK', 'tps cycle court', 2.5),
                               '3': (3, 'MARK', 'tps cycle court', 7)},
           'cycle_too_long': {'1': (6, 'MARK', 'tps cycle long', 5),
                              '2': (4, 'MARK', 'tps cycle long', 5),
                              '3': (4, 'MARK', 'tps cycle long', 9),
                              '4': (3, 'MARK MM', 'tps cycle long', 4),
                              '5': (5, 'MARK MM', 'tps cycle long', 5)},
           'negative_position': {'1': (7, 'MARK', 'negative position'),
                                 '2': (5, 'MARK', 'negative position'),
                                 '4': (4, 'MARK MM', 'negative position'),
                                 '5': (6, 'MARK MM', 'negative position')},
           'incomplete_cycle': {'1': (8, 'MARK', 'incomplete position'),
                                '2': (6, 'MARK', 'incomplete position'),
                                '3': (5, 'MARK', 'incomplete position')}}


def fill_context(ctxt, error_cycle, error_cycle_type, error_cycle_label):
    if ctxt['error_cycle'] == 0:
        ctxt['error_cycle'] = error_cycle
        ctxt['error_cycle_type'] = error_cycle_type
        ctxt['error_cycle_label'] = error_cycle_label
    else:
        ctxt['error_cycle'] = int(f"{ctxt['error_cycle']}{error_cycle}")
        ctxt['error_cycle_type'] = error_cycle_type
        ctxt['error_cycle_label'] = f"{ctxt['error_cycle_label']},{error_cycle_label}"
    return ctxt


def init_data(cycle, sys):
    sys_dict = {
        'P': ['courant_moteur_de_la_porte', 'tension_moteur_de_la_porte'],
        'M': ['courant_moteur_de_la_marche_mobile', 'tension_moteur_de_la_marche_mobile']}
    if cycle[sys_dict[sys]].values[0].sum() != 0:
        data_init = cycle.iloc[0, :].copy(deep=False)
        if data_init.Time > pd.Timedelta('0 days 00:00:00'):
            data_init.Time -= timedelta(milliseconds=50)
        data_init.Timestamp = data_init.Timestamp - 50
        data_init[sys_dict[sys][0]] = 0
        data_init[sys_dict[sys][1]] = 0
        cycle = pd.concat([data_init.to_frame().T, cycle])
    if cycle[sys_dict[sys]].values[-1].sum() != 0:
        data_init = cycle.iloc[-1, :].copy(deep=False)
        data_init.Time = data_init.Time + timedelta(milliseconds=50)
        data_init.Timestamp = data_init.Timestamp + 50
        data_init[sys_dict[sys][0]] = 0
        data_init[sys_dict[sys][1]] = 0
        cycle = pd.concat([cycle, data_init.to_frame().T])
    return cycle


def is_cycle_too_short(ctxt, param):
    (error_cycle, error_cycle_type, error_cycle_label, threshold) = param
    if ctxt['temps_cycle'] < threshold:
        fill_context(ctxt, error_cycle, error_cycle_type, error_cycle_label)


def is_cycle_too_long(ctxt, param):
    (error_cycle, error_cycle_type, error_cycle_label, threshold) = param
    if ctxt['temps_cycle'] > threshold:
        fill_context(ctxt, error_cycle, error_cycle_type, error_cycle_label)


def has_negative_position(cycle, ctxt, param):
    (error_cycle, error_cycle_type, error_cycle_label) = param
    pneg = cycle['position_moteur_de_la_porte'][cycle['position_moteur_de_la_porte'] < -7]
    if not pneg.empty:
        fill_context(ctxt, error_cycle, error_cycle_type, error_cycle_label)


def is_cycle_complete(cycle, ctxt, param):
    (error_cycle, error_cycle_type, error_cycle_label) = param
    pdelta = abs(cycle.PPOR.values[0] - cycle.PPOR.values[-1])
    if pdelta < 750:
        fill_context(ctxt, error_cycle, error_cycle_type, error_cycle_label)


def clean_cycles(marker, ctxt, cycle):
    if len(marker) > 0 and marker['sys'] == 'P':
        ctxt['error_cycle'] = 0
        ctxt['error_cycle_type'] = ''
        ctxt['error_cycle_label'] = 'no error'
        if (cycle.timestamp.diff() > 50).any():
            ctxt['error_cycle'] = 2
            ctxt['error_cycle_type'] = 'MARK'
            ctxt['error_cycle_label'] = 'non continuous acquisition'

        if ctxt['id_type'] == 1:
            # check position offset
            if math.isnan(cycle['position_moteur_de_la_porte'].values[0]):
                cycle['position_moteur_de_la_porte'].fillna(method='bfill', inplace=True)
            if math.isnan(cycle['courant_moteur_de_la_porte'].values[0]):
                cycle['courant_moteur_de_la_porte'].fillna(method='bfill', inplace=True)
            if math.isnan(cycle['tension_moteur_de_la_porte'].values[0]):
                cycle['tension_moteur_de_la_porte'].fillna(method='bfill', inplace=True)
            pstart = cycle['position_moteur_de_la_porte'].values[0]
            if 150 < pstart < 250:
                fill_context(ctxt, 3, 'MARK', 'grany test')
            else:
                if abs(pstart) > 7:
                    fill_context(ctxt, 4, 'MARK', 'offset position')

                cycle = init_data(cycle, marker['sys'])
                is_cycle_too_short(ctxt, my_dict['cycle_too_short']['1'])
                is_cycle_too_long(ctxt, my_dict['cycle_too_long']['2'])
                has_negative_position(cycle, ctxt, my_dict['negative_position']['1'])
                is_cycle_complete(cycle, ctxt, my_dict['incomplete_cycle']['1'])
        elif ctxt['id_type'] == 2:
            cycle = init_data(cycle, marker['sys'])
            is_cycle_too_short(ctxt, my_dict['cycle_too_short']['2'])
            is_cycle_too_long(ctxt, my_dict['cycle_too_long']['2'])
            has_negative_position(cycle, ctxt, my_dict['negative_position']['2'])
            is_cycle_complete(cycle, ctxt, my_dict['incomplete_cycle']['2'])
        elif ctxt['id_type'] == 3:
            cycle = init_data(cycle, marker['sys'])
            is_cycle_too_short(ctxt, my_dict['cycle_too_short']['3'])
            is_cycle_too_long(ctxt, my_dict['cycle_too_long']['3'])
            is_cycle_complete(cycle, ctxt, my_dict['incomplete_cycle']['3'])
        else:
            ctxt['error_cycle'] = 9
            ctxt['error_cycle_type'] = 'MARK'
            ctxt['error_cycle_label'] = 'cycle identification error'
    if len(marker) > 0 and marker['sys'] == 'M':
        if len(marker) > 0 and marker['sys'] == 'P':
            ctxt['error_cycle'] = 0
            ctxt['error_cycle_type'] = ''
            ctxt['error_cycle_label'] = 'no error'
            if (cycle.timestamp.diff() > 50).any():
                fill_context(ctxt, 2, 'MARK MM', 'non continuous acquisition')

            cycle = init_data(cycle, marker['sys'])

            if ctxt['id_type'] in {1, 2}:
                if math.isnan(cycle['position_moteur_de_la_marche_mobile'].values[0]):
                    cycle['position_moteur_de_la_marche_mobile'].fillna(method='bfill', inplace=True)
                if math.isnan(cycle['courant_moteur_de_la_marche_mobile'].values[0]):
                    cycle['courant_moteur_de_la_marche_mobile'].fillna(method='bfill', inplace=True)
                if math.isnan(cycle['tension_moteur_de_la_marche_mobile'].values[0]):
                    cycle['tension_moteur_de_la_marche_mobile'].fillna(method='bfill', inplace=True)

                is_cycle_too_long(ctxt, my_dict['cycle_too_long']['4'])
                has_negative_position(cycle, ctxt, my_dict['negative_position']['4'])
            else:
                is_cycle_too_long(ctxt, my_dict['cycle_too_long']['5'])
                has_negative_position(cycle, ctxt, my_dict['negative_position']['5'])

    return marker, ctxt, cycle.reset_index(drop=True)
