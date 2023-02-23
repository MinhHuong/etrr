import numpy as np
import pandas as pd


def compute_gradients(data, df, system: str):
    # current_column_name = f'I{system}'
    # position_column_name = f'P{system}'
    # voltage_column_name = f'T{system}'
    current_column_name = 'courant_moteur_de_la_porte'
    position_column_name = 'position_moteur_de_la_porte'
    voltage_column_name = 'tension_moteur_de_la_porte'
    df['time'] = data.time.astype('timedelta64[ms]')
    df['timestamp'] = data.timestamp
    df['testltao1'] = np.gradient(data.lt_ao1_est_active)
    df['testltao2'] = np.gradient(data.lt_ao2_est_active)
    df['testltcf1'] = np.gradient(data.lt_cf_est_active)
    df['testltcf2'] = np.gradient(data.lt_cf_est_active)  # odd, I don't have lt_cf2?
    gradient_current = data[current_column_name].interpolate(method='nearest').diff() / data.time.astype(
        'timedelta64[s]').diff()
    gradient_position = data[position_column_name].interpolate(method='nearest').diff() / data.time.astype(
        'timedelta64[s]').diff()
    gradient_voltage = data[voltage_column_name].interpolate(method='nearest').diff() / data.time.astype(
        'timedelta64[s]').diff()

    current_df = pd.concat([data.time, data[current_column_name], gradient_current, data[current_column_name].diff()],
                           keys=['time', 'current', 'gradient', 'diff'], axis=1)
    position_df = pd.concat(
        [data.time, data[position_column_name], gradient_position, data[position_column_name].diff()],
        keys=['time', 'position', 'gradient', 'diff'], axis=1)
    voltage_df = pd.concat([data.time, data[voltage_column_name], gradient_voltage, data[voltage_column_name].diff()],
                           keys=['time', 'voltage', 'gradient', 'diff'], axis=1)

    df['var_current'] = abs(current_df['diff'])
    df['var_position'] = 10 * abs(position_df['diff'])
    df['var_voltage'] = abs(voltage_df['diff'])
    df['var'] = 5 * abs(current_df['diff']) + 10 * abs(position_df['diff']) + abs(voltage_df['diff'])
    return df
