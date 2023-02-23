"""
UTILITY FUNCTIONS
"""

# python's native libraries
from enum import IntEnum


class TIMEUNIT(IntEnum):
    """Time unit to manage the decay function (in terms of seconds)"""
    HOUR = 3600
    MINUTE = 60
    SECOND = 1


def convert_time_unit_from_string(unit_str: str):
    """
    Convert a time unit from string to enum type

    Parameters
    ----------
    unit_str: str

    Returns
    -------

    """
    if unit_str == 'hour':
        return TIMEUNIT.HOUR
    elif unit_str == 'minute':
        return TIMEUNIT.MINUTE
    else:
        return TIMEUNIT.SECOND
