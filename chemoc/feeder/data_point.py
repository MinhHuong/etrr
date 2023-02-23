"""
STRUCTURE A DATA POINT WITH THE NECESSARY INFORMATION
"""


class Instance:
    """
    Data point, also called data instance or data example, composed of four attributes:
    - the content (the actual data)
    - the ID of the instance
    - the ID of the system that created this instance
    - the creation timestamp of this instance
    """

    def __init__(self, data, data_id, sys_id, tsp, ctxt):
        """

        Parameters
        ----------
        data
        data_id
        sys_id
        tsp
        """
        self.data = data
        self.id = data_id
        self.sys_id = sys_id
        self.tsp = tsp
        self.dim = len(data)
        self.ctxt = ctxt

    def to_document(self):
        return {
            '_id': self.id,
            'sys_id': self.sys_id,
            'timestamp': self.tsp,
            'data': self.data.tolist(),
            'context': self.ctxt
        }

    def copy(self):
        """
        Return an exact copy of this data point

        Returns
        -------

        """
        return Instance(data=self.data.copy(), data_id=self.id, sys_id=self.sys_id,
                        tsp=self.tsp, ctxt=self.ctxt.copy())
