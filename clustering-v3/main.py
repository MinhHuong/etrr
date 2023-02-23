"""
MAIN ENTRY POINT OF CHMOC.

- feed data to simulate a stream
- run CHMOC on the stream
"""

# python's native libraries
import os
import sys
import json
import time
from datetime import datetime as dt
import psutil

# custom
import numpy as np

# database-module
from database.mongo_conn import MongoDBConnector

# my module
from core.chmoc import CHMOC
from utils import TIMEUNIT, convert_time_unit_from_string
from feeder.r2n_feeder import R2NFeeder
from feeder.nat_feeder import NATFeeder


# main entry point
if __name__ == '__main__':

    # read the config file
    if len(sys.argv) <= 1:
        print('Please input a config file.')
        sys.exit(0)
    else:
        # read the time unit
        with open(os.path.join('config', sys.argv[1]), mode='r') as f:
            config = json.load(f)
        config['clustering']['time_unit'] = convert_time_unit_from_string(config['clustering']['time_unit'])
        # read the dataset
        dataset = config['dataset']

    # initialize a connection to mongodb
    mongo_db = MongoDBConnector(dataset=dataset)
    mongo_db.clean(expe_name=config['experiment']['expe_name'])

    # set up a new experiment
    expe_name = config['experiment']['expe_name']
    mongo_db.add_experiment(expe_name=expe_name, expe_note=config['experiment']['note'], expe_config=config)

    #  initialize CHMOC
    chmoc = CHMOC(config=config, db_conn=mongo_db)

    # read data in form of a stream
    start = time.perf_counter()
    if dataset == 'r2n':
        feeder = R2NFeeder(atts=config['preprocessing']['atts'],
                           excluded_cyc=config['preprocessing']['excluded_cycles'])
    else:
        feeder = NATFeeder(atts=config['preprocessing']['atts'], excluded_cyc='')
    end = time.perf_counter()
    print(f'Initializing the feeder: {(end - start) / 60:.3f} minutes')

    # get the info of this Python process
    process = psutil.Process(os.getpid())

    print('Start feeding the data.')
    i = 0
    cumu_start = time.perf_counter()
    while feeder.has_data():
        # fetch new data
        start = time.perf_counter()
        data = feeder.fetch_data()
        end = time.perf_counter()
        feed_time = end - start

        # print to keep track
        print(f'[{i}] Batch of day {int(feeder.weeks[feeder.current-1])}: {len(data)}')

        # process the new chunk
        start = time.perf_counter()
        # data_ = np.array([inst.data for inst in data])  # what's this doing???
        chmoc.process(batch=data)
        end = time.perf_counter()
        batch_time = end - start

        # save batch metrics in mongodb
        batch_metrics = {
            'start_time': dt.fromtimestamp(data[0].tsp),
            'end_time': dt.fromtimestamp(data[-1].tsp),
            'batch_number': i,
            'batch_size': len(data),
            # time measurement
            'batch_process_time': batch_time,  # total processing time of this batch
            'data_feeding_time': feed_time,  # feeding data from postgres
            'cumulative_time': time.perf_counter() - cumu_start,  # cumulative time (so far)
            'ws_time': chmoc.ws_time,  # warm start time
            'prune_time': chmoc.clusterer.pruning_time,  # pruning time
            'merge_time': chmoc.clusterer.merging_time,  # merging time inside DenStream
            'decay_time': chmoc.clusterer.decay_time,  # time to decay all clusters
            'denstream_time': chmoc.denstream_time,  # process all instances in denstream time
            'reassess_time': chmoc.reassess_time,  # reassessing clusters time
            'update_cluster_time': chmoc.updcluster_time,  # update all clusters at once
            'update_health_time': chmoc.updhealth_time,  # update health time
            # resource consumption management
            'memory_usage_bytes': process.memory_info().rss,  # memory usage in bytes
            # parameters of denstream
            'denstream': chmoc.get_clusterer_snapshot(),
            # 'cluster_snapshot': chmoc.get_cluster_snapshot(),
        }
        batch_id = mongo_db.update_experiment(expe_name=expe_name, expe_info=batch_metrics)
        mongo_db.add_cluster_snapshot(expe_name=expe_name, batch_id=batch_id, clusters=chmoc.get_cluster_snapshot())

        # early stop
        i += 1
        # if i == 2:
        #     break

    # shut things off
    mongo_db.close()
    feeder.close()
