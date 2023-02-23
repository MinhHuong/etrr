# --------------------------- #
# MAIN ENTRY POINT OF INTERCE #
# --------------------------- #

from core.interce import InterCE
from core.interce_offline import InterCEOffline
import time
import threading
import os
import json
import pandas as pd
from tools.utils import preprocess
from database.dbmanager import DBConnection

# enable memory leak debugging
import gc
gc.set_debug(gc.DEBUG_SAVEALL)

# global variable so interce-query can signal to interce-feedback that it's done
query_done = False


def interce_offline_input(learner: InterCEOffline, conf: dict):
    """
    Run InterCE in an offline setting

    Parameters
    ----------
    learner
    conf

    Returns
    -------

    """
    global query_done   # signal that query_done is a global variable

    # ----------------- #
    # PARAMETERS TO FIX #
    # ----------------- #
    limit = 1000          # run on N files
    n_train = 400         # number of files for training

    n_files = 1         # current count of files
    print(f'\nInterCE (OFFLINE) processes {limit} files.')
    print(f'InterCE (OFFLINE) trains on {n_train} files.\n')
    inpath = conf['inpath']

    for entry in os.scandir(inpath):
        # skip if it's not a file
        if entry.name.startswith('.') or not entry.is_file():
            continue

        # get the base filename
        inp = entry.name

        # read the pickle from the input
        x = pd.read_pickle(os.path.join(inpath, inp))

        # preprocess the data as necessary, depending on the data set
        x = preprocess(X=x, case=conf['dataset'])

        # ignore the file if it is invalid
        if x is None:
            continue

        # for each file in the input folder
        print(f'[{n_files}] {inp}...')

        # if the data are valid (not None), throw it to InterCE and wreak havoc on my PC
        # distinguish between training and testing
        if n_files <= n_train:
            learner.is_training = True
            learner.train(X=x, X_id=inp)
        else:
            # training done, start processing the feedback ONLY if all the queries have been answered
            while dbconn.get_nb_unanswered_queries() != 0:
                time.sleep(2)
                continue

            # start processing feedback only when all the official queries have been answered
            # do it ONCE
            if dbconn.get_nb_unprocessed_feedback() != 0:
                print('\nMAIN: training done, start processing feedback')
                learner.process_feedback()

            # # wait here until all the queries and feedback have been processed
            # while not dbconn.has_finished():
            #     time.sleep(2)
            #     continue
            # once all queries and all feedback have been processed, let it enter the testing phase

            learner.is_training = False
            learner.test(X=x, X_id=inp)

        # write metrics (ONLY WHEN WE ARE NOT RUNNING IN THREAD)
        if n_files % 10 == 0:
            learner.save_metrics()

        # stopping criterion
        if n_files == limit:
            query_done = True
            break
        n_files += 1


def interce_input(learner: InterCEOffline, conf: dict):
    """
    Run InterCE on all input files

    Parameters
    ----------
    learner
    conf

    Returns
    -------

    """
    global query_done   # signal that query_done is a global variable

    limit = 1000        # run on 100K files
    n_files = 1
    print(f'\nInterCE processes {limit} files.\n')
    inpath = conf['inpath']

    for entry in os.scandir(inpath):
        # skip if it's not a file
        if entry.name.startswith('.') or not entry.is_file():
            continue

        # get the base filename
        inp = entry.name

        # read the pickle from the input
        x = pd.read_pickle(os.path.join(inpath, inp))

        # preprocess the data as necessary, depending on the data set
        x = preprocess(X=x, case=conf['dataset'])

        # ignore the file if it is invalid
        if x is None:
            continue

        # for each file in the input folder
        print(f'[{n_files}] {inp}...')

        # if the data are valid (not None), throw it to InterCE and wreak havoc on my PC
        learner.process_input(X=x, X_id=inp)

        # write metrics (ONLY WHEN WE ARE NOT RUNNING IN THREAD)
        if n_files % 10 == 0:
            learner.save_metrics()

        # stopping criterion
        if n_files == limit:
            query_done = True
            break
        n_files += 1


def interce_feedback(learner: InterCEOffline, conf: dict):
    """
    Let InterCE wait on new feedback (continuously run)

    Parameters
    ----------
    learner
    conf

    Returns
    -------

    """
    # stop when the query side is done (no more input) and all queries + feedback have been answered
    while not query_done or not learner.dbconn.has_finished():
        time.sleep(1)   # let it rest for 1 second before launching processing again
        learner.process_feedback()


def interce_metrics(learner: InterCE, conf: dict):
    """
    Save the metrics periodically

    Parameters
    ----------
    learner
    conf

    Returns
    -------

    """
    # run infinitely until there are no more files to process and all queries + feedback have been processed
    while not query_done or not learner.dbconn.has_finished():
        time.sleep(1)  # let it rest for 1 second before launching metric recording
        if learner.n_processed_files > 0 and learner.n_processed_files % 10 == 0:
            learner.save_metrics()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # read the config
    print('\nMAIN: reading the config...')
    with open('config/config_nat.json', 'r') as f:
        config = json.load(f)

    # create a connection to the database
    print('\nMAIN: creating a connection to the DB...')
    dbconn = DBConnection(dbname=config['database']['dbname'],
                          user=config['database']['user'],
                          pwd=config['database']['pwd'])
    # choice = input('We are going to clean all data in the database! Press "y" to proceed, "n" to exit: ')
    # if choice.lower() == 'n':
    #     sys.exit(0)
    dbconn.reset()

    # create the new experiment
    print('\nMAIN: adding a new experiment. to the DB..')
    expe_name = config['experiment']['name']
    expe_desc = config['experiment']['desc']
    expe_id = dbconn.add_experiment(expe_name=expe_name, expe_desc=expe_desc)
    print('MAIN: new experiment id:', expe_id)

    # start the timer
    start = time.perf_counter()

    # initialize InterCE
    print('\nMAIN: initializing InterCE...')
    # interce = InterCE(config, dbconn, expe_id)
    interce = InterCEOffline(config, dbconn, expe_id)

    # USE ONLY WHEN RUNNING SCALABILITY TEST
    # interce_input(interce, config)

    # (1) thread 1: create queries
    print('\nMAIN: launching InterCE querying side')
    # thread_query = threading.Thread(target=interce_input, args=(interce, config))
    thread_query = threading.Thread(target=interce_offline_input, args=(interce, config))
    thread_query.start()

    # UNCOMMENT IF RUN BACK IN ONLINE MODE
    # # (2) thread 2: process feedback
    # print('MAIN: launching InterCE feedback side')
    # thread_feedback = threading.Thread(target=interce_feedback, args=(interce, config))
    # thread_feedback.start()

    # (3) thread to write metrics to the database
    print('\nMAIN: launching InterCE metrics recording')
    thread_metrics = threading.Thread(target=interce_metrics, args=(interce, config))
    thread_metrics.start()

    # wait for the query thread to end
    thread_query.join()
    print(f'\nInterCE querying ended ({(time.perf_counter() - start) / 60} mins).')

    # # wait for the feedback thread to end
    # thread_feedback.join()
    # print(f'InterCE feedback ended ({(time.perf_counter() - start) / 60} mins).')

    # wait for the metrics thread to end
    thread_metrics.join()
    print(f'\nInterCE metrics ended ({(time.perf_counter() - start) / 60} mins).')

    # close the database connection
    dbconn.conn.close()

    # finish
    end = time.perf_counter()
    print(f'\nTOTAL TIME: {(end - start) / 60} mins')
