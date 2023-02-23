"""Script to train an LSTM-AE for NAT data"""
import sys

# data manipulation
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# data visualization
import matplotlib.pyplot as plt

# native python's libraries
import os
import time
import pickle

# deep learning libraries
import tensorflow as tf
from tensorflow import keras as kr

# constants
seed = 420
tf.random.set_seed(seed)

# to tweak
epochs = 300

# no touching here
cycpath = 'E:\\PhD\\Data\\clean\\2021\\NAT_sample'
testpath = 'E:\\PhD\\Data\\clean\\2021\\NAT_testAE'
cyctypes = ['o', 'f']
atts = ['sampled_current', 'sampled_position', 'sampled_voltage', 'sampled_fdcf', 'sampled_fdcv']
colors = ['tab:green', 'tab:blue', 'tab:orange', 'tab:brown', 'tab:grey']


# functions
def encoder(input_, ctype_):
    # input = (#cycles, timesteps, dim)
    # masking = kr.layers.Masking(mask_value=0.0)(input_)
    encoded = kr.layers.LSTM(100, activation='tanh', return_sequences=True,
                             name=f'enc_01_{ctype_}_{training_mode}')(input_)
    encoded = kr.layers.LSTM(80, activation='tanh', return_sequences=True,
                             name=f'enc_02_{ctype_}_{training_mode}')(encoded)
    encoded = kr.layers.LSTM(60, activation='tanh', return_sequences=True,
                             name=f'enc_03_{ctype_}_{training_mode}')(encoded)

    # NB: have 20 features in the end
    # encoded = kr.layers.LSTM(40, activation='tanh', return_sequences=True, name='enc_04')(encoded)
    # encoded = kr.layers.LSTM(20, activation='tanh', return_sequences=False, name='features')(encoded)

    # NB: have 40 features in the end
    encoded = kr.layers.LSTM(40, activation='tanh', return_sequences=False,
                             name=f'features_{ctype_}_{training_mode}')(encoded)
    return encoded


def decoder(encoded_, timesteps_, dim_, ctype_):
    # NB: from 20 features codebook
    # decoded = kr.layers.RepeatVector(timesteps, name='repeat-vector')(encoded_)
    # decoded = kr.layers.LSTM(20, activation='tanh', return_sequences=True, name='dec_01')(decoded)
    # decoded = kr.layers.LSTM(40, activation='tanh', return_sequences=True, name='dec_02')(decoded)

    # NB: from 40 features codebook
    decoded = kr.layers.RepeatVector(timesteps_, name=f'repeat_vector_{ctype_}_{training_mode}')(encoded_)
    decoded = kr.layers.LSTM(40, activation='tanh', return_sequences=True,
                             name=f'dec_01_{ctype_}_{training_mode}')(decoded)

    decoded = kr.layers.LSTM(60, activation='tanh', return_sequences=True,
                             name=f'dec_03_{ctype_}_{training_mode}')(decoded)
    decoded = kr.layers.LSTM(80, activation='tanh', return_sequences=True,
                             name=f'dec_04_{ctype_}_{training_mode}')(decoded)
    decoded = kr.layers.LSTM(100, activation='tanh', return_sequences=True,
                             name=f'dec_05_{ctype_}_{training_mode}')(decoded)
    decoded = kr.layers.TimeDistributed(kr.layers.Dense(dim_, name=f'dense_recons_{ctype_}_{training_mode}'),
                                        name=f'recons_{ctype_}_{training_mode}')(decoded)
    return decoded


def classifier(encoded_, ctype_, n_classes_):
    den = kr.layers.Dense(20, activation='relu', name=f'clf_01_{ctype_}_{training_mode}')(encoded_)
    out = kr.layers.Dense(n_classes_, activation='softmax', name=f'classification_{ctype_}_{training_mode}')(den)
    return out


def offline_training(model, X_train_, y_train_, y_ctxt_train, X_val_, y_val_, y_ctxt_val):
    """
    TRAIN AN OFFLINE MODEL

    :return:
    """
    train_history = model.fit(X_train_, [y_train_, y_ctxt_train],
                              validation_data=(X_val_, [y_val_, y_ctxt_val]),
                              batch_size=32, epochs=epochs, verbose=1)
    return model, train_history, []


def online_training(model, X_train_, y_train_, y_ctxt_train, X_val_, y_val_, y_ctxt_val, is_incremental):
    """
    TRAIN AN ONLINE MODEL

    :return:
    """
    N = len(X_train_)

    # features returned as the model is training
    features_ = []

    # store the losses
    losses = {
        'loss': [],
        f'recons_{ctype}_{training_mode}_loss': [],
        f'classification_{ctype}_{training_mode}_loss': []
    }

    # for each new data instace, train the model on ONE epoch
    for i in range(N):
        if is_incremental:
            # USE M PREVIOUS EXAMPLES TO TRAIN THE MODEL (M <= EPOCHS)
            if (i+1) <= epochs:
                i_start, i_end = 0, i + 1
            else:
                i_start, i_end = i + 1 - epochs, i + 1
            x = X_train_[i_start:i_end]
            y = y_train_[i_start:i_end]
            y_ctxt = y_ctxt_train[i_start:i_end]
            print(f'Example {i+1} / {N} ({len(x)} training instances)...')
            i_loss = model.train_on_batch(x, [y, y_ctxt], reset_metrics=False)
        else:
            # USE ONE EXAMPLE TO TRAIN THE MODEL ---
            # change dimension to please tensorflow
            x = np.expand_dims(X_train_[i], axis=0)
            y = np.expand_dims(y_train_[i], axis=0)
            y_ctxt = np.expand_dims(y_ctxt_train[i], axis=0)
            i_loss = model.train_on_batch(x, [y, y_ctxt], reset_metrics=False)

        # record the training loss
        for _, mt in enumerate(model.metrics_names):
            losses[mt].append(i_loss[_])

        # extract features for this one cycle after each training
        enc_ = kr.models.Model(joint_model.inputs, joint_model.layers[4].output)
        feat = enc_.predict(x)
        features_.append(feat)

    return model, losses, features_


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Please select between "online" or "offline" or "online-incremental"')
        sys.exit(0)

    training_mode = sys.argv[1]

    """
    LOAD THE CYCLES IN FOR ANALYSIS (FROM PICKLE)
    """
    start = time.perf_counter()
    # load data from pickled file
    with open('nat_training_data.pkl', mode='rb') as f:
        res = pickle.load(f)
        X, cycles, train_files, stations, temperatures, missions = res[0], res[1], res[2], res[3], res[4], res[5]
    # display some number
    for ct in cyctypes:
        print(f'[{ct}] {len(X[ct])} cycles, size: {X[ct].shape}')
    # record time
    end = time.perf_counter()
    print(f'Loading the cycles: {(end - start) / 60:.3f} mins\n')

    """
    FOR EACH CYCLE TYPE
    """
    for ctype in cyctypes:
        print(f'\n{ctype.upper()} CYCLES\n')
        metrics = {'dataset': 'nat', 'mode': training_mode, 'cycle_type': ctype, 'n_clean_cycles': len(X[ctype])}

        # ONE-HOT ENCODED THE STATION --------------------------
        start = time.perf_counter()
        y_stations = OneHotEncoder(sparse=False).fit_transform(np.array(stations[ctype]).reshape(-1, 1))
        n_classes = y_stations.shape[1]
        print(f'Number of stations: {n_classes}')
        print('Shape of the labels:', y_stations.shape)
        end = time.perf_counter()
        print(f'One-hot encoding: {(end - start) / 60:.3f} mins\n')
        metrics['n_stations'] = n_classes

        # BUILD THE AE -------------------------------------------
        timesteps = X[ctype].shape[1]
        dim = len(atts)
        metrics['timesteps'] = timesteps
        metrics['dim'] = dim
        # input layer
        input_cyc = kr.layers.Input(shape=(timesteps, dim), name=f'input_layer_{ctype}')
        # full joint model
        enc = encoder(input_cyc, ctype)
        joint_model = kr.models.Model(input_cyc,
                                      [decoder(enc, timesteps, dim, ctype), classifier(enc, ctype, n_classes)],
                                      name=f'nat-{ctype}-joint-model-{training_mode}')
        joint_model.compile(
            loss={f'recons_{ctype}_{training_mode}': 'mean_squared_error',
                  f'classification_{ctype}_{training_mode}': 'categorical_crossentropy'},
            optimizer=kr.optimizers.Adam(learning_rate=0.0001)
        )
        joint_model.summary()
        # draw the model
        tf.keras.utils.plot_model(
            joint_model, show_shapes=True, show_layer_names=True,
            to_file=f'./plots/nat/nat-{ctype}-model-40ft-{training_mode}.png',
        )

        # SPLIT TO TRAINING & VALIDATION SET -------------------------------------
        X_train, X_val, y_train, y_val, y_stations_train, y_stations_val = train_test_split(
            X[ctype], X[ctype], y_stations, test_size=0.2, random_state=seed)
        print('Training set:', X_train.shape, y_train.shape, y_stations_train.shape)
        print('Validation set:', X_val.shape, y_val.shape, y_stations_val.shape)
        print()
        metrics['n_train'] = len(X_train)
        metrics['n_val'] = len(X_val)

        """
        LAUNCH THE TRAINING IN ONLINE OR OFFLINE MODE
        """
        start = time.perf_counter()
        # OFFLINE TRAINING
        if training_mode == 'offline':
            joint_model, history, train_features = offline_training(joint_model, X_train, y_train, y_stations_train,
                                                                    X_val, y_val, y_stations_val)
        # ONLINE TRAINING: BY INSTANCE OR INCREMENTALLY
        else:
            if training_mode == 'online':
                joint_model, history, train_features = online_training(joint_model, X_train, y_train, y_stations_train,
                                                                       X_val, y_val, y_stations_val,
                                                                       is_incremental=False)
            else:
                joint_model, history, train_features = online_training(joint_model, X_train, y_train, y_stations_train,
                                                                       X_val, y_val, y_stations_val,
                                                                       is_incremental=True)

        end = time.perf_counter()
        print(f'Training: {(end - start) / 60:.3f} mins')
        metrics['training_time'] = end - start
        with open(f'./logs/nat/train_history_{ctype}_{training_mode}.pkl', mode='wb') as f:
            if training_mode == 'offline':
                pickle.dump(history.history, f)
            else:
                pickle.dump(history, f)

        """
        VISUALIZE THE RECONSTRUCTED CYCLE AND THE TRAINING LOSS
        """
        fig = plt.figure(figsize=(18, 3.5))
        n_epochs = range(epochs)
        # training loss
        ax = fig.add_subplot(1, 3, 1)
        loss = history.history['loss'] if training_mode == 'offline' else history['loss']
        ax.plot(loss, label='Training loss')
        if training_mode == 'offline':
            val_loss = history.history['val_loss']
            ax.plot(val_loss, label='Validation loss')
        ax.set_title('Training and validation loss' if training_mode == 'offline' else 'Training loss',
                     fontsize=16)
        ax.legend(fontsize=12)
        # an example cycle
        ax = fig.add_subplot(1, 3, 2)
        cyc = X[ctype][0]
        ax.plot(cyc)
        ax.set_title('Original', fontsize=16)
        # the reconstructed cycle
        ax = fig.add_subplot(1, 3, 3)
        recons, _ = joint_model.predict(cyc.reshape(1, cyc.shape[0], cyc.shape[1]))
        recons = np.squeeze(recons)
        ax.plot(recons)
        ax.set_title('Reconstructed', fontsize=16)
        # save the figure
        plt.savefig(f'./plots/nat/nat-{ctype}-results-40ft-{training_mode}.png')
        plt.close()

        """
        SAVE THE MODEL
        """
        print('\nSaving the model...')
        model_path = f'./models/nat/joint-03-180123/{ctype}/{training_mode}'
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        # save the full model
        joint_model.save(f'{model_path}/full')
        # save only the encoder
        encoder_part = kr.models.Model(joint_model.inputs, joint_model.layers[4].output)
        encoder_part.save(f'{model_path}/encoder')

        """
        EXTRACT FEATURES FOR TEST CYCLES
        """
        # extract all features ONCE
        print(f'\nExtracting feature vectors for {ctype} cycles...')
        start = time.perf_counter()

        # read all the cycles to test
        to_predict = []
        for f in os.listdir(testpath):
            test_df = pd.read_pickle(os.path.join(testpath, f))
            test_cyc, test_ctype = test_df[atts].values, f.replace('p.pkl', '')[-1]
            if ctype == test_ctype:
                to_predict.append(test_cyc)
        # pad them
        to_predict = kr.preprocessing.sequence.pad_sequences(to_predict, dtype=float, value=0.0,
                                                             maxlen=X[ctype].shape[1],
                                                             padding='pre' if ctype == 'o' else 'post')
        print('Padded test cycles:', to_predict.shape)
        # predict all at once
        features = np.array(encoder_part.predict(to_predict))
        print('Feature vectors:', features.shape)
        np.save(f'./features/nat/nat-{ctype}-features-{training_mode}.npy', features)

        # if training mode, also save the features extracted while training
        if training_mode != 'online':
            train_features = np.array(train_features)
            print('Training features:', train_features.shape)
            np.save(f'./features/nat/nat-{ctype}-trainfeatures-{training_mode}.npy', train_features)

        end = time.perf_counter()
        print(f'Extracting features: {(end - start) / 60} mins')
        metrics['n_test'] = len(to_predict)
        metrics['extract_features_time'] = end - start

        # save in a metric file
        metrics_fname = f'./logs/nat/metrics_40ft_{training_mode}.csv'
        if not os.path.exists(metrics_fname):
            header = ','.join([k for k in metrics.keys()])
            with open(metrics_fname, mode='w') as f:
                f.write(header + '\n')
        with open(metrics_fname, mode='a+') as f:
            line = ','.join([str(v) for k, v in metrics.items()]) + '\n'
            f.write(line)
