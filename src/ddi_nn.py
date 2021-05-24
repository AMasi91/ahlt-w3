from src.utils.data_generator import DatasetGenerator
from keras.models import Model, Input, load_model
from keras.initializers import he_normal
from keras import optimizers
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from src.utils.utility import print_sentences_len_hist, plot_training
import numpy as np
import json
from eval import evaluator


def learn(train_dir, val_dir, model_name=None):
    train_data = load_data(train_dir)
    val_data = load_data(val_dir)

    # # TODO in the next line --> calculate the max len between all sentences:
    # #max_len = max(len(value) for value in train_data.values())
    # # TODO in the next line --> print a useful histogram of sentences length:
    # print_sentences_len_hist(train_data.values(), show_max=50)
    #
    # indexes = create_indexes(train_data, max_len=50)
    #
    # optimizer = optimizers.Adam(learning_rate=0.01)
    # model = build_network(indexes, optimizer)
    #
    # X_train = encode_words(train_data, indexes)
    # y_train = encode_labels(train_data, indexes)
    # X_val = encode_words(val_data, indexes)
    # y_val = encode_labels(val_data, indexes)
    #
    # # TODO note: My pc cannot allow setting steps_per_epochs and validation_steps...
    # # Better focus the training of maximizing the reduction of val loss --> better generalization!
    # batch_size = 64
    # epochs = 16
    # patience = 3
    # es = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto')
    # mc = ModelCheckpoint(f'../saved_models_ner/mc_{model_name}.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    # rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=1, min_lr=0.001, verbose=1)
    # history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size,
    #           epochs=epochs,
    #           callbacks=[es, mc, rlr],
    #           verbose=1)
    # plot_training(history, model_name, task='ner')
    # save_indexes(indexes, model_name)

def load_data(data_dir):
    dg = DatasetGenerator(split_path=data_dir, task='ddi')
    return dg.get_dataset_split()