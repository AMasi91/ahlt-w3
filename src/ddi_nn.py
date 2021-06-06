from nltk import pos_tag

from src.utils.data_generator import DatasetGenerator
from keras.models import Model, load_model, Input
from keras.initializers import he_normal
from keras import optimizers, Sequential
from keras.layers import Conv1D, GlobalMaxPool1D, Embedding, Dense, TimeDistributed, Bidirectional, Concatenate, \
    Flatten, LSTM
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from src.utils.utility import print_sentences_len_hist, plot_training
import numpy as np
import json
from nltk.corpus import stopwords as stopwords
from eval import evaluator

LEN_AFFIX = 3
sw = stopwords.words('english')


def learn(train_dir, val_dir, model_name=None):
    train_data = load_data(train_dir)
    val_data = load_data(val_dir)

    # # TODO in the next line --> calculate the max len between all sentences:
    #max_len = max(len(value) for value in train_data.values())
    # # TODO in the next line --> print a useful histogram of sentences length:
    #print_sentences_len_hist(train_data.values(), show_max=50)
    #
    indexes = create_indexes(train_data, max_len=75)
    #
    optimizer = optimizers.Adam(learning_rate=0.01)
    model = build_network(indexes, optimizer)
    #
    X_train = encode_words(train_data, indexes)
    y_train = encode_labels(train_data, indexes)
    X_val = encode_words(val_data, indexes)
    y_val = encode_labels(val_data, indexes)
    #
    # # TODO note: My pc cannot allow setting steps_per_epochs and validation_steps...
    # # Better focus the training of maximizing the reduction of val loss --> better generalization!
    batch_size = 64
    epochs = 16
    patience = 3
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto')
    mc = ModelCheckpoint(f'../saved_models_ddi/mc_{model_name}.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=1, min_lr=0.001, verbose=1)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size,
              epochs=epochs,
               callbacks=[es, mc, rlr],
               verbose=1)
    #plot_training(history, model_name, task='ddi')
    save_indexes(indexes, model_name)

def predict(model_name , data_dir):
    '''Loads a NN  model  from  file ’modelname ’ and  uses it to  extract  drugs4in  datadir.
    Saves  results  to ’outfile ’ in the  appropriate  format'''
    # load  model  and  associated  encoding  data
    model , indexes = load_model_and_indexes(model_name)
    # load  data to  annotate
    test_data = load_data(data_dir)
    # encode  dataset
    encoded_words = encode_words(test_data, indexes)
    X_test = encoded_words
    # tag  sentences  in  dataset
    y_pred = model.predict(X_test, verbose=1)
    # get  most  likely  tag  for  each  pair. Recall indexes['labels'] is dict and [np.argmax(y)] a key.
    Y = []
    for y in y_pred:
        Y.append(np.argmax(y))

    # extract  entities  and  dump  them to  output  file
    out_file_path = f'../results_ner/results_{model_name}.txt'

    output_interactions(test_data , Y, out_file_path)
    # evaluate  using  official  evaluator.
    evaluation(data_dir, out_file_path)

def load_data(data_dir):
    dg = DatasetGenerator(split_path=data_dir, task='ddi')
    return dg.get_dataset_split()

def evaluation(data_dir, out_file_path):
    evaluator.evaluate("DDI", data_dir, out_file_path)

def create_indexes(train_data, max_len=100):
    index_dict = {'words': {'<PAD>': 0, '<UNK>': 1},
                  'labels': {'null': 0, 'mechanism': 1, 'advise': 2, 'effect': 3,
                        'int': 4},
                  'suffixes': {'<PAD>': 0, '<UNK>':1},
                  'prefixes': {'<PAD>': 0, '<UNK>': 1},
                  'pos': {'<PAD>': 0, '<UNK>': 1},
                  'maxLen': max_len}
    word_index = 2
    suf_index = 2
    pref_index = 2
    pos_index = 2
    word_dict = index_dict['words']
    suffix_dict = index_dict['suffixes']
    prefix_dict = index_dict['prefixes']
    pos_dict = index_dict['pos']
    for instances_of_a_sentence in train_data.values():
        for elem in instances_of_a_sentence:
            if not isinstance(elem,list): continue
            for instance in elem:
                word = instance[0]
                if word not in word_dict:
                    word_dict[word] = word_index
                    word_index += 1

                # get the 3 length suffix/prefix if it is not a stop word.
                if word not in sw and len(word) > LEN_AFFIX:
                    suffix = word[-LEN_AFFIX:]
                    prefix = word[:LEN_AFFIX]
                    if suffix not in suffix_dict:
                        suffix_dict[suffix] = suf_index
                        suf_index += 1
                    if prefix not in prefix_dict:
                        prefix_dict[prefix] = pref_index
                        pref_index += 1

                tag = instance[3]
                if tag not in pos_dict:
                    pos_dict[tag] = pos_index
                    pos_index += 1

    return index_dict

def build_network(indexes, optimizer):
    n_words = len(indexes['words'])
    n_suff = len(indexes['suffixes'])
    n_pref = len(indexes['prefixes'])
    n_labels = len(indexes['labels'])
    n_pos = len(indexes['pos'])

    max_len = indexes['maxLen']

    word_embedding_size = max_len - int(max_len*0.1)  # max sentence len + 10%
    suffix_embedding_size = word_embedding_size # for now they have the same length
    prefix_embedding_size = word_embedding_size
    pos_embedding_size = word_embedding_size

    # 3 input layers, one for each feature
    input_words = Input(shape=(max_len,))
    input_prefixes = Input(shape=(max_len,))
    input_suffixes = Input(shape=(max_len,))
    input_pos = Input(shape=(max_len,))

    # 3 embeddings (one for each input)
    word_emb = Embedding(input_dim=n_words,  output_dim=word_embedding_size, input_length=max_len)(input_words)
    pref_emb = Embedding(input_dim=n_pref, output_dim=prefix_embedding_size, input_length=max_len, mask_zero=True)(input_prefixes)
    suff_emb = Embedding(input_dim=n_suff, output_dim=suffix_embedding_size, input_length=max_len, mask_zero=True)(input_suffixes)
    pos_emb = Embedding(input_dim=n_pos, output_dim=pos_embedding_size, input_length=max_len, mask_zero=True)(input_pos)

    # concatenate embeddings and feed the convolutional model
    input = word_emb
    #cnn_model = Sequential()(input)
    #model = Concatenate([word_emb, pref_emb, suff_emb,pos_emb])(model)
    cnn_model = Conv1D(filters=128, kernel_size=5, activation='relu', kernel_initializer=he_normal())(input)
    #model.add(Conv1D(filters=128, kernel_size=5, activation='relu', kernel_initializer=he_normal()))
    #model.add(Conv1D(filters=50, kernel_size=5, activation='relu', kernel_initializer=he_normal()))
    cnn_model = GlobalMaxPool1D()(cnn_model)
    cnn_model = Dense(units=100, activation='relu')(cnn_model)
    out = Dense(units=n_labels, activation="softmax")(cnn_model)


    #input = Embedding(input_dim=n_words,output_dim=word_embedding_size, input_length=max_len)(input_words)
    #lstm_model = Sequential()(input)
    #lstm_model = Bidirectional(LSTM(units=word_embedding_size, return_sequences=True, recurrent_dropout=0.1, dropout=0.1,
    #                           kernel_initializer=he_normal()))(input)

    #merge = Concatenate()([lstm_model, cnn_model])
    #hidden1 = Dense(units=100, activation='relu')
    #hidden2 = Dense(units=n_labels, activation="softmax")
    #conc_model = Sequential()
    #conc_model.add(merge)
    #conc_model.add(hidden1)
    #conc_model.add(hidden2)

    #model = Bidirectional(LSTM(units=word_embedding_size, return_sequences=True, recurrent_dropout=0.1, dropout=0.1,
    #                           kernel_initializer=he_normal()))(model)
    #out = TimeDistributed(Dense(n_labels, activation="softmax"))(model)

    model = Model(input_words, out)
    #model = Model([input_words, input_prefixes, input_suffixes, input_pos], out)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    #3print(conc_model.summary())

    return model


def encode_words(split_data, indexes):
    word_dict = indexes['words']
    max_len = indexes['maxLen']
    encoded_matrix = []
    for instances_of_a_sentence in split_data.values():
        #pos_tags_of_sentence = pos_tag([token[0] for token in instances_of_a_sentence])

        encoded_sentence = []

        for idx, instance in enumerate(instances_of_a_sentence[3]):
            # If the sentence is bigger than max_len we need to cut the sentence
            if idx < max_len:
                word = instance[0]
                if word in word_dict:
                    encoded_sentence.append(word_dict[word])
                else:  # '<UNK>' : 1
                    encoded_sentence.append(1)
            else:
                break
        sent_len = len(encoded_sentence)
        # Check if we need padding
        if max_len - sent_len > 0:
            padding = [0] * (max_len - sent_len)
            encoded_sentence.extend(padding)
        encoded_matrix.append(encoded_sentence)
    return np.array(encoded_matrix)

def encode_affixes(split_data, indexes) -> tuple:
    # load both dictionaries. Instantiate the lists which will contain all the encoded sentences for
    # both prefix and suffix policies.

    suffix_dict = indexes['suffixes']
    prefix_dict = indexes['prefixes']
    max_len = indexes['maxLen']  # TODO: try different max lentgths for both approaches
    encoded_dataset_suf = []
    encoded_dataset_pre = []

    for instances_of_a_sentence in split_data.values():
        encoded_sentence_suf = []
        encoded_sentence_pref = []
        # words that do not have the required length or are stopwords will be filtered out.
        # TODO: (solved) better just put a <UNK>

        for idx, instance in enumerate(instances_of_a_sentence):
            # If the sentence is bigger than max_len we need to cut the sentence
            if idx < max_len:
                word = instance[0]
                # words that do not have the required length or are stopwords have <UNK>
                if len(word) < LEN_AFFIX:
                    encoded_sentence_suf.append(suffix_dict['<UNK>'])
                    encoded_sentence_pref.append(prefix_dict['<UNK>'])
                    continue

                # beyond this point, words are adequate to extract affixes.

                suffix = word[-LEN_AFFIX:]
                prefix = word[:LEN_AFFIX]

                if suffix in suffix_dict:
                    encoded_sentence_suf.append(suffix_dict[suffix])
                else:  # '<UNK>' : 1
                    encoded_sentence_suf.append(suffix_dict['<UNK>'])

                if prefix in prefix_dict:
                    encoded_sentence_pref.append(prefix_dict[prefix])
                else:  # '<UNK>' : 1
                    encoded_sentence_pref.append(prefix_dict['<UNK>'])
            else:
                break
        sent_len_suf = len(encoded_sentence_suf)
        sent_len_pref = len(encoded_sentence_pref)
        # Check if we need padding
        if max_len - sent_len_suf > 0:
            padding = [suffix_dict['<PAD>']] * (max_len - sent_len_suf)
            encoded_sentence_suf.extend(padding)

        if max_len - sent_len_pref > 0:
            padding = [prefix_dict['<PAD>']] * (max_len - sent_len_pref)
            encoded_sentence_pref.extend(padding)


        encoded_dataset_suf.append(encoded_sentence_suf)
        encoded_dataset_pre.append(encoded_sentence_pref)

    return (np.array(encoded_dataset_pre), np.array(encoded_dataset_suf))

def encode_labels(split_data, indexes):
    label_dict = indexes['labels']
    #max_len = indexes['maxLen']
    encoded_labels = []
    for idx, instances_of_a_sentence in enumerate(split_data.values()):
        ddi = instances_of_a_sentence[2]
        if True: #idx < max_len:
            if ddi in label_dict:
                encoded_labels.append(label_dict[ddi])
            else:
                encoded_labels.append(1)
        #else:
        #    break

    encoded_label_matrix = np.array(encoded_labels)
    y = [to_categorical(i, num_classes=len(label_dict.values())) for i in encoded_label_matrix]
    return np.array(y)

def encode_postags(split_data, indexes):
    pos_dict = indexes['pos']
    max_len = indexes['maxLen']
    encoded_matrix = []
    for instances_of_a_sentence in split_data.values():
        pos_tags_of_sentence = [analysis[1] for analysis in pos_tag([token[0] for token in instances_of_a_sentence])]
        encoded_sentence = []
        for idx, tag in enumerate(pos_tags_of_sentence):
            # If the sentence is bigger than max_len we need to cut the sentence
            if idx < max_len:
                if tag in pos_dict:
                    encoded_sentence.append(pos_dict[tag])
                else:  # '<UNK>' : 1
                    encoded_sentence.append(1)
            else:
                break
        sent_len = len(encoded_sentence)
        # Check if we need padding
        if max_len - sent_len > 0:
            padding = [0] * (max_len - sent_len)
            encoded_sentence.extend(padding)
        encoded_matrix.append(encoded_sentence)
    return np.array(encoded_matrix)


def save_indexes(indexes, model_name):
    with open(f'../saved_models_ddi/encoding_{model_name}.json', 'w') as fp:
        json.dump(indexes, fp, indent=4)


def load_model_and_indexes(model_name):
    with open(f'../saved_models_ddi/encoding_{model_name}.json', 'r') as fp:
        indexes = json.load(fp)
    model = load_model(f'../saved_models_ddi/mc_{model_name}.h5')

    return model, indexes

def output_interactions(test_data , y_pred, out_file_path):
    out_file = open(out_file_path, "w+")
    for idx, instance in enumerate(test_data.items()):
        sid, id_e1, id_e2 = instance[0], instance[1][0], instance[1][1]
        ddi_type = decode_idx_DDI(y_pred[idx])

        if ddi_type is not None and ddi_type != "null":
            print(sid + "|" + id_e1 + "|" + id_e2 + "|" + str(ddi_type), file=out_file)

def decode_idx_DDI(ddi_encoded):
    if ddi_encoded == 0:
        return "null"
    elif ddi_encoded == 1:
        return 'mechanism'
    elif ddi_encoded == 2:
        return 'advise'
    elif ddi_encoded == 3:
        return 'effect'