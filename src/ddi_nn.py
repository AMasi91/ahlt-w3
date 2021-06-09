from nltk import pos_tag
from keras.initializers import Constant
from src.utils.data_generator import DatasetGenerator
from keras.models import Model, load_model, Input
from keras.initializers import he_normal
from keras import optimizers, Sequential
from keras.layers import Conv1D, GlobalMaxPool1D, Embedding, Dense, TimeDistributed, Bidirectional, Concatenate, \
    Flatten, LSTM, concatenate, MaxPooling1D, Dropout
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


    # filt_data = {}
    # for sid, sentence in val_data.items():
    #     analysis_pair = []
    #     for pair in sentence:
    #         analysis_pair.append(pair[3])
    #     filt_data[sid] = analysis_pair
    # val_data = filt_data

    # # TODO in the next line --> calculate the max len between all sentences:
    #max_len = max(len(value) for value in train_data.values())
    # # TODO in the next line --> print a useful histogram of sentences length:
    #print_sentences_len_hist(train_data.values()[-1], show_max=50)
    #
    indexes = create_indexes(train_data, max_len=75)
    #
    optimizer = optimizers.Adam(learning_rate=0.01)
    #optimizer = optimizers.sgd(learning_rate=0.01)
    model = build_network(indexes, optimizer)
    #
    words_enc,lemmas_enc = encode_words_and_lemmas(train_data, indexes)
    #prefixes_encoded, suffixes_encoded = encode_affixes(train_data, indexes)
    pos_enc = encode_postags(train_data, indexes)
    #X_train = [words_enc, lemmas_enc, pos_enc, prefixes_encoded, suffixes_encoded]
    y_train = encode_labels(train_data, indexes)
    X_train = [words_enc,lemmas_enc, pos_enc]


    words_enc, lemmas_enc = encode_words_and_lemmas(val_data, indexes)
    prefixes_encoded, suffixes_encoded = encode_affixes(val_data, indexes)
    pos_enc = encode_postags(val_data, indexes)
    #X_val = [words_enc, lemmas_enc, pos_enc, prefixes_encoded, suffixes_encoded]
    y_val = encode_labels(val_data, indexes)
    X_val = [words_enc, lemmas_enc, pos_enc]
    #
    # # TODO note: My pc cannot allow setting steps_per_epochs and validation_steps...
    # # Better focus the training of maximizing the reduction of val loss --> better generalization!
    batch_size = 64
    epochs = 1
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
    encoded_words, encoded_lemmas = encode_words_and_lemmas(test_data, indexes)
    prefixes_encoded, suffixes_encoded = encode_affixes(test_data, indexes)
    pos_enc = encode_postags(test_data, indexes)
    #X_test = [encoded_words, encoded_lemmas, pos_enc, prefixes_encoded, suffixes_encoded]
    X_test = [encoded_words, encoded_lemmas, pos_enc]
    # tag  sentences  in  dataset
    y_pred = model.predict(X_test, verbose=1)
    # get  most  likely  tag  for  each  pair. Recall indexes['labels'] is dict and [np.argmax(y)] a key.
    Y = []
    for y in y_pred:
        Y.append(np.argmax(y))

    # extract  entities  and  dump  them to  output  file
    out_file_path = f'../results_ddi/results_{model_name}.txt'

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
                  'lemmas': {'<PAD>': 0, '<UNK>': 1},
                  'maxLen': max_len}
    word_index = 2
    lemma_index = 2
    suf_index = 2
    pref_index = 2
    pos_index = 2
    word_dict = index_dict['words']
    lemmas_dict = index_dict['lemmas']
    suffix_dict = index_dict['suffixes']
    prefix_dict = index_dict['prefixes']
    pos_dict = index_dict['pos']
    for instances_of_a_sentence in train_data.values():
        for pair_descriptor in instances_of_a_sentence:
            try:
                pair_descriptor[3]
            except:
                continue
            if not isinstance(pair_descriptor[3],list): continue
            for instance in pair_descriptor[3]:
                word = instance[0]
                lemma = instance[-1]
                if word not in word_dict:
                    word_dict[word] = word_index
                    word_index += 1
                if lemma not in lemmas_dict:
                    lemmas_dict[lemma] = lemma_index
                    lemma_index += 1

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
                if tag not in pos_dict and tag.isalpha():
                    pos_dict[tag] = pos_index
                    pos_index += 1

    return index_dict

def build_network(indexes, optimizer):
    n_words = len(indexes['words'])
    n_labels = len(indexes['labels'])
    n_pos = len(indexes['pos'])
    n_lemmas = len(indexes['lemmas'])
    n_suffixes = len(indexes['suffixes'])
    n_prefixes = len(indexes['prefixes'])

    max_len = indexes['maxLen']

    word_embedding_size = 150
    pos_embedding_size = 25
    lemma_embedding_size = 110
    prefixes_embedding_size = 200
    suffixes_embedding_size = 200
    total_size = word_embedding_size + pos_embedding_size + lemma_embedding_size + 400

    # 3 input layers, one for each feature
    input_words = Input(shape=(max_len,))
    input_lemmas = Input(shape=(max_len,))
    input_pos = Input(shape=(max_len,))
    input_pre = Input(shape=(max_len,))
    input_suf = Input(shape=(max_len,))


    # 3 embeddings (one for each input)
    emb_word_mat = create_embedding_matrix('../resources/glove.6B.200d.txt', indexes['words'], 150, num_reserved=2)
    emb_lemma_mat = create_embedding_matrix('../resources/glove.6B.200d.txt', indexes['lemmas'], 110, num_reserved=2)
    emb_pos_mat = create_embedding_matrix('../resources/glove.6B.50d.txt', indexes['pos'], 25, num_reserved=2)

    word_emb = Embedding(input_dim=n_words, output_dim=word_embedding_size, input_length=max_len)(input_words)
    lemma_emb = Embedding(input_dim=n_lemmas, output_dim=lemma_embedding_size, input_length=max_len)(input_lemmas)
    pos_emb = Embedding(input_dim=n_pos, output_dim=pos_embedding_size, input_length=max_len)(input_pos)
    #pref_emb = Embedding(input_dim=n_prefixes, output_dim=prefixes_embedding_size, input_length=max_len)(input_pre)
    #suff_emb = Embedding(input_dim=n_suffixes, output_dim=suffixes_embedding_size, input_length=max_len)(input_suf)

    #word_emb = Embedding(input_dim=n_words+2,  output_dim=word_embedding_size, input_length=max_len,
    #                     embeddings_initializer=Constant(emb_word_mat),trainable=False,)(input_words)
    #lemma_emb = Embedding(input_dim=n_lemmas+2, output_dim=lemma_embedding_size, input_length=max_len,
    #                      embeddings_initializer = Constant(emb_lemma_mat), trainable = False,)(input_lemmas)
    #pos_emb = Embedding(input_dim=n_pos+2, output_dim=pos_embedding_size, input_length=max_len,
    #                    embeddings_initializer = Constant(emb_pos_mat), trainable = False,)(input_pos)
    #aux = concatenate([word_emb], [lemma_emb])
    #cnn_model = concatenate(aux, [pos_emb])
    # concatenate embeddings and feed the convolutional model
    ##################BASIC MODEL#################
    cnn_model = Concatenate()([word_emb, lemma_emb, pos_emb])

    #cnn_model = Conv1D(filters=40, kernel_size=4, activation='relu', padding='valid', kernel_initializer=he_normal())(cnn_model)
    ##################APPROACH#################
    # filter_sizes = [2,3,4,5]
    # convs = []
    # for filter_size in filter_sizes:
    #     l_conv = Conv1D(filters=128, kernel_size=filter_size, activation='relu', kernel_initializer=he_normal())(
    #     cnn_model)
    #     l_pool = GlobalMaxPool1D()(l_conv)
    #     convs.append(l_pool)
    # cnn_model = concatenate(convs, axis=1)
    #
    # cnn_model = Dropout(0.1)(cnn_model)
    ############################################




    cnn_model = Conv1D(filters=128, kernel_size=4, activation='relu', padding='valid', kernel_initializer=he_normal())(
        cnn_model)
    #cnn_model = Conv1D(filters=128, kernel_size=4, activation='relu', padding='valid', kernel_initializer=he_normal())(
    #    cnn_model)
    #cnn_model = Conv1D(filters=40, kernel_size=4, activation='relu', padding='valid', kernel_initializer=he_normal())(cnn_model)
    #cnn_model = GlobalMaxPool1D()(cnn_model)
    #cnn_model = Conv1D(filters=10, kernel_size=4, activation='relu', padding='valid', kernel_initializer=he_normal())(
    #    cnn_model)

    #cnn_model = Conv1D(filters=25, kernel_size=4, activation='relu', padding='valid', kernel_initializer=he_normal())(
    #    cnn_model)
    #cnn_model = MaxPooling1D(pool_size=2, strides=None, padding='valid',
     #                      input_shape=(max_len, 100))(cnn_model)  # strides=None means strides=pool_size
    #cnn_model = LSTM(units = 50, return_sequences=True, recurrent_dropout=0.1,
    #                 dropout=0.1, kernel_initializer=he_normal())(cnn_model)
    cnn_model = LSTM(units=40, return_sequences=True, recurrent_dropout=0.1,
                     dropout=0.1, kernel_initializer=he_normal())(cnn_model)
    cnn_model = GlobalMaxPool1D()(cnn_model)
    cnn_model = Dense(units=50, activation='relu')(cnn_model)
    #cnn_model = Dropout(0.2)(cnn_model)
    #cnn_model = Dense(units=50, activation='relu')(cnn_model)
    #cnn_model = Dropout(0.2)(cnn_model)
    out = Dense(units=n_labels, activation='softmax')(cnn_model)


    cnn_model = Model([input_words, input_lemmas, input_pos], out)
    #lstm_model = Concatenate()([word_emb, lemma_emb, pos_emb])
    #lstm_model = LSTM(word_embedding_size, activation='relu', return_sequences=True)(lstm_model)
    #lstm_model = LSTM(word_embedding_size, activation='relu', return_sequences=True)(lstm_model)

    #lstm_model = LSTM(word_embedding_size, activation='relu', return_sequences=True)(lstm_model)
    #lstm_model = Flatten()(lstm_model)

    #merge = Concatenate()([cnn_model, lstm_model])
    #merge = Dense(units=100, activation='relu')(merge)
    #merge = Dense(units=50, activation='relu')(merge)
    #out = Dense(units=n_labels, activation="softmax")(merge)

    #merge = Model([input_words, input_lemmas ,input_pos],  out)
    cnn_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    print(cnn_model.summary())

    return cnn_model

def encode_words_and_lemmas(split_data, indexes):
    word_dict = indexes['words']
    lemma_dict = indexes['lemmas']
    max_len = indexes['maxLen']
    encoded_matrix_words = []
    encoded_matrix_lemmas = []
    for instances_of_a_sentence in split_data.values():
        if not instances_of_a_sentence: # sentence dont have pair info
            continue
        for instance_of_a_pair in instances_of_a_sentence:
            encoded_sentence_word = []
            encoded_sentence_lemma = []

            for idx, instance in enumerate(instance_of_a_pair[3]):
                # If the sentence is bigger than max_len we need to cut the sentence
                if idx < max_len:
                    word = instance[0]
                    lemma = instance[-1]
                    if word in word_dict:
                        encoded_sentence_word.append(word_dict[word])
                    else:  # '<UNK>' : 1
                        encoded_sentence_word.append(1)
                    if lemma in lemma_dict:
                        encoded_sentence_lemma.append(lemma_dict[lemma])
                    else:
                        encoded_sentence_lemma.append(1)

                else:
                    break
                    # Check if we need padding for words

            sent_len = len(encoded_sentence_word)
            if max_len - sent_len > 0:
                padding = [0] * (max_len - sent_len)
                encoded_sentence_word.extend(padding)
            encoded_matrix_words.append(encoded_sentence_word)

            # Check if we need padding for lemmas
            sent_len = len(encoded_sentence_lemma)
            if max_len - sent_len > 0:
                padding = [0] * (max_len - sent_len)
                encoded_sentence_lemma.extend(padding)
            encoded_matrix_lemmas.append(encoded_sentence_lemma)



    return np.array(encoded_matrix_words), np.array(encoded_matrix_lemmas)

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
        if not instances_of_a_sentence: # sentence dont have pair info
            continue
        encoded_sentence = []
        for instance_of_a_pair in instances_of_a_sentence:
            ddi = instance_of_a_pair[2]
            if True: #idx < max_len:
                if ddi in label_dict:
                    encoded_sentence.append(label_dict[ddi])
                else:
                    encoded_sentence.append(1)
            #else:
            #    break
        encoded_labels += encoded_sentence

    encoded_label_matrix = np.array(encoded_labels)
    y = [to_categorical(i, num_classes=len(label_dict.values())) for i in encoded_label_matrix]
    return np.array(y)

def encode_postags(split_data, indexes):
    pos_dict = indexes['pos']
    max_len = indexes['maxLen']
    encoded_matrix = []
    for instances_of_a_sentence in split_data.values():
        if not instances_of_a_sentence: # sentence dont have pair info
            continue
        for instance_of_a_pair in instances_of_a_sentence:
            pos_tags_of_sentence = [analysis[3] for analysis in instance_of_a_pair[3]]
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
        sid, id_e1, id_e2 = instance[0], instance[1][0][0], instance[1][0][1]
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
    elif ddi_encoded == 4:
        return 'int'

# GlOVE
def create_embedding_matrix(filepath, word_index, embedding_dim, num_reserved = 1):
    vocab_size = len(word_index) + num_reserved
    # Adding again a because of reserved index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix