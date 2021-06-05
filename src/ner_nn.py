from utils.data_generator import DatasetGenerator
from keras.models import Model, Input, load_model
from keras.initializers import he_normal
from keras import optimizers
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, Concatenate
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from src.utils.utility import print_sentences_len_hist, plot_training
import numpy as np
# from keras_contrib.layers import CRF
import json
from nltk import pos_tag
from nltk.corpus import stopwords as stopwords
from eval import evaluator
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy


# Define global attributes
LEN_AFFIX = 3
sw = stopwords.words('english')

def learn(train_dir, val_dir, model_name=None):
    train_data = load_data(train_dir)
    val_data = load_data(val_dir)

    # TODO in the next line --> calculate the max len between all sentences:
    # max_len = max(len(value) for value in train_data.values())
    # TODO in the next line --> print a useful histogram of sentences length:
    # print_sentences_len_hist(train_data.values(), show_max=75)

    indexes = create_indexes(train_data, max_len=75)

    optimizer = optimizers.Adam(learning_rate=0.01)
    #model = build_network_with_CRF(indexes, optimizer) Only prepared for "word_embedding" input. Not affixes nor pos.
    model = build_network(indexes,optimizer)

    prefixes_encoded, suffixes_encoded = encode_affixes(train_data, indexes)
    pos_encoded = encode_postags(train_data, indexes)
    X_train = [encode_words(train_data, indexes),prefixes_encoded,suffixes_encoded, pos_encoded]
    y_train = encode_labels(train_data, indexes)

    prefixes_encoded, suffixes_encoded = encode_affixes(val_data, indexes)
    pos_encoded = encode_postags(val_data, indexes)
    X_val = [encode_words(val_data, indexes),prefixes_encoded,suffixes_encoded,pos_encoded]
    y_val = encode_labels(val_data, indexes)


    # TODO note: My pc cannot allow setting steps_per_epochs and validation_steps...
    # Better focus the training of maximizing the reduction of val loss --> better generalization!
    batch_size = 64
    epochs = 16
    patience = 3
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto')
    mc = ModelCheckpoint(f'../saved_models_ner/mc_{model_name}.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=1, min_lr=0.001, verbose=1)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size,
              epochs=epochs,
              callbacks=[es, mc, rlr],
              verbose=1)
    print(model.metrics_names)
    plot_training(history, model_name, task='ner')
    save_indexes(indexes, model_name)


def predict(model_name, data_dir):
    model, indexes = load_model_and_indexes(model_name)
    test_data = load_data(data_dir)
    encoded_words = encode_words(test_data, indexes)
    encoded_preffixes, encoded_suffixes = encode_affixes(test_data, indexes)
    pos_encoded = encode_postags(test_data, indexes)

    X_test = [encoded_words, encoded_preffixes, encoded_suffixes,pos_encoded ]
    y_pred = model.predict(X_test, verbose=1)
    # get most likely tag for each word
    pred_labels = y_pred_to_labels(y_pred, indexes)

    out_file_path = f'../results_ner/results_{model_name}.txt'
    output_entities(test_data, pred_labels, out_file_path)
    evaluation(data_dir, out_file_path)


def load_data(data_dir):
    dg = DatasetGenerator(split_path=data_dir, task='ner')
    return dg.get_dataset_split()


def create_indexes(train_data, max_len=100):
    index_dict = {'words': {'<PAD>': 0, '<UNK>': 1},
                  'labels': {'<PAD>': 0, 'O': 1, 'B-drug': 2, 'I-drug': 3,
                        'B-group': 4, 'I-group': 5,  'B-brand': 6, 'I-brand': 7,
                        'B-drug_n': 8, 'I-drug_n': 9},
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
        for instance in instances_of_a_sentence:
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
                #suffix_dict[suffix] = suffix_dict.get(suffix, len(suffix_dict)) + 1
                #prefix_dict[prefix] = prefix_dict.get(prefix, len(prefix_dict)) + 1
        tokenized_sentence = [w[0] for w in instances_of_a_sentence]
        tags = [analysis[1] for analysis in pos_tag(tokenized_sentence) if analysis[1].isalpha()]
        for tag in tags:
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
    word_emb = Embedding(input_dim=n_words,  output_dim=word_embedding_size, input_length=max_len, mask_zero=True)(input_words)
    pref_emb = Embedding(input_dim=n_pref, output_dim=prefix_embedding_size, input_length=max_len, mask_zero=True)(input_prefixes)
    suff_emb = Embedding(input_dim=n_suff, output_dim=suffix_embedding_size, input_length=max_len, mask_zero=True)(input_suffixes)
    pos_emb = Embedding(input_dim=n_pos, output_dim=pos_embedding_size, input_length=max_len, mask_zero=True)(input_pos)
    # concatenate embeddings and feed the biLSTM model
    model = Concatenate()([word_emb, pref_emb, suff_emb,pos_emb])
    model = Bidirectional(LSTM(units=word_embedding_size, return_sequences=True, recurrent_dropout=0.1, dropout=0.1,
                               kernel_initializer=he_normal()))(model)
    out = TimeDistributed(Dense(n_labels, activation="softmax"))(model)
    model = Model([input_words, input_prefixes, input_suffixes, input_pos], out)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

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

def build_network_with_CRF(indexes, optimizer):
    n_words = len(indexes['words'])
    n_labels = len(indexes['labels'])
    max_len = indexes['maxLen']
    word_embedding_size = max_len - int(max_len*0.1)  # max sentence len + 10%
    input = Input(shape=(max_len,))
    # mask_zero = True must de removed from the embedding
    model = Embedding(input_dim=n_words,  output_dim=word_embedding_size, input_length=max_len)(input)
    model = Bidirectional(LSTM(units=word_embedding_size, return_sequences=True, recurrent_dropout=0.1, dropout=0.1,
                               kernel_initializer=he_normal()))(model)

    model = TimeDistributed(Dense(n_labels, activation="relu"))(model)
    # 'join' (default) or 'marginal'
    crf = CRF(n_labels, learn_mode='join')
    out = crf(model)
    model = Model(input, out)
    model.compile(optimizer=optimizer, loss=crf_loss, metrics=[crf_viterbi_accuracy])
    return model


def encode_words(split_data, indexes):
    word_dict = indexes['words']
    max_len = indexes['maxLen']
    encoded_matrix = []
    for instances_of_a_sentence in split_data.values():
        #pos_tags_of_sentence = pos_tag([token[0] for token in instances_of_a_sentence])

        encoded_sentence = []
        for idx, instance in enumerate(instances_of_a_sentence):
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
    max_len = indexes['maxLen']
    encoded_labels = []
    for instances_of_a_sentence in split_data.values():
        encoded_sentence_labels = []
        for idx, instance in enumerate(instances_of_a_sentence):
            # If the sentence is bigger than max_len we need to cut the sentence
            if idx < max_len:
                label = instance[3]
                if label in label_dict:
                    encoded_sentence_labels.append(label_dict[label])
                else:  # 'O': 1 # is this correct? Fixed. Now it does not enter. TODO: delete this when finished
                    encoded_sentence_labels.append(1)
            else:
                break
        sent_len = len(encoded_sentence_labels)
        # Check if we need padding
        if max_len - sent_len > 0:
            padding = [0] * (max_len - sent_len)
            encoded_sentence_labels.extend(padding)
        encoded_labels.append(encoded_sentence_labels)
    encoded_label_matrix = np.array(encoded_labels)
    y = [to_categorical(i, num_classes=len(label_dict.values())) for i in encoded_label_matrix]
    return np.array(y)


def save_indexes(indexes, model_name):
    with open(f'../saved_models_ner/encoding_{model_name}.json', 'w') as fp:
        json.dump(indexes, fp, indent=4)


def load_model_and_indexes(model_name):
    with open(f'../saved_models_ner/encoding_{model_name}.json', 'r') as fp:
        indexes = json.load(fp)
    try:
        model = load_model(f'../saved_models_ner/mc_{model_name}.h5')
    except ValueError:
        model = load_model(f'../saved_models_ner/mc_{model_name}.h5', custom_objects={'CRF': CRF, 'crf_loss': crf_loss,
                                                           'crf_viterbi_accuracy': crf_viterbi_accuracy})
    return model, indexes


def y_pred_to_labels(pred, indexes):
    inv_map = {v: k for k, v in indexes['labels'].items()}
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(inv_map[p_i])
        out.append(out_i)
    return out


def output_entities(test_data, pred_labels, out_file_path):
    out_file = open(out_file_path, "w+")
    for sentence_index, (sid, features) in enumerate(test_data.items()):
        tags = pred_labels[sentence_index][:len(features)]  # we remove the padding
        translate_BIO_to_NE(sid, features, tags, out_file)


def translate_BIO_to_NE(sid, tokens, tags, out_file):
    merged_tokens = []
    first_type_was = None
    print(tags)
    for index, tag in enumerate(tags):
        if tag != 'O' and tag != '<PAD>':
            tag_splitted = tag.split('-')
            bio_tag = tag_splitted[0]
            entity_type = tag_splitted[1]
            first_type_was = entity_type
        else:
            bio_tag = 'O'
            entity_type = None
        if bio_tag == 'B':
            if merged_tokens:
                print(
                    f"{sid}|{merged_tokens[0][1]}-{merged_tokens[-1][2]}|{' '.join([element[0] for element in merged_tokens])}|{entity_type}", file=out_file)
                merged_tokens = []
            merged_tokens.append(tokens[index])
            first_type_was = entity_type

        elif bio_tag == 'O':
            if merged_tokens and first_type_was is not None:
                print(
                    f"{sid}|{merged_tokens[0][1]}-{merged_tokens[-1][2]}|{' '.join([element[0] for element in merged_tokens])}|{first_type_was}", file=out_file)
                merged_tokens = []

            elif merged_tokens and first_type_was is None:
                print(
                    f"{sid}|{merged_tokens[0][1]}-{merged_tokens[-1][2]}|{' '.join([element[0] for element in merged_tokens])}|{first_type_was}")
                break
        else:
            merged_tokens.append(tokens[index])


def evaluation(data_dir, out_file_path):
    evaluator.evaluate("NER", data_dir, out_file_path)
