from src.utils.data_generator import DatasetGenerator
from keras.models import Model, Input, load_model
from keras.initializers import he_normal
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from src.utils.utility import print_sentences_len_hist
import numpy as np
import json
from eval import evaluator


def learn(train_dir, val_dir, model_name):
    train_data = load_data(train_dir)
    val_data = load_data(val_dir)

    # TODO in the next line --> calculate the max len between all sentences:
    # max_len = max(len(value) for value in train_data.values())
    # TODO in the next line --> print a useful histogram of sentences length:
    # print_sentences_len_hist(train_data.values(), show_max=75)

    indexes = create_indexes(train_data, max_len=75)

    model = build_network(indexes)

    X_train = encode_words(train_data, indexes)
    y_train = encode_labels(train_data, indexes)
    X_val = encode_words(val_data, indexes)
    y_val = encode_labels(val_data, indexes)

    # TODO note: My pc cannot allow setting steps_per_epochs and validation_steps...
    # Better focus the training of maximizing the reduction of val loss --> better generalization!
    batch_size = 64
    epochs = 8
    patience = 1
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto')
    mc = ModelCheckpoint(f'../saved_models/mc_{model_name}.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size,
              epochs=epochs,
              callbacks=[es, mc],
              verbose=1)
    save_indexes(indexes, model_name)


def predict(model_name, data_dir):
    model, indexes = load_model_and_indexes(model_name)
    test_data = load_data(data_dir)
    X_test = encode_words(test_data, indexes)
    y_pred = model.predict(X_test, verbose=1)
    # get most likely tag for each word
    pred_labels = y_pred_to_labels(y_pred, indexes)

    # TODO finish next part
    out_file_path = f'../results/results_{model_name}.txt'
    output_entities(test_data, pred_labels, out_file_path)
    evaluation(data_dir, out_file_path)


def load_data(data_dir):
    dg = DatasetGenerator(split_path=data_dir)
    return dg.get_dataset_split()


def create_indexes(train_data, max_len=100):
    index_dict = {'words': {'<PAD>': 0, '<UNK>': 1},
                  'labels': {'<PAD>': 0, 'O': 1, 'B-drug ': 2, 'I-drug ': 3,
                        'B-group': 4, 'I-group ': 5,  'B-brand ': 6, 'I-brand ': 7,
                        'B-drug_n': 8, 'I-drug_n': 9},
                  'maxLen': max_len}
    word_index = 2
    word_dict = index_dict['words']
    for instances_of_a_sentence in train_data.values():
        for instance in instances_of_a_sentence:
            word = instance[0]
            if word not in word_dict:
                word_dict[word] = word_index
                word_index += 1
    return index_dict


def build_network(indexes):
    n_words = len(indexes['words'])
    n_labels = len(indexes['labels'])
    max_len = indexes['maxLen']
    word_embedding_size = max_len + int(max_len*0.1)  # max sentence len + 10%
    input = Input(shape=(max_len,))
    model = Embedding(input_dim=n_words,  output_dim=word_embedding_size, input_length=max_len, mask_zero=True)(input)
    model = Bidirectional(LSTM(units=word_embedding_size, return_sequences=True, recurrent_dropout=0.1, dropout=0.1,
                               kernel_initializer=he_normal()))(model)
    out = TimeDistributed(Dense(n_labels, activation="softmax"))(model)
    model = Model(input, out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def encode_words(split_data, indexes):
    word_dict = indexes['words']
    max_len = indexes['maxLen']
    encoded_matrix = []
    for instances_of_a_sentence in split_data.values():
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
                else:  # 'O': 1 #TODO is this correct?
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
    with open(f'../saved_models/encoding_{model_name}.json', 'w') as fp:
        json.dump(indexes, fp, indent=4)


def load_model_and_indexes(model_name):
    with open(f'../saved_models/encoding_{model_name}.json', 'r') as fp:
        indexes = json.load(fp)
    model = load_model(f'../saved_models/mc_{model_name}.h5')
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
    """
    Task: Output detected entities in the format expected by the evaluator
    Input:
    dataset: A dataset produced by load_data.
    preds: For each sentence in dataset, a list with the labels for each
    sentence token ,
    as predicted by the model
    Output:
    prints the detected entities to stdout in the format required by the
    evaluator.
    Example:
    output_entities(dataset , preds)
    DDI-DrugBank.d283.s4|14-35|bile acid sequestrants|group
    DDI-DrugBank.d283.s4|99-104|tricor|group
    DDI-DrugBank.d283.s5|22-33|cyclosporine|drug
    DDI-DrugBank.d283.s5|196-208|fibrate drugs|group
    """
    # TODO Note: Most of this function can be reused from NER-ML exercise.
    out_file = open(out_file_path, "w+")
    for sentence_index, (sid, features) in enumerate(test_data.items()):
        tags = pred_labels[sentence_index][:len(features)]
        translate_BIO_to_NE(sid, features, tags, out_file)


def translate_BIO_to_NE(sid, tokens, tags, out_file):
    merged_tokens = []
    first_type_was = None
    for index, tag in enumerate(tags):
        if tag != 'O':
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
