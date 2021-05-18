import numpy as np


def learn(train_dir, val_dir, model_name):
    """
    learns a NN model using traindir as training data, and validationdir as validation data. Saves learnt model in a file named modelname
    """
    # load train and validation data in a suitable form
    train_data = load_data(train_dir)
    val_data = load_data(val_dir)
    # create indexes from training data
    max_len = 100
    indexes = create_indexes(train_data, max_len)
    # build network
    model = build_network(indexes)
    # encode datasets
    X_train = encode_words(train_data, indexes)
    y_train = encode_labels(train_data, indexes)
    X_val = encode_words(val_data, indexes)
    y_val = encode_labels(val_data, indexes)
    # train model
    # TODO model.fit(X_train, y_train, validation_data=(X_val, y_val))
    # save model and indexs, for later use in prediction
    save_model_and_indexes(model, indexes, model_name)


def predict(model_name, data_dir, out_file) :
    """Loads a NN model from file ’modelname’ and uses it to extract drugs in datadir.
    Saves results to ’outfile’ in the appropriate format.
    """
    # load model and associated encoding data
    model, idx = load_model_and_indexes(model_name)
    # load data to annotate
    test_data = load_data(data_dir)
    # encode dataset
    X = encode_words(test_data, idx)
    # tag sentences in dataset
    Y = model.predict(X)
    # get most likely tag for each word
    Y_pred = [[idx['labels'][np.argmax(y)] for y in s] for s in Y]  # extract entities and dump them to output file
    output_entities(test_data, Y_pred, out_file)
    # evaluate using official evaluator.
    evaluation(data_dir, out_file)


def load_data(data_dir):
    """
    Task:
    Load XML files in given directory , tokenize each sentence , and ground truth BIO labels for each token.
    Input:
    datadir: A directory containing XML files.
    extract
    Output:
    A dictionary containing the dataset. Dictionary key is sentence_id, and
    the value is a list of token tuples (word, start, end, ground truth).
    Example:
    load_data(’data/Train’)
    {’DDI-DrugBank.d370.s0’: [(’as’,0,1,’O’), (’differin’,3,10,’B-brand’),
    (’gel’,12,14,’O’), ..., (’with’,343,346,’O’),
    (’caution ’,348,354,’O’), (’.’,355,355,’O’)],
    ’DDI-DrugBank.d370.s1’: [(’particular’,0,9,’O’), (’caution’,11,17,’O’),
    (’should’,19,24,’O’), ...,(’differin’,130,137,’B-brand’), (’gel’,139, 141,’O’), (’.’,142,142,’O’)],
    ...}
    """
    # TODO Use XML parsing and tokenization functions from previous exercises
    pass


def create_indexes(train_data, max_len):
    """
    Task:
    Create index dictionaries both for input (words) and output (labels)
    from given dataset. Input:
    dataset: dataset produced by load_data.
    max_length: maximum lenght of a sentence (longer sentences will
    be cut, shorter ones will be padded).
    Output:
    A dictionary where each key is an index name (e.g. "words", "labels"), and the value is a dictionary mapping each word/label to a number.
    An entry with the value for maxlen is also stored
    Example:
    create_indexs(traindata)
    {’words’: {’<PAD>’:0, ’<UNK>’:1, ’11-day’:2, ’murine’:3, ’criteria’:4,
    ’stroke’:5,... ,’levodopa’:8511, ’terfenadine’: 8512} ’labels’: {’<PAD>’:0, ’B-group’:1, ’B-drug_n’:2, ’I-drug_n’:3, ’O’:4, ’I-group ’:5, ’B-drug ’:6, ’I-drug ’:7, ’B-brand ’:8, ’I-brand ’:9}
    ’maxlen’ : 100
    }
    """
    # TODO Add a ’<PAD>’:0 code to both ’words’ and ’labels’ indexes. Add an ’<UNK>’:1 code to ’words’.
    # TODO The coding of the rest of the words/labels is arbitrary.
    # TODO This indexes will be needed by the predictor to properly use the model.
    pass


def build_network(indexes):
    """
    Task: Create network for the learner.
    Input:
    idx: index dictionary with word/labels codes, plus maximum sentence
    length.
    Output:
    Returns a compiled Keras neural network with the specified layers
    """
    pass

    n_words = len(indexes['words'])
    n_labels = len(indexes['labels'])
    max_len = indexes['maxlen']

    # create network layers
    # input = Input(shape=(max_len ,))
    # ... add missing layers here ... #
    # output = TODO final output layer

    # TODO create and compile model
    # model = Model(input, output)
    # model.compile() # set appropriate parameters (optimizer, loss, etc)

    # return model
    pass


def encode_words(train_data, indexes):
    """
    Task:
    Encode the words in a sentence dataset formed by lists of tokens into lists of indexes
    suitable for NN input.
    Input:
    dataset: A dataset produced by load_data.
    idx: A dictionary produced by create_indexs, containing word and
    label indexes, as well as the maximum sentence length.
    Output:
    The dataset encoded as a list of sentence, each of them is a list of word indices. If the word is not in the index, <UNK> code is used. If the sentence is shorter than max_len it is padded with <PAD> code.
    Example:
    encode_words(traindata ,idx)
    [ [6882 1049 4911 ... [2290 7548 8069 ... ...
    [2002 6582 7518 ...
    ’’’
    0 00] 0 00]
    0 0 0]]
    """
    pass


def encode_labels(train_data, indexes):
    """
    Task:
    Encode the ground truth labels in a sentence dataset formed by lists of tokens into lists of indexes suitable for NN output.
    Input:
    dataset: A dataset produced by load_data.
    idx: A dictionary produced by create_indexs, containing word and
    label indexes, as well as the maximum sentence length.
    Output:
    The dataset encoded as a list of sentence, each of them is a list of BIO label indices. If the sentence is shorter than max_len it is padded with <PAD> code.
    Example:
    encode_labels(traindata ,idx)
    [[ [4] [6] [4] [4] [4] [4] ... [0] [0] ]
    [ [4] [4] [8] [4] [6] [4] ... [0] [0] ] ...
    [ [4] [8] [9] [4] [4] [4] ... [0] [0] ] ]
    """
    # TODO Note: The shape of the produced list may need to be adjusted depending on the architecture of your network
    # TODO and the kind of output layer you use.
    pass


def save_model_and_indexes(model, indexes, model_name):
    """
    Task: Save given model and indexs to disk Input:
    model: Keras model created by _build_network, and trained.
    idx: A dictionary produced by create_indexs, containing word and
    label indexes, as well as the maximum sentence length. filename: filename to be created
    Output:
    Saves the model into filename.nn and the indexes into filename.idx
    """
    pass


def load_model_and_indexes(model_name):
    """
    Task: Load model and associate indexs from disk Input:
    filename: filename to be loaded Output:
    Loads a model from filename.nn, and its indexes from filename.idx
    Returns the loaded model and indexes.
    """
    # TODO Note: Use Keras model.save and keras.models.load model functions to save/load the model.
    # TODO Note: Use your preferred method (pickle, plain text, etc) to save/load the index dictionary.
    pass


def output_entities(test_data, Y_pred, out_file):
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
    output\_entities(dataset , preds)
    DDI-DrugBank.d283.s4|14-35|bile acid sequestrants|group
    DDI-DrugBank.d283.s4|99-104|tricor|group
    DDI-DrugBank.d283.s5|22-33|cyclosporine|drug
    DDI-DrugBank.d283.s5|196-208|fibrate drugs|group
    """
    # TODO Note: Most of this function can be reused from NER-ML exercise.
    pass


def evaluation(data_dir, out_file):
    pass


