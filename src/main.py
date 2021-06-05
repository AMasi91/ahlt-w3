from src import ner_nn, ddi_nn


if __name__ == '__main__':
    ner_nn.learn(train_dir="../data/train", val_dir="../data/devel", model_name='test_model')
    ner_nn.predict(model_name='test_model', data_dir="../data/devel")
    # ddi_nn.learn(train_dir="../data/train", val_dir="../data/devel", model_name='test_model')
