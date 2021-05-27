from src import ner_nn, ddi_nn


if __name__ == '__main__':
    ner_nn.learn(train_dir="../data/train", val_dir="../data/devel", model_name='test')
    ner_nn.predict(model_name='test', data_dir="../data/devel")
