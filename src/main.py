from src import ner_nn


if __name__ == '__main__':
    # ner_nn.learn(train_dir="../data/train", val_dir="../data/devel", model_name='first_model_bs=64_ep=8')
    ner_nn.predict(model_name='first_model_bs=64_ep=8', data_dir="../data/devel")
