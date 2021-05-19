from src.nerc_nn import learn

if __name__ == '__main__':
    learn(train_dir="../data/train", val_dir="../data/devel", model_name=None)