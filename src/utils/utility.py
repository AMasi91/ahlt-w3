from nltk.tokenize import word_tokenize
import numpy as np
import matplotlib.pyplot as plt


def tokenize(sentence_text):
    word_list = word_tokenize(sentence_text)
    index = 0
    tokenized_list = []
    for word in word_list:
        start_pos = sentence_text.find(word, index)
        end_pos = start_pos + len(word)-1
        tokenized_list.append((word, start_pos, end_pos))
        index += len(word)
    return tokenized_list


def print_sentences_len_hist(split, show_max=None):
    if show_max is not None:
        pre_max = [len(sen) for sen in split if len(sen) <= show_max]
        post_max = [len(sen) for sen in split if len(sen) > show_max]
        plt.hist(pre_max, bins=50, label=f'≤{show_max}: {len(pre_max)}')
        plt.hist(post_max, bins=50, color='r', label=f'>{show_max}: {len(post_max)}')
        plt.legend(title="Covered sentences:", loc='best')
    else:
        plt.hist([len(sen) for sen in split], bins=50)
    plt.title('Histogram of sentences length')
    plt.xlabel('N° words')
    plt.ylabel('Frequency')
    plt.show()


# Plot the training and validation loss + accuracy
def plot_training(history, model_name, task):
    epochs = range(1, len(history.history['acc']) + 1)
    min_index = np.argmin(history.history['val_loss'])
    min_value = history.history['val_loss'][min_index]
    correspondent_val_acc = history.history['val_acc'][min_index]

    # Accuracy plot
    plt.plot(epochs, history.history['acc'], '-o', markersize=6)
    plt.plot(epochs, history.history['val_acc'], '-o', markersize=6)
    plt.plot(min_index+1, correspondent_val_acc, 'rx', label=f'{correspondent_val_acc}', markersize=12)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='best')
    plt.title('Training and validation accuracy')
    plt.savefig(f'../saved_models_{task}/plots/{model_name}_accuracy.png')
    plt.close()

    # Loss plot
    plt.plot(epochs, history.history['loss'], '-o', markersize=6)
    plt.plot(epochs, history.history['val_loss'], '-o', markersize=6)
    plt.plot(min_index+1, min_value, 'rx', label=f'{min_value}', markersize=12)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='best')
    plt.title('Training and validation loss')
    plt.savefig(f'../saved_models_{task}/plots/{model_name}_loss.png')