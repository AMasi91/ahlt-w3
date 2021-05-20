from nltk.tokenize import word_tokenize
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
