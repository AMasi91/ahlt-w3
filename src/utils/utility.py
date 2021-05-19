from nltk.tokenize import word_tokenize


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