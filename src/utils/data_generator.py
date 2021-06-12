from os import listdir
from xml.dom.minidom import parse
import os.path
import ast
from src.utils import utility as util
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import nltk
nltk.download('wordnet')


def generate_output_path(path: str, task) -> str:
    out_file_path = f"../data/processed_{task}/"
    if "train" in path:
        out_file_path += "train"
    elif "devel" in path:
        out_file_path += "val"
    elif "test" in path:
        out_file_path += "test"
    out_file_path += ".txt"
    return out_file_path


class DatasetGenerator:
    def __init__(self, split_path, task):
        self.split_path = split_path
        self.task = task
        self.out_path_dict = generate_output_path(self.split_path, task)
        self.mask = {0: '<DRUG_OTHER>', 1: '<DRUG1>', 2: '<DRUG2>'}

    def get_dataset_split(self) -> dict:
        if self.task == 'ner':
            return self._read_file_ner()
        else:
            return self._read_file_ddi()

    def _read_file_ner(self) -> dict:
        out_file_path = self.out_path_dict
        if os.path.isfile(out_file_path):
            print("File found. Reading...")
            with open(out_file_path, "r") as file:
                dataset = {}
                for line in file:
                    content = line.splitlines()[0]
                    if content != "":
                        sid, word, s_offset, e_offset, tag = content.split('\t')
                        if sid not in dataset:
                            dataset[sid] = []
                        dataset[sid].append((word, s_offset, e_offset, tag))
            return dataset
        else:
            print("File not found.")
            if self.task == 'ner':
                self._generate_dataset_ner()
            else:
                self._generate_dataset_ddi()
    
    def _generate_dataset_ner(self) -> dict:
        print("Generating dataset...")
        data_dir = self.split_path
        dataset = {}
        # process each file in directory
        with open(self.out_path_dict, "w+") as file:
            for f in listdir(data_dir):
                # parse XML file, obtaining a DOM tree
                tree = parse(data_dir + "/" + f)
                # process each sentence in the file
                sentences = tree.getElementsByTagName("sentence")
                for s in sentences:
                    sid = s.attributes["id"].value  # get sentence id
                    sentence_text = s.attributes["text"].value  # get sentence text
                    # load ground truth entities.
                    gold = []
                    entities = s.getElementsByTagName("entity")
                    for e in entities:
                        # for discontinuous entities, we only get the first span
                        offset = e.attributes["charOffset"].value
                        (start, end) = offset.split(";")[0].split("-")
                        gold.append((int(start), int(end), e.attributes["type"].value))
                    # tokenize text
                    tokens = util.tokenize(sentence_text)
                    sentence_instances = []
                    for i in range(0, len(tokens)):
                        # see if the token is part of an entity, and which part (B/I)
                        tag = self._get_tag(tokens[i], gold)
                        word = tokens[i][0]
                        s_offset = str(tokens[i][1])
                        e_offset = str(tokens[i][2])
                        format = word + "\t" + s_offset + "\t" + e_offset + "\t" + tag
                        print(sid + '\t' + format, file=file)
                        sentence_instances.append(format)
                    dataset[sid] = sentence_instances
        return self._read_file_ner()

    def _get_tag(self, param, gold_list):
        tag = "O"
        tag_dict = {0: "B", 1: "I"}
        try:
            assert len(param) == 3
        except AssertionError:
            print("error. The input token is not properly formatted. ")

        start, stop = param[1], param[2]
        for gold_tag in gold_list:
            if gold_tag[0] <= start != stop <= gold_tag[1]:
                tag = tag_dict.get(abs(start - gold_tag[0]), "I") + "-" + gold_tag[2]
                return tag
        return tag

    def _generate_dataset_ddi(self):
        print("Generating dataset...")
        data_dir = self.split_path
        dataset = {}
        with open(self.out_path_dict, "w+") as file:
            index = 0
            list_dir = listdir(data_dir)
            number_of_files = len(list_dir)
            # process each file in directory
            for f in list_dir:
                # parse XML file, obtaining a DOM tree
                tree = parse(data_dir + "/" + f)
                # process each sentence in the file
                sentences = tree.getElementsByTagName("sentence")
                for s in sentences:
                    sid = s.attributes["id"].value  # get sentence id
                    # get sentence text # load sentence ground truth entities
                    sentence_text = s.attributes["text"].value
                    entities = {}
                    ents = s.getElementsByTagName("entity")
                    for e in ents:
                        id = e.attributes["id"].value
                        entity_text = e.attributes["text"].value
                        # analyze sentence if there is at least a pair of entities
                        entities[id] = (entity_text, e.attributes["charOffset"].value.split("-"))
                    if len(entities) > 1:
                        tokens = util.tokenize(sentence_text)
                        tokens_plus_info = self._add_pos_and_lemmas(tokens)
                    # for each pair of entities, decide whether it is DDI and its type
                    pairs = s.getElementsByTagName("pair")
                    for p in pairs:
                        # get ground truth
                        ddi = p.attributes["ddi"].value
                        # target entities
                        dditype = p.attributes["type"].value if ddi == "true" else "null"
                        id_e1 = p.attributes["e1"].value
                        id_e2 = p.attributes["e2"].value
                        features = self.prepare_sentence_features(id_e1, id_e2, entities, tokens_plus_info)
                        dataset[sid] = [id_e1, id_e2, dditype, features]
                        print(sid + '\t' + id_e1 + '\t' + id_e2 + '\t' + dditype + '\t' + str(features), file=file)
                index += 1
                print("{:.1%}".format(index / number_of_files))
        return dataset

    def prepare_sentence_features(self, id_e1, id_e2, entities, tokens_plus_info) -> list:
        interacting_entities = [entities[id_e1], entities[id_e2]]
        entity_1 = self.extract_offsets(entities[id_e1])
        entity_2 = self.extract_offsets(entities[id_e2])
        non_interacting_entities = [self.extract_offsets(entity) for entity in entities.values()
                                    if entity not in interacting_entities]

        sentence_features = []
        for token_info in tokens_plus_info:
            token_start = str(token_info[1])
            if token_start == entity_1[0]:
                token_info = (self.mask[1], token_info[1], token_info[2], self.mask[1], self.mask[1])

            elif token_start == entity_2[0]:
                token_info = (self.mask[2], token_info[1], token_info[2], self.mask[2], self.mask[2])

            elif any(ent[0] == token_start for ent in non_interacting_entities):
                token_info = (self.mask[0], token_info[1], token_info[2], self.mask[0], self.mask[0])
            sentence_features.append(token_info)
        return sentence_features

    def extract_offsets(self, entity):
        entity_offset = entity[1]
        if len(entity_offset) > 2:
            offsets = ''
            # ['29', '56;77', '100']
            for e in entity_offset:
                offsets += e + ";"  # 29;56;77;100
            offsets_list = offsets.split(";")[:-1]  # ['29', '56', '77', '100']
        else:
            offsets_list = entity_offset
        return offsets_list

    def _add_pos_and_lemmas(self, tokens):
        just_tokens = [t for t, start, end in tokens]
        token_pos_tag_pairs = pos_tag(just_tokens)
        sentence_lemmas = [self._extract_lemma(pair) for pair in token_pos_tag_pairs]
        result = []
        for index in range(len(tokens)):
            word, start, end = tokens[index]
            pos = token_pos_tag_pairs[index][1]
            lemma = sentence_lemmas[index]
            result.append((word, start, end, pos, lemma))
        return result

    def _extract_lemma(self, word_and_tag):
        wnl = WordNetLemmatizer()
        tag = word_and_tag[1][:2]
        if tag == 'NN':
            return wnl.lemmatize(word_and_tag[0], wn.NOUN)
        elif tag == 'VB':
            return wnl.lemmatize(word_and_tag[0], wn.VERB)
        elif tag == 'JJ':
            return wnl.lemmatize(word_and_tag[0], wn.ADJ)
        elif tag == 'RB':
            return wnl.lemmatize(word_and_tag[0], wn.ADV)
        return word_and_tag[0]

    def _read_file_ddi(self) -> dict:
        out_file_path = self.out_path_dict
        if os.path.isfile(out_file_path):
            print("File found. Reading...")
            with open(out_file_path, "r") as file:
                dataset = {}
                for line in file:
                    content = line.splitlines()[0]
                    if content != "":
                        sid, id_e1, id_e2, tag, tokenized_sentence = content.split('\t')
                        if sid not in dataset:
                            dataset[sid] = [[id_e1, id_e2, tag, ast.literal_eval(tokenized_sentence)]]
                        else:
                            dataset[sid].append([id_e1, id_e2, tag, ast.literal_eval(tokenized_sentence)])
            return dataset
        else:
            print("File not found.")
            if self.task == 'ner':
                return self._generate_dataset_ner()
            else:
                return self._generate_dataset_ddi()
