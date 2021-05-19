from os import listdir
from xml.dom.minidom import parse
import os.path
from src.utils import utility as util


def generate_output_path(path: str) -> str:
    out_file_path = "../data/processed/"
    if "train" in path:
        out_file_path += "train"
    elif "devel" in path:
        out_file_path += "val"
    elif "test" in path:
        out_file_path += "test"
    out_file_path += ".txt"
    return out_file_path


class DatasetGenerator:
    def __init__(self, split_path):
        self.split_path = split_path
        self.out_path_dict = generate_output_path(self.split_path)

    def get_dataset_split(self) -> dict:
        return self._read_file()

    def _read_file(self) -> dict:
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
            self._generate_dataset()

    def _generate_dataset(self) -> dict:
        print("Generating dataset...")
        data_dir = self.split_path
        list_dir = listdir(data_dir)
        list_dir.sort()
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
        return self._read_file()

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