"""
处理数据
"""
from os.path import join
from codecs import open


def build_corpus(split, make_vocab=True, data_dir="./ResumeNER"):
    """读取数据"""
    dirty_word = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    assert split in ['train1', 'dev1', 'test1', 'train2', 'dev2', 'test2', 'train3', 'dev3', 'test3']
    word_lists = []
    tag_lists = []
    with open(join(data_dir, split+".char.bmes"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []

        for line in f:
            # 分开原数据一行 为字符和标签
            if line != '\r\n' and line != ' ' and line != '' and line != '\t' and line != '\r' and line != '\n':
                # 去除脏数据
                if line[0] == ' ' or line[0] == '　' or line[0] == '0' or line[0] in dirty_word:
                    continue
                else:
                    word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            # 添加一句字符和一句标签
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists


def build_map(lists):
    """得到字符和标签编码的字典"""
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps
