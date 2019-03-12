from .read import *


def get_nouns_frequency(dict_nouns):
    """
    dict_nouns key를 key, dict_nouns value를 noun별 빈도수 dict를 value로 반환하는 dict 생산.
    :param dict_nouns: dict. get_nouns_from_topics 함수로 처리된 결과물.
    :return: dict
    """
    dict_frequency = dict()
    for category in dict_nouns.keys():
        text_vocab = Text(dict_nouns[category], name=category).vocab()
        dict_frequency[category] = OrderedDict(sorted(text_vocab.items(), key=itemgetter(1), reverse=True))

    return dict_frequency


def get_preprocessed_nouns(dict_nouns, delete_words):
    """
    dict_nouns key를 key, dict_nouns value를 value로 반환하는 dict 생산.
    단, delete_words 에 없는 단어만 value에 포함함.
    :param dict_nouns: dict. get_nouns_from_topics 함수로 처리된 결과물.
    :param delete_words: list. 삭제할 key 정보를 가진 list
    :return: dict
    """
    dict_nouns_ppd = dict_nouns.copy()

    # delete_words에 있는 단어 제거
    for category in dict_nouns_ppd.keys():
        for word in delete_words:
            try:
                dict_nouns_ppd[category] = list(filter(lambda x: x != word, dict_nouns_ppd[category]))
            except KeyError:
                continue

    return dict_nouns_ppd


def get_preprocessed_nouns_frequency(dict_frequency, delete_words):
    """
    dict_frequency key를 key, dict_frequency value를 value로 반환하는 dict 생산.
    단, delete_words 에 없는 단어만 value의 key에 포함함.
    :param dict_frequency: dict. get_nouns_frequency 함수로 처리된 결과물.
    :param delete_words: list. 삭제할 key 정보를 가진 list
    :return: dict
    """
    dict_frequency_ppd = dict_frequency.copy()

    # delete_words에 있는 단어 제거
    for category in dict_frequency_ppd.keys():
        for word in delete_words:
            try:
                dict_frequency_ppd[category].pop(word)
            except KeyError:
                continue

    return dict_frequency_ppd
