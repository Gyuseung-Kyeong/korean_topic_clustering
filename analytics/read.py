import re

from collections import OrderedDict
from operator import itemgetter

import matplotlib.font_manager as fm
from konlpy.tag import Komoran
from matplotlib import pyplot as plt
from nltk import Text

# font_path = '/Users/gs/Library/Fonts/NanumBarunGothic.otf'
# original
# plt.rc('font', family=fm.FontProperties(fname=font_path, size=50).get_name())
# plt.rc('font', family='NanumBarunGothicOTF')


# case1
# Create a font properties object
# font = fm.FontProperties(fname=font_path)
# Use the font in your plot
# plt.rc('font', family=font.get_name())

# case2
# font_family = 'NanumBarunGothicOTF'
# plt.rc('font', family=font_family)

# case3
# Check if the font is already in the font cache
# if font_path not in fm.findSystemFonts():
#     # If it's not, add it to the cache
#     fm.fontManager.addfont(font_path)
#     fm._rebuild()

# Verify that the font is now available
# print(fm.findfont('NanumBarunGothicOTF'))
# plt.rc('font', family='NanumBarunGothicOTF')

komoran = Komoran()


def get_dict_string(dict_target, target_key):
    """
    text 하위 폴더(폴더명: target_key)의 .txt 파일명을 key, .txt를 읽은 string을 value로 반환하는 dict 생산.
    :param dict_target: dict
    :param target_key: string
    :return: dict
    """
    dict_topic = dict()
    for target_file in dict_target[target_key]:
        dict_topic[target_file] = open('text/{0}/{1}.txt'.format(target_key, target_file),
                                       "r", encoding="utf-8").read()

    return dict_topic


def get_dict_strings(dict_target):
    """
    text 하위폴더별로 .txt 파일명을 key, .txt를 읽은 string을 value로 반환하는 dict 생산.
    :param dict_target: dict
    :return: dict
    """
    dict_strings = dict()
    for target_key in dict_target.keys():
        dict_strings[target_key] = get_dict_string(dict_target, target_key)

    return dict_strings


def get_dict_merged_strings(dict_strings):
    """
    text 하위폴더명을 key, text 하위폴더의 모든 .txt 읽은 string을 value로 반환하는 dict 생산.
    :param dict_target: dict
    :return: dict
    """
    dict_merged_strings = dict()

    for key_topic in dict_strings.keys():
        for key_file in dict_strings[key_topic].keys():
            if key_topic in dict_merged_strings.keys():
                dict_merged_strings[key_topic] += dict_strings[key_topic][key_file]
            else:
                dict_merged_strings[key_topic] = dict_strings[key_topic][key_file]

    return dict_merged_strings


def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def get_nouns_from_topics(dict_merged_strings, komoran):
    """
    dict_merged_strings의 key를 key, dict_merged_strings의 value를 noun만 뽑은 list를 value로 반환하는 dict 생산.
    :param dict_merged_strings: dict. get_dict_merged_strings 함수로 처리된 결과물.
    :param komoran: konlpy object.
    :return: dict
    """
    dict_nouns = dict()
    for key in dict_merged_strings.keys():
        print(key)
        dict_nouns[key] = komoran.nouns(" ".join(
            [s for s in remove_emojis(dict_merged_strings[key].replace('\u200b', '')).split("\n") if s]))

    return dict_nouns


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
