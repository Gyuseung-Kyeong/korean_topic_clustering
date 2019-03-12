import math

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from .preprocess import *


def get_score(dict_frequency_benchmark, dict_frequency_test, score_type):
    """
    get_nouns_frequency 함수로 처리된 결과 dict내의 두 value 선택하면 단어별 빈도 비율의 차이를 0~1점 사이의 score로 반환한다.
    1점에 가까울수록 유사하다.
    :param dict_frequency_benchmark: dict. get_nouns_frequency 함수로 처리된 결과물의 value.
    :param dict_frequency_test: dict. get_nouns_frequency 함수로 처리된 결과물의 value.
    :param score_type: string. choose 'sse' or 'log_sse'
    :return: float
    """

    df_benchmark = pd.DataFrame.from_dict(dict_frequency_benchmark,
                                          orient='index', columns=['word_count'])
    df_test = pd.DataFrame.from_dict(dict_frequency_test,
                                     orient='index', columns=['word_count'])

    # 1회 나온 단어 log 씌우면 0되는 것 방지
    df_benchmark['log_word_count'] = np.log(df_benchmark.word_count + 1)
    df_test['log_word_count'] = np.log(df_test.word_count + 1)

    df_benchmark['word_portion'] = df_benchmark.word_count.apply(
        lambda x: x / df_benchmark.word_count.sum())
    df_test['word_portion'] = df_test.word_count.apply(
        lambda x: x / df_test.word_count.sum())

    df_benchmark['log_word_portion'] = df_benchmark.log_word_count.apply(
        lambda x: x / df_benchmark.log_word_count.sum())
    df_test['log_word_portion'] = df_test.log_word_count.apply(
        lambda x: x / df_test.log_word_count.sum())

    df_merge = pd.merge(df_benchmark, df_test, left_index=True, right_index=True, how='outer',
                        suffixes=('_benchmark', '_test'))
    df_merge.fillna(0.0, inplace=True)
    df_merge['sse'] = (df_merge.word_portion_benchmark - df_merge.word_portion_test) ** 2
    df_merge['log_sse'] = (df_merge.log_word_portion_benchmark - df_merge.log_word_portion_test) ** 2

    if score_type == 'sse':
        return 1 - math.sqrt(df_merge.sse.sum())
    elif score_type == 'log_sse':
        return 1 - math.sqrt(df_merge.log_sse.sum())


def get_lda_model(dict_merged_strings, decided_ld, decided_n_components, delete_words,
                  category='benchmark'):
    """
    dict_merged_strings[category]의 lda model object와 feature 단어 list 반환.
    :param dict_merged_strings: dict. get_dict_merged_strings 함수로 처리된 결과물.
    :param decided_ld: float. learning decay.
    :param decided_n_components: int. lda topic 수
    :param delete_words: list. 삭제할 key 정보를 가진 list
    :param category: string. dict_merged_strings안의 key e.g. 'benchmark', 'personal'
    :return: dict
    """
    documents = []
    for sentence in [s for s in dict_merged_strings[category].split("\n") if s]:
        documents.append(' '.join([s for s in komoran.nouns(sentence) if s]))

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=delete_words)
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()

    # Run LDA
    lda_model = LatentDirichletAllocation(learning_decay=decided_ld,
                                          n_components=decided_n_components,
                                          max_iter=5, learning_method='online',
                                          learning_offset=50., random_state=0,
                                          doc_topic_prior=0.1, topic_word_prior=0.01,
                                          ).fit(tf)

    return lda_model, tf_feature_names


def display_topics(model, feature_names, no_top_words):
    """
    토픽별 단어별 비중을 내림차순 정렬해서 no_top_words 만큼 보여준다.
    reference: https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
    :param model: lda model object.
    :param feature_names: array-like.
    :param no_top_words: int.
    :return: None
    """
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {}:".format(topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[::-1]][:no_top_words]))


def get_lda_topic_clustering(lda_model, tf_feature_names):
    """
    topic clustering 결과를 반환한다.
    :param lda_model: lda model object.
    :param tf_feature_names: array-like.
    :return: DataFrame
    """
    df_topic_clustering = pd.DataFrame()

    for topic_idx, topic in enumerate(lda_model.components_):
        cut_off_len = len([i for i in topic if i >= 1])
        df_topic_clustering = \
            df_topic_clustering.append([[topic_idx,
                                         [tf_feature_names[i] for i in topic.argsort()[::-1]][:cut_off_len]]]
                                       , ignore_index=True)
    df_topic_clustering.columns = ['topic_index', 'topic_words']
    df_topic_clustering.set_index('topic_index', inplace=True)

    return df_topic_clustering


def get_topic_score(dict_frequency_benchmark, dict_frequency_test, df_topic_clustering,
                    topic_index, score_type='sse'):
    """
    get_nouns_frequency 함수로 처리된 결과 dict내의 두 value 선택하면
    df_topic_clustering의 topic index와 연관된 주제의 단어만을 남긴다.
    단어별 빈도 비율의 차이를 0~1점 사이의 score로 반환한다.
    1점에 가까울수록 유사하다.
    :param dict_frequency_benchmark: dict. get_nouns_frequency 함수로 처리된 결과물.
    :param dict_frequency_test: dict. get_nouns_frequency 함수로 처리된 결과물.
    :param df_topic_clustering: DataFrame. get_lda_topic_clustering 함수의 결과물.
    :param topic_index: int
    :param score_type: string. choose 'sse' or 'log_sse'.
    :return: float
    """
    dict_frequency_benchmark_topic = {topic_key: dict_frequency_benchmark[topic_key]
                                      for topic_key in df_topic_clustering.loc[topic_index, 'topic_words']
                                      if topic_key in dict_frequency_benchmark.keys()}
    dict_frequency_test_topic = {topic_key: dict_frequency_test[topic_key]
                                 for topic_key in df_topic_clustering.loc[topic_index, 'topic_words']
                                 if topic_key in dict_frequency_test.keys()}

    df_benchmark = pd.DataFrame.from_dict(dict_frequency_benchmark_topic,
                                          orient='index', columns=['word_count'])
    df_test = pd.DataFrame.from_dict(dict_frequency_test_topic,
                                     orient='index', columns=['word_count'])

    # 1회 나온 단어 log 씌우면 0되는 것 방지
    df_benchmark['log_word_count'] = np.log(df_benchmark.word_count + 1)
    df_test['log_word_count'] = np.log(df_test.word_count + 1)

    df_benchmark['word_portion'] = df_benchmark.word_count.apply(
        lambda x: x / df_benchmark.word_count.sum())
    df_test['word_portion'] = df_test.word_count.apply(
        lambda x: x / df_test.word_count.sum())

    df_benchmark['log_word_portion'] = df_benchmark.log_word_count.apply(
        lambda x: x / df_benchmark.log_word_count.sum())
    df_test['log_word_portion'] = df_test.log_word_count.apply(
        lambda x: x / df_test.log_word_count.sum())

    df_merge = pd.merge(df_benchmark, df_test, left_index=True, right_index=True, how='outer',
                        suffixes=('_benchmark', '_test'))
    df_merge.fillna(0.0, inplace=True)
    df_merge['sse'] = (df_merge.word_portion_benchmark - df_merge.word_portion_test) ** 2
    df_merge['log_sse'] = (df_merge.log_word_portion_benchmark - df_merge.log_word_portion_test) ** 2

    if score_type == 'sse':
        return 1 - math.sqrt(df_merge.sse.sum())
    elif score_type == 'log_sse':
        return 1 - math.sqrt(df_merge.log_sse.sum())


def get_topic_scores(dict_frequency_benchmark, dict_frequency_test, df_topic_clustering, score_type='sse'):
    """
    get_nouns_frequency 함수로 처리된 결과 dict내의 두 value 선택하면
    df_topic_clustering의 topic_index 별로 0~1점 사이의 score로 반환한다.
    1점에 가까울수록 유사하다.
    :param dict_frequency_benchmark: dict. get_nouns_frequency 함수로 처리된 결과물.
    :param dict_frequency_test: dict. get_nouns_frequency 함수로 처리된 결과물.
    :param df_topic_clustering: DataFrame. get_lda_topic_clustering 함수의 결과물.
    :param score_type: string. choose 'sse' or 'log_sse'.
    :return: float
    """
    list_result = []
    for topic_index in df_topic_clustering.index:
        list_result.append(
            [topic_index,
             get_topic_score(
                 dict_frequency_benchmark, dict_frequency_test, df_topic_clustering, topic_index, score_type)])
    df_result = pd.DataFrame(list_result, columns=['topic_index', 'topic_similarity_score'])
    df_result.set_index('topic_index', inplace=True)

    return df_result
