# import re
import random
import pandas as pd  # type:ignore
from typing import List

# from abc import ABC, abstractmethod
from functools import cache, partial
import spacy  # type:ignore
from .utils import PreprocessLayer


@cache
def load_nlp():
    return spacy.load("en_core_web_sm")


def select_random_keyword(text: str) -> str | None:
    nlp = load_nlp()
    text = text.strip()
    list_items = nlp(text)
    list_items_without_stops_words = [item for item in list_items if not item.is_stop]
    if list_items_without_stops_words:
        return random.choice(list_items_without_stops_words)
    else:
        return None


def select_hash_tag_word(text: str) -> str | None:
    list_hash_tag = []
    list_words = text.strip()
    for word in list_words:
        if word.startswith("#"):
            word = word.replace("#", "")
            if len(word) == 0:
                continue
            list_hash_tag.append(word)
    nlp = load_nlp()
    list_without_keyword = [
        elem for elem in list_hash_tag if not (nlp(elem)[0]).is_stop
    ]
    if list_without_keyword:
        return random.choice(list_without_keyword)
    return None


def get_list_location_for_target(df: pd.DataFrame, target: int) -> List[str]:
    df = df[~df["location"].isna()]
    target_dataframe = df.loc[df.target == target]
    target_location = target_dataframe.location.unique()
    target_list = target_location.tolist()
    target_list = list(map(str, target_list))
    target_list.sort()
    return target_list


def get_list_keyword_for_target(df: pd.DataFrame, target: int) -> List[str]:
    df = df[~df["keyword"].isna()]
    target_dataframe = df.loc[df.target == target]
    target_keyword = target_dataframe.keyword.unique()
    target_list = target_keyword.tolist()
    target_list = list(map(str, target_list))
    target_list.sort()
    return target_list


def get_list_keyword_series(series: pd.Series) -> List[str]:
    target_series = series.dropna()
    target_keyword = target_series.unique()
    target_list = target_keyword.tolist()
    target_list = list(map(str, target_list))
    target_list.sort()
    return target_list


def get_list_keyword(df: pd.DataFrame) -> List[str]:
    df = df[~df["keyword"].isna()]
    target_list = df.keyword.unique()
    target_list = target_list.tolist()
    target_keyword = list(map(str, target_list))
    target_keyword.sort()
    return target_keyword


def _sample_if_nan(value: str, list_of_words: List[str]) -> str:
    if pd.isna(value):
        return random.choice(list_of_words)
    else:
        return value


def populate_nan_values(series: pd.Series, list_of_words: List[str]) -> pd.Series:
    return series.map(partial(_sample_if_nan, list_of_words=list_of_words))


def _preprocess_keyword_row(row: pd.Series, list_of_words: List[str]) -> str:
    keyword = row.keyword
    text = row.text
    if not pd.isna(keyword):
        return keyword
    else:
        # search keyword in text
        for word in text.split():
            if word in list_of_words:
                return word
        # find hashtag
        word_hash_tag = select_hash_tag_word(text)
        if word_hash_tag:
            return word_hash_tag
        # Random word from text
        random_word = select_random_keyword(text)
        if random_word:
            return random_word
        random_word = random.choice(list_of_words)
        return random_word


class PreprocessKeywords(PreprocessLayer):
    def __init__(self):
        pass

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        list_keyword = get_list_keyword_series(df["keyword"])
        df["keyword_prep"] = df.apply(
            partial(_preprocess_keyword_row, list_of_words=list_keyword), axis=1
        )
        return df
