import numpy as np
import pandas as pd

import json
import re

import category_encoders as ce

import impute


class _Encoder:
    def __init__(
        self,
        df: pd.DataFrame
    ) -> None:
        self.df = df



    """ label encoding"""
    def _cat_encoder(self) -> pd.DataFrame:
        object_cols = [
            "channelId", "channelTitle", "categoryId", "comments_disabled", "ratings_disabled"]
            #"isJa_title", "isJa_tags", "isJa_description", "onEn_tags", "onEn_description", "music_title",
            #"music_tags", "music_description", "isOff", "isOffChannell", "isOffJa", "isOffChannellJa",
            #"cm_title", "cm_tags", "cm_description""""
        
        
        ce_oe = ce.OrdinalEncoder(cols=object_cols,handle_unknown='impute')
        df = ce_oe.fit_transform(self.df)

        for obj_col in object_cols:
            df[obj_col] = df[obj_col].astype("category")

        return df


    """ one hot encoding"""
    def _onehot_encoder(self, columns: list) -> pd.DataFrame:
        df = pd.get_dummies(self.df[columns], drop_first=True, dummy_na=False)
    
        return df

    
    """ Adjust the number of label types"""
    def relabeler(
        self,
        column: str,
        th: int = 100,
        comma_sep: bool = False
        ) -> pd.DataFrame:
        df = self.df.copy()
        category_dict = df[column].value_counts().to_dict()
        if comma_sep:
            misc_list = [key for key, value in category_dict.items() if len(key.split("、")) == 2 or value < th]
        else:
            misc_list = [key for key, value in category_dict.items() if value < th]
        df[column] = df[column].mask(df[column].isin(misc_list), "misc")

        return df

def is_japanese(string):
    for ch in string:
        try:
            name = unicodedata.name(ch) 
            if "CJK UNIFIED" in name \
            or "HIRAGANA" in name \
            or "KATAKANA" in name:
                return True
        except:
            continue
    return False
  
class Preprocessor(_Encoder):
    def __init__(self, df: pd.DataFrame):
        super(Preprocessor, self).__init__(df)
        
    def to_onehot(self) -> pd.DataFrame:
        """Convert a pandas.DataFrame element to a one-hot vector
        """
        df = self.df.copy()
        cols = ["channelId", "channelTitle"]
        tmp = self._onehot_encoder(cols)
        df = pd.concat([df, tmp], axis=1)
        # for idempotent
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    def to_datetime(self) -> pd.DataFrame:
        df = self.df.copy()
        """ publishedAt"""
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], utc=True)
        df["year"] = df["publishedAt"].dt.year
        df["month"] = df["publishedAt"].dt.month
        df["day"] = df["publishedAt"].dt.day
        df["hour"] = df["publishedAt"].dt.hour
        df["minute"] = df["publishedAt"].dt.minute
        df["dayofweek"] = df["publishedAt"].apply(lambda x: x.dayofweek)

        """ collection_date"""
        df["collection_date"] = "20" + df["collection_date"]
        df["collection_date"] = pd.to_datetime(df["collection_date"], format="%Y.%d.%m", utc=True)
        df["c_year"] = df["collection_date"].dt.year
        df["c_month"] = df["collection_date"].dt.month
        df["c_day"] = df["collection_date"].dt.day

        """ delta"""
        df["delta"] = (df["collection_date"] - df["publishedAt"]).apply(lambda x: x.days)
        df["published_delta"] = (df["publishedAt"] - df["publishedAt"].min()).apply(lambda x: x.days)
        df["collection_delta"] = (df["collection_date"] - df["collection_date"].min()).apply(lambda x: x.days)

        return df

    
    def tags_to_num(self) -> pd.DataFrame:
        df = self.df.copy()
        tagdic = dict(
            pd.Series("|".join(list(df["tags"])).split("|")).value_counts().sort_values()
            )
        df["num_tags"] = df["tags"].astype(str).apply(lambda x: len(x.split("|")))
        df["length_tags"] = df["tags"].astype(str).apply(lambda x: len(x))
        df["tags_point"] = df["tags"].apply(
            lambda tags: sum([tagdic[tag] for tag in tags.split("|")]))
        df["count_en_tag"] = df["tags"].apply(
            lambda x: sum([bool(re.search(r'[a-zA-Z0-9]', x_)) for x_ in x.split("|")]))
        df["count_ja_tag"] = df["tags"].apply(
            lambda x: sum([is_japanese(x_) for x_ in x.split("|")]))

        return df


    def change_to_Date(self) -> pd.DataFrame:
        df = self.df.copy()
        df["collection_date"] = df["collection_date"].map(lambda x: x.split('.'))
        df["collection_date"] = df["collection_date"].map(lambda x: '20'+x[0]+'-'+x[2]+'-'+x[1]+'T00:00:00.000Z')

        df['publishedAt'] =  pd.to_datetime(df['publishedAt']).map(pd.Timestamp.to_julian_date)
        df['collection_date'] =  pd.to_datetime(df['collection_date']).map(pd.Timestamp.to_julian_date)

        return df


    def tags_to_col(self) -> pd.DataFrame:
        df = self.df.copy()
        source_list0 = []
        for i in range(len(df["tags"])):
            source_list0 = source_list0 + [x for x in df["tags"].iloc[i].split("|") if x != ""]
        source_list1 = list(set(source_list0))
        tags_df = pd.DataFrame()
        tags_df["tag"] = source_list1
        tags_df["count"] = tags_df["tag"].map(lambda x: source_list0.count(x))
        tags_df = tags_df.sort_values("count", ascending=False)[1:30]
        tag_list = list(tags_df["tag"])
        for tag in tag_list:
            df[tag] = df["tags"].map(lambda x: 1 if tag in x else 0)
        
        return df

    
    def min_max(self) -> pd.DataFrame:
        df = self.df.copy()
        num_list = [
        "likes", "dislikes", "comment_count"
        ]
        for num in num_list:
            min_value = df[num].min()
            max_value = df[num].max()
            result = (df[num] - min_value)/(max_value - min_value)
            df[num] = result
        return df

    def create_features(self) -> pd.DataFrame:
        df = self.df.copy()
        df["like_dislike_ratio"] = df["likes"]/(df["dislikes"]+1)
        df["comments_like_ratio"] = df["comment_count"]/(df["likes"]+1)
        df["comments_dislike_ratio"] = df["comment_count"]/(df["dislikes"]+1)
        df["likes_com"] = df["likes"] * df["comments_disabled"]
        df["dislikes_com"] = df["dislikes"] * df["comments_disabled"]
        df["comments_likes"] = df["comment_count"] * df["ratings_disabled"]
        df["ishttp_in_dis"] = df["description"].apply(lambda x: x.lower().count("http"))
        df["len_description"] = df["description"].apply(lambda x: len(x))
        df["len_title"] = df["title"].apply(lambda x: len(x))
        df["isJa_title"] = df["title"].apply(lambda x: is_japanese(x))
        df["isJa_tags"] = df["tags"].apply(lambda x: is_japanese(x))
        df["isJa_description"] = df["description"].apply(lambda x: is_japanese(x))
        df["onEn_tags"] = df["tags"].apply(lambda x: x.encode('utf-8').isalnum())
        df["onEn_description"] = df["description"].apply(lambda x: x.encode('utf-8').isalnum())
        df["music_title"] = df["title"].apply(lambda x: "music" in x.lower())
        df["music_tags"] = df["tags"].apply(lambda x: "music" in x.lower())
        df["music_description"] = df["description"].apply(lambda x: "music" in x.lower())
        df["isOff"] = df["title"].apply(lambda x: "fficial" in x.lower())
        df["isOffChannell"] = df["channelTitle"].apply(lambda x: "fficial" in x.lower())
        df["isOffJa"] = df["title"].apply(lambda x: "公式" in x.lower())
        df["isOffChannellJa"] = df["channelTitle"].apply(lambda x: "公式" in x.lower())
        df["cm_title"] = df["title"].apply(lambda x: "cm" in x.lower())
        df["cm_tags"] = df["tags"].apply(lambda x: "cm" in x.lower())
        df["cm_description"] = df["description"].apply(lambda x: "cm" in x.lower())
        for col in ["categoryId", "channelTitle"]:
            freq = df[col].value_counts()
            df["freq_"+col] = df[col].map(freq)


        return df

    def all(
        self, 
        encoding_type: str,
        policy: str
        ) -> pd.DataFrame:
        self.df = impute.Imputer(self.df).tags_imputer()
        self.df = impute.Imputer(self.df).description_imputer()
        self.df = impute.Imputer(self.df).title_imputer()
        self.df = self.tags_to_num()
        self.df = self.create_features()
        if encoding_type == "onehot":
            self.df = self.to_onehot()
            self.df = self.min_max()
        elif encoding_type == "label":
            self.df = self._cat_encoder()
        else:
            raise ValueError('Select "onehot" or "label"')

        if policy == "date":
            self.df = self.to_datetime()
        elif policy == "Date":
            self.df = self.change_to_Date()
        else:
            raise ValueError('Select "date" or "Date"')

        return self.df
