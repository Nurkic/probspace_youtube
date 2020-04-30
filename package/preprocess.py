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
        df["publishedAt"] = pd.to_datetime(df["publishedAt"])
        df["year"] = df["publishedAt"].dt.year
        df["month"] = df["publishedAt"].dt.month
        df["day"] = df["publishedAt"].dt.day
        df["hour"] = df["publishedAt"].dt.hour
        df["minute"] = df["publishedAt"].dt.minute

        df["collection_date"] = "20" + df["collection_date"]
        df["collection_date"] = pd.to_datetime(df["collection_date"], format="%Y.%d.%m")
        df["c_year"] = df["collection_date"].dt.year
        df["c_month"] = df["collection_date"].dt.month
        df["c_day"] = df["collection_date"].dt.day

        return df

    
    def tags_to_num(self) -> pd.DataFrame:
        df = self.df.copy()
        df["length_tags"] = df["tags"].astype(str).apply(lambda x: len(x.split("|")))

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

    def all(
        self, 
        encoding_type: str,
        policy: str
        ) -> pd.DataFrame:
        self.df = self.tags_to_num()
        self.df = impute.Imputer(self.df).tags_imputer()
        """self.df = self.tags_to_col()"""
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
