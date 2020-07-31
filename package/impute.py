""" For missing values imputation"""

import numpy as np
import pandas as pd


class Imputer:
    def __init__(
        self,
        df: pd.DataFrame
    ) -> None:
        self.df = df

    
    def num_imputer(
        self, 
        cols: list
    ) -> pd.DataFrame:
        df = self.df.copy()
        for col in cols:
            df[col] = df[col].fillna(df[col].mean())
        
        return df


    def cat_imputer(
        self,
        cols: list
    ) -> pd.DataFrame:
        df = self.df.copy()
        for col in cols:
            df[col] = df[col].fillna(df[col].mode())
        
        return df

    
    def tags_imputer(self) -> pd.DataFrame:
        df = self.df.copy()
        df["tags"] = df["tags"].map(lambda x: "[none]" if type(x) == np.float else x)

        return df

    def description_imputer(self) -> pd.DataFrame:
        df = self.df.copy()
        df["description"].fillna(" ", inplace=True)

        return df
    
    def title_imputer(self) -> pd.DataFrame:
        df = self.df.copy()
        df["title"].fillna(" ", inplace=True)

        return df


