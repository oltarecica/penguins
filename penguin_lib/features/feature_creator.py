import pandas as pd


class FeatureCreator:
    def add_bill_ratio(self, df: pd.DataFrame):
        df = df.copy()
        df["bill_ratio"] = df["bill_length_mm"] / df["bill_depth_mm"]
        return df

    def add_mass_flipper_ratio(self, df: pd.DataFrame):
        df = df.copy()
        df["mass_flipper_ratio"] = df["body_mass_g"] / df["flipper_length_mm"]
        return df

    def add_year_centered(self, df: pd.DataFrame):
        df = df.copy()
        df["year_centered"] = df["year"] - df["year"].mean()
        return df

    def add_bill_sum(self, df: pd.DataFrame):
        df = df.copy()
        df["bill_sum"] = df["bill_length_mm"] + df["bill_depth_mm"]
        return df

    def add_interaction_term(self, df: pd.DataFrame):
        df = df.copy()
        df["interaction"] = df["bill_length_mm"] * df["flipper_length_mm"]
        return df

    def create_all(self, df: pd.DataFrame):
        df = self.add_bill_ratio(df)
        df = self.add_mass_flipper_ratio(df)
        df = self.add_year_centered(df)
        df = self.add_bill_sum(df)
        df = self.add_interaction_term(df)
        return df
