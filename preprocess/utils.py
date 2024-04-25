from abc import ABC, abstractmethod
import pandas as pd  # type:ignore


class PreprocessLayer(ABC):
    """Abstract class representing a preprocess layer"""

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataframe


            df (pd.Dataframe): The dataframe to preprocess
        Returns:
            pd.Dataframe: The preprocessed dataframe
        """

        pass
