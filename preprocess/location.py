import asyncio
import re
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple,Callable,Optional,Any,List
from geopy.extra.rate_limiter import RateLimiter,AsyncRateLimiter
from geopy.geocoders.base import Geocoder

def is_address_form(address:str) -> Optional[re.Match]:
    """Check if the given string can be considered as an address form
    Args:
        address (str): The address to check

    Returns:
        re.Match: The match object if the address is in address form, None otherwise
    Examples:
    >>> is_address_form("Cachan Paris, France; Et toi ?")
    >>> Match object, match='Cachan Paris, France'
    """
    match = re.search(R"((\w+)[ ,.]*)+", address)
    return match


def match_everything(address: str) -> Optional[re.Match]:
    """Match everything
    Args:
        address (str): The address to match

    Returns:
        Optional[re.Match]: The match object
    """
    return re.match(".*", address)


def register_metadata(metadata, index, value):
    if metadata is not None and index is not None:
        metadata[index] = value
def to_address_form(address:str, scope:str)-> Tuple[str,bool]:
    """Extract the corresponding address form from a location string based on the scope
    Args:
        address (str): the full adress
        scope (str): One of "country" or "state" or "county"

    Raises:
        ValueError: If the scope is invalid

    Returns:
        Tuple[str, bool]: The extracted address form and a boolean indicating if the address was transformed
        
    Examples:
    >>> to_address_form("Paris, France", "country")
    >>>("France", True)
    """
    if not is_address_form(address):
        return address, False
    match scope:
        case "country":
            return address.split(",")[-1], True
        case "state":
            return address.split(",")[-2], True
        case "county":
            return address.split(",")[-3], True
        case _:
            raise ValueError("Invalid scope")

def extract_lat_long_form(location:str)-> Optional[re.Match]:
    """Extract latitude and longitude from a location

    Args:
        location (str): The location to extract

    Returns:
        Optional[re.Match]: The match object if the location is in lat-long form, None otherwise
    """
    match = re.search("([-+]?[0-9]*\.?[0-9]+),([-+]?[0-9]*\.?[0-9]+)", location)
    return match


async def async_transform(location, transformers):
    if len(transformers) == 0:
        return np.nan, False
    result = await transformers[0](location)
    location, transformed = result
    if transformed:
        return location, transformed
    return await async_transform(location, transformers[1:])

class PreprocessLayer(ABC):
    """Abstract class representing a preprocess layer
    """
    @abstractmethod
    def preprocess(self, df:pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataframe

        
            df (pd.Dataframe): The dataframe to preprocess
        Returns:
            pd.Dataframe: The preprocessed dataframe
        """

        pass
class GeolocationStepConverter(ABC):
    """
    Abstract class for converting a location to a geolocation
    """

    @abstractmethod
    def convert(self, location : str) -> Tuple[Any,bool]:
        """
        Try to convert a location to its geolocation in standard format.
        Args:
            location (Any): The location to convert
        Returns: 
            The potentially converted location and a boolean indicating if the location was transformed
        """
        pass

    def __call__(self, location :str):
        return self.convert(location)

class StripLocation(GeolocationStepConverter):
    """
    Strip the location of any leading or trailing whitespace
    """
    def convert(self, location:str)-> Tuple[str,bool]:
        return location.strip(), False

class AddressToGeolocation(GeolocationStepConverter):
    """
    Convert an address to a geolocation
    """
    def __init__(self, geolocator: Geocoder, address_extractor:Callable[[str],Optional[re.Match]] | bool=False):
        """
        Initialize the converter
        Args: 
            geolocator: The geolocator to use
            address_extractor: The function to use to extract the address

        """
        self.geolocator = geolocator
        if address_extractor == True: # noqa: E712
            self.extractor = to_address_form
        elif address_extractor == False: # noqa: E712
            self.extractor = match_everything
        else:
            self.extractor = address_extractor

    def convert(self, address:str)-> Tuple[Any,bool]:
        extracted = self.extractor(address)
        if not extracted:
            return address, False
        location = self.geolocator.geocode(address, addressdetails=True)
        if location is None:
            return address, False
        return location, True

class LatLongToGeolocation(GeolocationStepConverter):
    """
    Convert a lat-long to a geolocation
    """
    def __init__(self, geolocator:Geocoder):
        """
        Initialize the converter
        Args:
             geolocator(Geocoder): The geolocator to use
        """
        self.geolocator = geolocator

    def convert(self, location:str) ->Tuple[str,bool]:
        match = extract_lat_long_form(location)
        if not match:
            return location, False
        location = self.geolocator.reverse(match[0])
        return location, True
    
def transform(location:str, transformers : List[GeolocationStepConverter]) -> Tuple[Any, bool]:
    """Apply a list of transformers to a location

    Args:
        location (str): The location to transform
        transformers (List[GeolocationStepConverter]): List of transformers

    Returns:
        Tuple[Any, bool]: The transformed location and a boolean indicating if the location was transformed
    """
    if len(transformers) == 0:
        return np.nan, False
    location, transformed = transformers[0](location)
    if transformed:
        return location, transformed
    return transform(location, transformers[1:])

class DefaultPreprocessLocationLayer(PreprocessLayer):
    """
    Default location preprocessing layer
    """

    def __init__(self, force_update:bool=False,transformers:List[GeolocationStepConverter] | None = None):
        """ Inialize the location preprocessing layer

        Args:
            force_update (bool, optional): preprocess locations, even it was already done. Defaults to False.
            transformers (List[GeolocationStepConverter], optional): list of preprocessing layers. Defaults to None.
        """
        if transformers is None:
            transformers = []

        self.force_update = force_update
        self.transformers = transformers

    def _task(self, row:pd.Series)-> list:
        if row.geolocation_converted:
            return [row.geolocation,row.geolocation_converted]
        return list(transform(row.location, self.transformers))

    def _prepare(self, df:pd.DataFrame) -> None:
        # First iteration to check if the location is in address form
        if self.force_update or "geolocation" not in df.columns:
            df["geolocation"] = df["location"]
            df["geolocation_converted"] = False

    def preprocess(self, df:pd.DataFrame) -> pd.DataFrame:
        """Preprocess the location

        Args:
            df (pd.DataFrame): The dataframe to preprocess

        Returns:
            pd.DataFrame: The preprocessed dataframe, with the additional geolocation and geolocation_converted columns
        """

        self._prepare(df)
        df[["geolocation", "geolocation_converted"]] = df.apply(
            self._task, axis=1, result_type="expand"
        )
        return df

class RateLimitedGeocoder:
    def __init__(self, geolocator, *args,**kwargs):
        self.geolocator=geolocator
        self.geocode=RateLimiter(self.geolocator.geocode, *args, **kwargs)
        self.reverse=RateLimiter(self.geolocator.reverse, *args, **kwargs)


class AsyncRateLimitedGeocoder:
    def __init__(self, geolocator, *args,**kwargs):
        self.geolocator=geolocator
        self.geocode=AsyncRateLimiter(self.geolocator.geocode, *args, **kwargs)
        self.reverse=AsyncRateLimiter(self.geolocator.reverse, *args, **kwargs)