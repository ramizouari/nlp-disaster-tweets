import asyncio
import re
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class PreprocessLayer(ABC):
    @abstractmethod
    def preprocess(self, df):
        pass


def is_address_form(address):
    match = re.search("((\w+)[ ,.]*)+", address)
    return match


def match_everything(address):
    return re.match(".*", address)


def register_metadata(metadata, index, value):
    if metadata is not None and index is not None:
        metadata[index] = value


class GeolocationStepConverter(ABC):
    """
    Abstract class for converting a location to a geolocation
    """

    @abstractmethod
    def convert(self, location, *, metadata=None, index=None):
        """
        Try to convert the location to a geolocation
        :param location: The location to convert
        :param metadata: The metadata to update
        :param index: The index to update
        :return: The potentially converted location and a boolean indicating if the location was transformed
        """
        pass

    def __call__(self, location, *, metadata=None, index=None):
        return self.convert(location, metadata=metadata, index=index)


class AsyncGeolocationStepConverter(ABC):
    """
    Abstract class for converting a location to a geolocation
    """

    @abstractmethod
    async def convert(self, location, *, metadata=None, index=None):
        """
        Try to convert the location to a geolocation
        :param location: The location to convert
        :param metadata: The metadata to update
        :param index: The index to update
        :return: The potentially converted location and a boolean indicating if the location was transformed
        """
        pass

    async def __call__(self, location, *, metadata=None, index=None):
        return await self.convert(location, metadata=metadata, index=index)


class StripLocation(GeolocationStepConverter):
    def convert(self, location, *, metadata=None, index=None):
        return location.strip(), False


class AsyncStripLocation(GeolocationStepConverter):
    async def convert(self, location, *, metadata=None, index=None):
        return location.strip(), False


class AddressToGeolocation(GeolocationStepConverter):
    def __init__(self, geolocator, address_extractor=False):
        self.geolocator = geolocator
        if address_extractor == True:
            self.extractor = to_address_form
        elif address_extractor == False:
            self.extractor = match_everything
        else:
            self.extractor = address_extractor

    def convert(self, address, *, metadata=None, index=None):
        extracted = self.extractor(address)
        if not extracted:
            return address, False
        location = self.geolocator.geocode(address, addressdetails=True)
        if location is None:
            return address, False
        register_metadata(metadata, index, location)
        return location, True


class AsyncAddressToGeolocation(AsyncGeolocationStepConverter):
    def __init__(self, geolocator, address_extractor=False):
        self.geolocator = geolocator
        if address_extractor == True:
            self.extractor = to_address_form
        elif address_extractor == False:
            self.extractor = match_everything
        else:
            self.extractor = address_extractor

    async def convert(self, address, *, metadata=None, index=None):
        extracted = self.extractor(address)
        if not extracted:
            return address, False
        location = await self.geolocator.geocode(address)
        if location is None:
            return address, False
        register_metadata(metadata, index, location)
        return location, True


def extract_lat_long_form(location):
    match = re.search("([-+]?[0-9]*\.?[0-9]+),([-+]?[0-9]*\.?[0-9]+)", location)
    return match


class LatLongToGeolocation(GeolocationStepConverter):
    def __init__(self, geolocator):
        self.geolocator = geolocator

    def convert(self, location, *, metadata=None, index=None):
        match = extract_lat_long_form(location)
        if not match:
            return location, False
        register_metadata(metadata, index, location)
        # Use geocoding to convert lat-long to address
        location = self.geolocator.reverse(match[0])
        return location, True


class AsyncLatLongToGeolocation(AsyncGeolocationStepConverter):
    def __init__(self, geolocator):
        self.geolocator = geolocator

    async def convert(self, location, *, metadata=None, index=None):
        match = extract_lat_long_form(location)
        if not match:
            return location, False
        register_metadata(metadata, index, location)
        # Use geocoding to convert lat-long to address
        location = await self.geolocator.reverse(match[0])
        return location, True


def transform(location, transformers):
    if len(transformers) == 0:
        return np.nan, False
    location, transformed = transformers[0](location)
    if transformed:
        return location, transformed
    return transform(location, transformers[1:])


async def async_transform(location, transformers):
    if len(transformers) == 0:
        return np.nan, False
    result = await transformers[0](location)
    location, transformed = result
    if transformed:
        return location, transformed
    return await async_transform(location, transformers[1:])


class DefaultPreprocessLocationLayer(PreprocessLayer):
    TRANSFORMERS = []

    def __init__(self, force_update=False):
        self.force_update = force_update

    def _task(self, row):
        if row.geolocation_converted:
            return row.geolocation
        return list(transform(row.location, self.TRANSFORMERS))

    async def _task_async(self, row):
        if row.geolocation_converted:
            return row.geolocation
        return list(await async_transform(row.location, self.TRANSFORMERS))

    def _prepare(self, df):
        # First iteration to check if the location is in address form
        if self.force_update or "geolocation" not in df.columns:
            df["geolocation"] = df["location"]
            df["geolocation_converted"] = False

    def preprocess(self, df):
        self._prepare(df)
        df[["geolocation", "geolocation_converted"]] = df.apply(
            self._task, axis=1, result_type="expand"
        )
        return df

    async def apreprocess(self, df):
        self._prepare(df)
        df[["geolocation", "geolocation_converted"]] = pd.DataFrame(
            await asyncio.gather(
                *[asyncio.create_task(self._task_async(row)) for row in df.itertuples()]
            ),
            index=df.index,
            columns=["geolocation", "geolocation_converted"],
        )
        return df
