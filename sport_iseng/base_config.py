# pylint: disable=all
# type: ignore

from collections.abc import MutableMapping


class BaseConfig(MutableMapping):
    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def _keytransform(self, key):
        return key

    def __repr__(self) -> str:
        return self.__dict__.__repr__()

    def __str__(self) -> str:
        return self.__dict__.__str__()
