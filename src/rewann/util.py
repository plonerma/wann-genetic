from collections.abc import Mapping, MutableMapping
from collections import UserDict
from typing import Iterable

def nested_update(d : MutableMapping, u : Mapping) -> MutableMapping:
    """ Update nested parameters.

    Source:
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth#3233356"""
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = nested_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

class ParamTree(UserDict):
    """Wraps a dict to allow access of fields via list of keys."""

    def _proper_keys(self, keys):
        if not isinstance(keys, tuple) and not isinstance(keys, list):
            return [keys]

    def __getitem__(self, keys):
        d = self.data
        for k in self._proper_keys(keys):
            d = d[k]
        return d

    def __setitem__(self, keys, value):
        *first_keys, last_key = self._proper_keys(keys)

        d = self.data

        for k in first_keys:
            d[k] = d.get(k, dict())
            d = d[k]

        d[last_key] = value

    def __contains__(self, keys):
        d = self.data

        for k in self._proper_keys(keys):
            try: d = d[k]
            except: return False
        return True

    def update_params(self, update : Mapping):
        """Perform a nested update on the stored parameters."""
        nested_update(self.data, update)

    def update_params_at(self, keys, update : Mapping):
        """Perform a nested update on the stored parameters starting at field referenced by keys."""
        *first_keys, last_key = self._proper_keys(keys)

        target = self.data

        for k in first_keys:
            target[k] = d.get(k, dict())
            target = target[k]

        if (isinstance(update, Mapping)
            and last_key in target
            and isinstance(target[last_key], Mapping)):
            nested_update(target[last_key], update)
        else:
            target[last_key] = update
