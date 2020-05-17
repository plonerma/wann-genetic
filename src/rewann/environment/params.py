from collections.abc import Mapping

def nested_update(d, u):
    """ Update nested parameters.

    Source:
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth#3233356"""
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = nested_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

class ParamTree:
    """Wraps a dict to allow access of fields via list of keys."""

    def __init__(self):
        self.params = dict()

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = [keys]

        d = self.params
        for k in keys: d = d[k]
        return d

    def __setitem__(self, keys, value):
        if not isinstance(keys, tuple):
            keys = [keys]

        *first_keys, last_key = keys
        d = self.params
        for k in first_keys:
            d[k] = d.get(k, dict())
            d = d[k]
        d[last_key] = value

    def __contains__(self, keys):
        if not isinstance(keys, tuple):
            keys = [keys]

        d = self.params

        for k in keys:
            try: d = d[k]
            except: return False
        return True

    def update_params(self, params):
        nested_update(self.params, params)
