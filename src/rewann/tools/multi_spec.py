import copy
import os
import toml

class Specification:
    name_parts = None  # used for temporarily storing variation value names

    def __init__(self, path):
        assert os.path.isfile(path)

        spec = toml.load(path)

        self.name_fstr = spec['experiment_name']

        self.name_parts

        self.base_params_path = os.path.join(
            os.path.dirname(path),
            spec.get('base_params', 'base.toml'),
        )

        self.base_params = toml.load(self.base_params_path)

        self.variations = list()

        for name, variation in spec.items():
            if not isinstance(variation, dict):
                if not name in ('experiment_name', 'base_params'):
                    logging.warning(f'Key {name} unkown. Skipping.')
                continue

            self.variations.append((
                name, variation['key'], variation['values'],
                variation.get('fmt', None)))

    def update_name_parts(self, var_index, v):
        name, _, _, fmt = self.variations[var_index]

        # update name parts
        if isinstance(v, dict) and '_fmt' in v:
            fmt = v.pop('_fmt')

        if fmt is None:
            self.name_parts[name] = str(v)
        else:
            if isinstance(v, dict):
                self.name_parts[name] = fmt.format(**v)
            elif isinstance(v, list):
                self.name_parts[name] = fmt.format(*v)
            else:
                self.name_parts[name] = fmt.format(v)


    def get_exp_name(self):
        return self.name_fstr.format(**self.name_parts)

    def update_params(self, params, var_index, value, flat=False):
        name, key, _, _ = self.variations[var_index]
        params = copy.deepcopy(params)
        target = params

        if flat:
            if isinstance(value, dict):
                for k, v in value.items():
                    params[f"{name}/{k}"] = v
            else:
                params[name] = value
            params[f"{name}/_name"] = self.name_parts[name]
        else:
            if isinstance(key, list):
                for k in key[:-1]:
                    try:
                        target = target[k]
                    except KeyError:
                        target[k] = dict()
                        target = target[k]

                last_key = key[-1]
            else:
                last_key = key

            # modify params
            if (isinstance(value, dict)
                and last_key in target
                and isinstance(target[last_key], dict)):

                target[last_key].update(value)
            else:
                target[last_key] = value

        return params

    def generate_flat_values(self):
        yield from self.generate_experiments(params=dict(), flat=True)


    def generate_experiments(self, params=None, var_index=0, flat=False):
        if var_index == 0:
            self.name_parts = dict()

            if params is None:
                params = self.base_params

        # apply variation
        _, _, values, _ = self.variations[var_index]

        for v in values:
            self.update_name_parts(var_index, v)


            p_next = self.update_params(params, var_index, v, flat)


            if len(self.variations) - 1 == var_index:  # last variation
                p_next['experiment_name'] = self.get_exp_name()
                yield p_next
            else:
                # traverse to next variation
                yield from self.generate_experiments(p_next, var_index+1, flat=flat)
