""" Command Line Script for generating a series of experiments based on
    specification and base parameters for multivariate testing. """


import argparse
import toml
import os
import logging
from collections import namedtuple
import copy




def apply_variations(params, variations, exp_name_fstr, name_parts=dict()):
    if len(variations) == 0:
        params['experiment_name'] = exp_name_fstr.format(**name_parts)
        yield params
    else:
        # apply variation
        name, key, values, fmt = variations.pop(0)

        for v in values:
            p_next = copy.deepcopy(params)

            target = p_next

            if isinstance(key, list):
                for k in key[:-1]:
                    target = target[k]

                last_key = key[-1]
            else:
                last_key = key

            # update name parts
            if fmt is not None:
                if isinstance(v, dict):
                    name_parts[name] = fmt.format(**v)
                elif isinstance(v, list):
                    name_parts[name] = fmt.format(*v)
                else:
                    name_parts[name] = fmt.format(v)
            else:
                name_parts[name] = str(v)

            # modify params
            if isinstance(v, dict) and isinstance(target[last_key], dict):
                target[last_key].update(v)
            else:
                target[last_key] = v

            # traverse to next variation
            yield from apply_variations(p_next, list(variations), exp_name_fstr, name_parts)

def generate_experiments(specification_path, out_dir):
    assert os.path.isfile(specification_path)

    specification = toml.load(specification_path)

    exp_fmt_str = specification['experiment_name']

    base_params_path = os.path.join(
        os.path.dirname(template_path),
        specification['base_params']
    )

    base_params = toml.load(base_params_path)

    variations = list()

    for name, variation in specification.items():
        if not isinstance(variation, dict):
            if not name in ('experiment_name', 'base_params'):
                logging.warning(f'Key {name} unkown. Skipping.')
            continue

        variations.append((
            name, variation['key'], variation['values'],
            variation.get('fmt', None)))

    if not os.path.exists(out_dir): os.makedirs(out_dir)
    for params in apply_variations(base_params, variations, exp_fmt_str):
        file_name = f"{params['experiment_name'].lower().replace(' ', '_')}.toml"
        file_path = os.path.join(out_dir, file_name)
        print(f'Saving {file_path}')
        with open(file_path, 'w') as f:
            toml.dump(params, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a series of experiments based on specification and base parameters for multivariate testing.")
    parser.add_argument('specification', type=str)
    parser.add_argument('--build_dir', '-o', type=str, default='build')
    args = parser.parse_args()
    generate_experiments(args.specification, args.build_dir)
