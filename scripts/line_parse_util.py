from functools import wraps
import inspect
import shlex
import argparse

class LineParser(argparse.ArgumentParser):
    def exit(self, status=0, message=None):
        if status:
            raise ValueError(f'Exiting because of an error: {message}')

def parse_line(func):
    parser = LineParser()

    params = inspect.signature(func).parameters
    parser.prog = func.__name__[3:]

    for name, p in list(params.items())[1:]:
        args = dict()

        as_flag = False

        if issubclass(p.annotation, list):
            args['nargs'] = '+'
            args['type'] = str
            args['action'] = 'store'


        elif issubclass(p.annotation, bool) or isinstance(p.default, bool):
            as_flag = True

            if p.default is inspect._empty:
                args['action'] = 'store_true'
            else:
                args['action'] = 'store_const'
                args['const'] = not p.default

        elif p.annotation is not inspect._empty:
            args['type'] = p.annotation
            args['action'] = 'store'

        elif p.default is not inspect._empty:
            args['type'] = type(p.default)
            args['action'] = 'store'

        else:
            args['type'] = str
            args['action'] = 'store'

        if p.default is not inspect._empty:
            as_flag = True
            args['default'] = p.default

        if as_flag:
            name = f'--{name}'

        parser.add_argument(name, **args)

    @wraps(func)
    def decorated_func(self, line):
        try:
            args = parser.parse_args(shlex.split(line))
            func(self, **vars(args))
        except ValueError as e:
            pass

    return decorated_func
