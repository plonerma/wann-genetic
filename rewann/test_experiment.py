import logging
import sys
from rewann import Environment

# setup logging
root = logging.getLogger()
root.setLevel(logging.WARNING)

sh = logging.StreamHandler()
fh = logging.FileHandler('root.log')

f = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sh.setFormatter(f)
fh.setFormatter(f)

sh.setLevel(logging.WARNING)
fh.setLevel(logging.WARNING)

# streamlit registers a handler -> overwrite that
root.handlers = [sh, fh]



exp = Environment(params='test_experiment.toml', root_logger=root)
exp.run()
