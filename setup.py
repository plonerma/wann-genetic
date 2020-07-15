from setuptools import setup, find_packages

src_dir = 'src'

setup(
    name="wann_genetic",
    packages=find_packages(src_dir),
    package_dir={'': src_dir},
    author='Max Ploner',
    version='0.1.1',
    author_email='wann_genetic@maxploner.de',
    #scripts=['src/wann_genetic/cli/inspect_run', 'src/wann_genetic/cli/inspect_run.py'],
    entry_points={
        'console_scripts': [
            # execution of a single experiment
            'run_experiment = wann_genetic.environment.environment:run_experiment',

            # post execution reporting
            'compile_report = wann_genetic.postopt.report:compile_report',
            'draw_network = wann_genetic.postopt.report:draw_network',
            'plot_gen_quartiles = wann_genetic.postopt.report:plot_gen_quartiles',
            'plot_gen_lines = wann_genetic.postopt.report:plot_gen_lines',

            # multivariate experiment series generation
            'generate_experiment_series = wann_genetic.tools.cli:generate_experiments',
        ],
    },
    package_data={'wann_genetic': ['environment/default.toml']},
    install_requires=[
        'pytest-watch', 'sklearn', 'numpy', 'mnist', 'opencv-python',
        'h5py', 'tabulate', 'matplotlib', 'networkx', 'toml', 'pandas', 'torch']
)
