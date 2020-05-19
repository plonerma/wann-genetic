from setuptools import setup, find_packages

src_dir = 'src'

setup(
    name="rewann",
    packages=find_packages(src_dir),
    package_dir={'': src_dir},
    author='Max Ploner',
    version='0.1a.0',
    author_email='rewann@maxploner.de',
    #scripts=['src/rewann/cli/inspect_run', 'src/rewann/cli/inspect_run.py'],
    entry_points={
        'console_scripts': [
            # execution of a single experiment
            'run_experiment = rewann.environment.run:run_experiment',

            # post execution reporting
            'compile_report = rewann.postopt.report:compile_report',
            'draw_network = rewann.postopt.report:draw_network',

            # multivariate experiment series generation
            'generate_experiment_series = rewann.tools.generate_experiments:generate_experiments',
        ],
    },
    package_data={'rewann': ['environment/default.toml']},
    install_requires=[
        'pytest-watch', 'sklearn', 'numpy', 'mnist', 'opencv-python',
        'h5py', 'tabulate', 'matplotlib', 'networkx', 'toml', 'pandas']
)
