from setuptools import setup, find_packages

src_dir = 'src'

setup(
    name="rewann",
    packages=find_packages(src_dir),
    package_dir={'': src_dir},
    author='Max Ploner',
    author_email='rewann@maxploner.de',
    entry_points={
        'console_scripts': [
            'run_experiment = rewann.cli:run_experiment',
        ],
    },
)
