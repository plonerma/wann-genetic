from setuptools import setup, find_packages

src_dir = 'src'

setup(
    name="rewann",
    packages=find_packages(src_dir),
    package_dir={'': src_dir},
    author='Max Ploner',
    author_email='rewann@maxploner.de',
    #scripts=['src/rewann/cli/inspect_run', 'src/rewann/cli/inspect_run.py'],
    entry_points={
        'console_scripts': [
            'run_experiment = rewann.cli.run:run_experiment',
        ],
    },
    data_files=[('', ['src/rewann/environment/default.toml'])],
    install_requires=['pytest-watch', 'streamlit', 'sklearn', 'numpy', 'mnist', 'opencv-python', 'h5py']
)
