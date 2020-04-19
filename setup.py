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
<<<<<<< HEAD
    data_files=[('environment', ['src/rewann/environment/default.toml'])],
=======
    data_files=[('environment', ['src/rewann/environment/default.toml'])]
>>>>>>> 6604a75f95c3889f0d04bd04b8bf35538299f467
    install_requires=['pytest-watch', 'streamlit', 'sklearn', 'numpy', 'mnist', 'opencv-python', 'h5py']
)
