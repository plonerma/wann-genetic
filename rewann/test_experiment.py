from rewann import Experiment


print ("Creating experiment.")

exp = Experiment(params='test_experiment.toml')


print (f"Running experiment (path: {exp.fs.base_path})")

exp.run()
