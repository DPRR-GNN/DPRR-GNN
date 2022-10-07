# DPRR-GNN

This is a source code of DPRR (Degree-Preserving Randomized Response) for GNN (Graph Neural Networks) in the following paper: 

https://arxiv.org/abs/2202.10209

# Installation
You should refer to required modules and install these. 
If you would like to use pipenv, the commands for installations are available.

```
$ pipenv --python 3
$ pipenv lock 
$ pipenv sync
$ pipenv shell
```
(Optional) If you would like to use jupyter and visualize results, this command is available.

```
$ pipenv sync --dev
```


# Executation
Command of execution of models is below: 

```
$ python ./execute_models.py --conf_file [config file path]
```

For example,
```
$ python ./execute_models.py --conf_file ./conf/no_noise_powerful.yml
```

Set the parameters in the yaml files. The samples is located in ./conf/.

# Evaluation
Tensorboard is available.

```
$ tensorboard --logdir [log dir]
```

For example,
```
$ tensorboard --logdir ./logs/no_noise_seed:0_alpha:0.1
```