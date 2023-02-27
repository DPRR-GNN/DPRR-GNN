# Setup
1. Build docker
Create docker image.

    ```
     $ sudo docker-compose build 
    ```

    If you would like to use dev environment, you should use below command.

    ```
     $ sudo docker-compose -f ./docker-compose_dev.yml build
    ```

2. Run docker container.

    ```
    $ sudo docker-compose up -d
    ```
    
    In dev environment,
    ```
    $ sudo docker-compose -f ./docker-compose_dev.yml up -d
    ```

3. Down docker container.

    ```
    sudo docker-compose down
    ```

    In dev enviroment,

    ```
    sudo docker-compose -f ./docker-compose_dev.yml  down
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

# Results

|dataset|noise|alpha|epsilon|training_loss|accuracy_train|accuracy_val|accuracy_test|AUC_train|AUC_test
|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|REDDITBINARY|DPRR-rnd|0|0.5|0.544±0.016|0.737±0.016|0.722±0.047|0.723±0.030|0.815±0.011|0.792±0.017
|REDDITMULTI5K|DPRR-rnd|0|0.5|1.393±0.011|0.404±0.010|0.380±0.020|0.372±0.015|0.745±0.006|0.725±0.010
|github_stargazers|DPRR-rnd|0|0.5|0.677±0.002|0.574±0.006|0.570±0.013|0.570±0.011|0.590±0.007|0.581±0.016
|twitch_egos|DPRR-rnd|0|0.5|0.670±0.001|0.606±0.001|0.608±0.005|0.605±0.003|0.633±0.002|0.632±0.004
