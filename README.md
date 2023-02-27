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

