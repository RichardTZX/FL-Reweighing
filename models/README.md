# Server-Client Simulations

## Adult Classifier Instructions
- Ensure that the ```data/adult/data/train``` and ```data/adult/data/test``` directories contain data
- If you use the 'Simulated GAN' setup make sure to delete ```data_niid_*.json``` in ```data/adult/data/train```. You should only have ```gan_data_*.json``` in this directory.
- Write a config file following the example of ```adult_default.cfg```. You should change the number of clients selected at each round, the learning rate, heterogeneities,...
- Run ```python3 main.py --config adult_default.cfg```
- For more simulation options and details, see 'Additional Notes' section

## Additional Notes
- In order to run these reference implementations, the ```-t sample``` tag must have been used when running the ```./preprocess.sh``` script for the respective dataset
- The total number of clients simulated equals the total number of users in the respective dataset's training data
- If you don't use a config file, ```main.py``` supports these additional tags:
    - ```--model```: name of model; options are listed the respective dataset folder, for example ```cnn``` for femnist; defaults to first model in the respective dataset folder
    - ```--num_rounds```: number of rounds to simulate
    - ```--eval_every```: evaluate every ___ rounds
    - ```--clients_per_round```: number of clients trained per round
    - ```--batch_size```: batch size when clients train on data
    - ```--num_epochs```: number of epochs when clients train on data
    - ```-t```: simulation time: small, medium, or large; greater time corresponds to higher accuracy; for large runs, generate data using arguments similar to those listed in the 'large-sized dataset' option in the respective dataset README file for optimal model performance; default: large
    - ```-lr```: learning rate for local optimizers. 
