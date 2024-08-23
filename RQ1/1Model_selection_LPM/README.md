# Training
Firstly, modify the configuration file:
Config
Update the dataset name and model.

Run the following command to train the model in the background:
```sh
nohup python Train.py > ./Log/Train_log/<dataset>/<model>.txt &
```

Copy the last two lines of the log file to `Result/Table3record.txt`.

# Testing
Insert the best model from the training folder into the configuration file.

Check if the dataset name on line 210 in `test.py` is correct.
Verify if the model selection on line 205 in `test.py` is correct.

Run the following command to test the model in the background:
```sh
nohup python Test.py > ./Log/Test_log/<dataset>/<model>.txt &
```