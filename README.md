# Car Predictor
Simple ML project to predict cars based on the stanford cars dataset

## How to train your model
Clone the dataset found in https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder into 
`.helpers/car_data/` and ensure that the train and test data is split properly into folders.

Run  `./helpers/model-trainer.py`

When the training is done you will have a model created in `./trained_models/car_classifier.pt`

## TODO
Create a FLask webapp that will test new images to see which type of car they are
