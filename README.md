## Deploying keras model with periodic trainer

This code presents an approach to train keras models periodically which make your predictions always trained by latest dataset.

## Requirements

docker 

docker-compose

## Installation:

```
git clone https://github.com/patron-labs/periodic_keras_model_training_example

cd periodic_keras_model_training_example

mkdir data

mkdir ml_models

docker-compose up --build -d
```

## Documentation:

Please check [this](https://medium.com/patron-ai/deploying-keras-model-to-production-by-periodic-training-37842eb8e84e) medium post. 

You can import keras_model_deploy_example.json to insomnia-rest and postman to test endpoints in api service.

