# Adversarial Attacks on Variational Autoencoders

Arxiv link will be posted in the near future!

Please cite our work:
> George Gondim-Ribeiro, Pedro Tabacof, and Eduardo Valle. Adversarial Attacks on Variational Autoencoders.

## Running

After downloading and preparing the data, to reproduce our experiments, first, run the script trainer.py to train a model and the attack.py to start the attack choosing the same directory. Then, you can compute the AUDDC on the normalized results.


For instance:
```
# Download data
python get_data.py

# Prepare CelebA dataset
python celeba_prepare_data.py

# Train model
python trainer.py --epochs 50 --dataset mnist --architecture vae --dir /tmp/test

# Attack model
python attack.py --dir /tmp/test/ --num_attacks 5

# Compute metrics
python compute_metrics.py --dir /tmp/test
```

### Architectures

vae: Variational autoencoders with only fully-connected layers

cvae: Variational autoencoders with convolutional layers

draw: DRAW


### Datasets

mnist: MNIST dataset

svhn: SVHN dataset

celeba: CelebA dataset

### License
This code is based upon the repositories below, please check their license before any use. This code is intended ONLY for academic purposes.
* https://github.com/sjchoi86/advanced-tensorflow
* https://github.com/ericjang/draw
