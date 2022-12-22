HiFi-GAN
=========
PyTorch implementation of text-to-speech vocoder HiFi-GAN. Implemented as a home assignment on deep learning in audio course.

Requirements
============
Install requirements by

`pip install -r requirements.txt`

Training
========
To run training you only need to setup your model in config by path `config/config.yaml`
and run `train.py` script. Audios, whitch you get during training will be stored by path
`results/`, and checkpoints of your model will be available in `checkpoint/` after every
epoch.