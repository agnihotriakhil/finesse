##### Introduction

This repository contains two approaches towards generating fashion clothing items given a prompt, similar to the dataset given in `$ROOT/data` directory.

The first approach is using Stable Diffusion (`stable-diffusion-v1-4`) and finetuning it to given dataset. The second approach is creating a Conditional Variational Auto-Encoder (CVAE) model and training it on the given dataset.

More information about each of the above approaches can be found in `stable-diffusion/` and `cvae/` directories.

Please download the data and store it in `$ROOT/data` repository. This `data/` folder should look like:

└── data/
    ├── compressed_images/
    │   ├── image_id_1.jpg
    │   ├── image_id_2.jpg
    │   └── ...
    └── product_data.json

###### Model and Training

Please refer to the `cvae/` and `stable-diffusion/` directories for more details.

###### Sample Results

Sample results for `stable-diffusion-v1-4` can be found in `$ROOT/stable-diffusion/README.md`.
