## MUNIT: Multimodal UNsupervised Image-to-image Translation

### License

Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

### Paper

[Xun Huang](http://www.cs.cornell.edu/~xhuang/), [Ming-Yu Liu](http://mingyuliu.net/), [Serge Belongie](https://vision.cornell.edu/se3/people/serge-belongie/), [Jan Kautz](http://jankautz.com/), [Multimodal Unsupervised Image-to-Image Translation]()

Please cite our paper if this software is used in your publications.

### Dependency


pytorch, yaml, and tensorboard (from https://github.com/dmlc/tensorboard).


The code base was developed using Python 2 in Anaconda2 with the following packages.
```
conda install pytorch torchvision cuda80 -c soumith
conda install -y -c anaconda pip; 
conda install -y -c anaconda yaml;
pip install tensorboard;
```

We also provide a [Dockerfile](Dockerfile) for building an environment for running the MUNIT code.

### Example Usage

#### Testing 

First, download the [pretrained models](https://drive.google.com/drive/folders/10IEa7gibOWmQQuJUIUOkh-CV4cm6k8__?usp=sharing) and put them in `models` folder.

###### Multimodal Translation

Run the following command to translate edges to shoes
    
    python test.py --config configs/edges2shoes.yaml --input inputs/edge.jpg --output_folder outputs --checkpoint models/edges2shoes.pt --a2b 1
    
or vice versa
    
    python test.py --config configs/edges2shoes.yaml --input inputs/shoe.jpg --output_folder outputs --checkpoint models/edges2shoes.pt --a2b 0

The results are stored in `outputs` folder. By default, it produces 10 random translation outputs.
 
###### Example-guided Translation

The above command outputs diverse shoes from an edge input. In addition, it is possible to control the style of output using an example shoe image.
    
    python test.py --config configs/edges2shoes.yaml --input inputs/edge.jpg --output_folder outputs --checkpoint models/edges2shoes.pt --a2b 1 --style inputs/shoe.jpg

 
#### Training
1. Download the dataset you want to use. For example, you can use the edges2shoes dataset provided by [Zhu et al.](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

3. Setup the yaml file. Check out `configs/edges2shoes.yaml`. Change the `data_root` field to the path of your downloaded dataset.

3. Start training
     ```
    python train.py --config configs/edges2shoes.yaml
    ```
    
4. Intermediate image outputs and model binary files are stored in `outputs/edges2shoes`
