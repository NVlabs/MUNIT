## MUNIT: Multimodal UNsupervised Image-to-image Translation

### License

Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

### Paper

[Xun Huang](http://www.cs.cornell.edu/~xhuang/), [Ming-Yu Liu](http://mingyuliu.net/), [Serge Belongie](https://vision.cornell.edu/se3/people/serge-belongie/), [Jan Kautz](http://jankautz.com/), [Multimodal Unsupervised Image-to-Image Translation"]()

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

Download the pretrained model in [link](https://drive.google.com/open?id=0BwpOatrZwxK6V1Bwai1GZFQ2Q0k) to <models> folder

###### Multimodal Translation

2. Run the following command to translate house cat to dog
    ```
    python test.py --config configs/housecat2dog.yaml --input ./inputs/cat.jpg --output_folder outputs --checkpoint models/housecat2dog.pt
    ```
    or vice versa
    ```
    python test.py --config configs/housecat2dog.yaml --input ./inputs/dog.jpg --output_folder outputs --checkpoint models/housecat2dog.pt --a2b 0
    ```

3. Check out the diverse results in <outputs>.
 
###### Example-guided Translation
1. Download the pretrained model in [link](https://drive.google.com/open?id=0BwpOatrZwxK6NktUSWZRNE14Ym8) to <outputs/unit/corgi2husky>

2. Go to <src> and run to translate the first cat and second cat to tigers
    ```
    python cocogan_translate_one_image.py --config ../exps/unit/corgi2husky.yaml --a2b 1 --weights ../outputs/unit/corgi2husky/corgi2husky_gen_00500000.pkl --image_name ../images/corgi001.jpg --output_image_name ../results/corgi2husky_corgi001.jpg
    ```
    ```
    python cocogan_translate_one_image.py --config ../exps/unit/corgi2husky.yaml --a2b 0 --weights ../outputs/unit/corgi2husky/corgi2husky_gen_00500000.pkl --image_name ../images/husky001.jpg --output_image_name ../results/husky2corgi_husky001.jpg
    ```

4. Check out the results in <results>. Left: Input. Right: Output
 - ![](./results/corgi2husky_corgi001.jpg)
 - ![](./results/husky2corgi_husky001.jpg)
 
#### Training
1. Download the aligned and crop version of the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to <datasets/celeba>. 

2. Go to <datasets/celeba> and crop the middle region of the face images and resize them to 128x128
    ```
    python crop_and_resize.py;
    ```

3. Setup the yaml file. Check out <exps/unit/blondhair.yaml>

4. Go to <src> and do training
     ```
    python cocogan_train.py --config ../exps/unit/blondhair.yaml --log ../logs
    ```
5. Go to <src> and do resume training 
     ```
    python cocogan_train.py --config ../exps/unit/blondhair.yaml --log ../logs --resume 1
    ```
    
6. Intermediate image outputs and model binary files are in <outputs/unit/blondhair>

For more pretrained models, please check out the google drive folder [Pretrained models](https://drive.google.com/open?id=0BwpOatrZwxK6UGtheHgta1F5d28).
