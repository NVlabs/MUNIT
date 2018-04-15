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

First, download the [pretrained models](https://drive.google.com/open?id=0BwpOatrZwxK6V1Bwai1GZFQ2Q0k) and put them in `models` folder.

###### Multimodal Translation

Run the following command to translate house cat to dog
    
    python test.py --config configs/housecat2dog.yaml --input inputs/housecat.jpg --output_folder outputs --checkpoint models/housecat2dog.pt --a2b 1
    
or vice versa
    
    python test.py --config configs/housecat2dog.yaml --input inputs/dog.jpg --output_folder outputs --checkpoint models/housecat2dog.pt --a2b 0

The results are stored in `outputs` folder. By default, it produces 10 random translation outputs.
 
###### Example-guided Translation

The above command translates a cat into random dogs. In addition, it is possible to translate a cat to the style of a specific dog. For example, the following command produces an output image that combines the pose of `inputs/housecat.jpg` and the style of `inputs/dog.jpg`.
    
    python test.py --config configs/housecat2dog.yaml --input inputs/housecat.jpg --output_folder outputs --checkpoint models/housecat2dog.pt --a2b 1 --style inputs/dog.jpg

 
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
