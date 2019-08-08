"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from torch.utils.serialization import load_lua
from torch.utils.data import DataLoader,Dataset
from networks import Vgg16
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms, models
from data import ImageFilelist, ImageFolder
from PIL import Image
from torchvision.transforms import functional as F
import torch
import torch.nn as nn
import os
import math
import torchvision.utils as vutils
import yaml
import numpy as np
import torch.nn.init as init
import time
from resnet import resnet34
# Methods
# get_all_data_loaders      : primary data loader interface (load trainA, testA, trainB, testB)
# get_data_loader_list      : list-based data loader
# get_data_loader_folder    : folder-based data loader
# get_config                : load yaml file
# eformat                   :
# write_2images             : save output image
# prepare_sub_folder        : create checkpoints and images folders for saving outputs
# write_one_row_html        : write one row of the html file for output images
# write_html                : create the html file.
# write_loss
# slerp
# get_slerp_interp
# get_model_list
# load_vgg16
# load_flood_classifier
# load_inception
# vgg_preprocess
# get_scheduler
# weights_init

def get_all_data_loaders(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    if 'new_size' in conf:
        new_size_a = new_size_b = conf['new_size']
    else:
        new_size_a = conf['new_size_a']
        new_size_b = conf['new_size_b']
    height = conf['crop_image_height']
    width = conf['crop_image_width']

    if 'data_root' in conf:
        train_loader_a = get_data_loader_folder(os.path.join(conf['data_root'], 'trainA'), batch_size, True,
                                              new_size_a, height, width, num_workers, True)
        test_loader_a = get_data_loader_folder(os.path.join(conf['data_root'], 'testA'), batch_size, False,
                                             new_size_a, new_size_a, new_size_a, num_workers, True)
        train_loader_b = get_data_loader_folder(os.path.join(conf['data_root'], 'trainB'), batch_size, True,
                                              new_size_b, height, width, num_workers, True)
        test_loader_b = get_data_loader_folder(os.path.join(conf['data_root'], 'testB'), batch_size, False,
                                             new_size_b, new_size_b, new_size_b, num_workers, True)
    else:
        train_loader_a = get_data_loader_list(conf['data_folder_train_a'], conf['data_list_train_a'], batch_size, True,
                                                new_size_a, height, width, num_workers, True)
        test_loader_a = get_data_loader_list(conf['data_folder_test_a'], conf['data_list_test_a'], batch_size, False,
                                                new_size_a, new_size_a, new_size_a, num_workers, True)
        train_loader_b = get_data_loader_list(conf['data_folder_train_b'], conf['data_list_train_b'], batch_size, True,
                                                new_size_b, height, width, num_workers, True)
        test_loader_b = get_data_loader_list(conf['data_folder_test_b'], conf['data_list_test_b'], batch_size, False,
                                                new_size_b, new_size_b, new_size_b, num_workers, True)
    return train_loader_a, train_loader_b, test_loader_a, test_loader_b

def seg_transform():
    segmentation_transform = transforms.Compose(
            [
                 transforms.Normalize((0.485, 0.456, 0.406),\
                                      (0.229, 0.224, 0.225))
            ])
    return(segmentation_transform)

def transform_torchVar():
    transfo = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],\
                                                 [0.229, 0.224, 0.225])
                            ])    
    return(transfo)
    
    
def get_data_loader_list(root, file_list, batch_size, train, new_size=None,
                           height=256, width=256, num_workers=4, crop=True):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFilelist(root, file_list, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader

def default_txt_reader(flist):
    """
    txt format: impath \n
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip().split()
            imlist.append(impath)
    return imlist


class MyDataset(Dataset):
    def __init__(self, file_list, mask_list,new_size,height,width):
        self.image_paths  = default_txt_reader(file_list)
        self.target_paths = default_txt_reader(mask_list)
        self.new_size     = new_size
        self.height       = height
        self.width        = width
        
    def transform(self, image, mask):
        
        #print('debugging mask transform 1 size',mask.size)
        
        # Random horizontal flipping
        if torch.rand(1) > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask  = mask.transpose(Image.FLIP_LEFT_RIGHT)
            
        #print('debugging mask transform 2 size',mask.size)
        # Resize
        resize = transforms.Resize(size=self.new_size)
        image  = resize(image)
        #print('dim image after resize',image.size)
        
        # Resize mask
        
        mask        = mask.resize((image.width,image.height),Image.NEAREST)
        
        #print('debugging mask transform 3 size',mask.size)
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
                        image, output_size=(self.height, self.width))
        image      = F.crop(image, i, j, h, w)
        mask       = F.crop(mask, i, j, h, w)
        
        #print('debugging mask transform 4 size',mask.size)
        # Transform to tensor
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        
        if np.max(mask) ==1:
            mask  = to_tensor(mask) * 255 
        else :
            mask  = to_tensor(mask)
        
        # print('debugging mask transform 5 size',mask.size)       
        # Normalize
        normalizer = transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))
        image = normalizer(image)
        return image, mask

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index][0]).convert('RGB')
        mask  = Image.open(self.target_paths[index][0])
        x, y  = self.transform(image, mask)
        return x, y

    def __len__(self):
        return len(self.image_paths)

def get_data_loader_mask_and_im(file_list, mask_list, batch_size, train, new_size=None,
                           height=256, width=256, num_workers=4, crop=True):
    
    dataset    = MyDataset(file_list, mask_list,new_size,height,width)
    loader     = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader
    
    

def get_data_loader_folder(input_folder, batch_size, train, new_size=None,
                           height=256, width=256, num_workers=4, crop=True):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFolder(input_folder, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def eformat(f, prec):
    s = "%.*e"%(prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%d"%(mantissa, int(exp))


def __write_images(image_outputs, display_image_num, file_name):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs] # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_2images(image_outputs, display_image_num, image_directory, postfix, comet_exp=None):
    n = len(image_outputs)
    __write_images(image_outputs[0:n//2], display_image_num, '%s/gen_a2b_%s.jpg' % (image_directory, postfix))
    __write_images(image_outputs[n//2:n], display_image_num, '%s/gen_b2a_%s.jpg' % (image_directory, postfix))
    if comet_exp is not None:
        comet_exp.log_image('%s/gen_a2b_%s.jpg' % (image_directory, postfix))
        comet_exp.log_image('%s/gen_b2a_%s.jpg' % (image_directory, postfix))

def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_one_row_html(html_file, iterations, img_filename, all_size):
    html_file.write("<h3>iteration [%d] (%s)</h3>" % (iterations,img_filename.split('/')[-1]))
    html_file.write("""
        <p><a href="%s">
          <img src="%s" style="width:%dpx">
        </a><br>
        <p>
        """ % (img_filename, img_filename, all_size))
    return


def write_html(filename, iterations, image_save_iterations, image_directory, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = %s</title>
      <meta http-equiv="refresh" content="30">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>current</h3>")
    write_one_row_html(html_file, iterations, '%s/gen_a2b_train_current.jpg' % (image_directory), all_size)
    write_one_row_html(html_file, iterations, '%s/gen_b2a_train_current.jpg' % (image_directory), all_size)
    for j in range(iterations, image_save_iterations-1, -1):
        if j % image_save_iterations == 0:
            wrseg_transformite_one_row_html(html_file, j, '%s/gen_a2b_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_a2b_train_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_train_%08d.jpg' % (image_directory, j), all_size)
    html_file.write("</body></html>")
    html_file.close()


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


def slerp(val, low, high):
    """
    original: Animating Rotation with Quaternion Curves, Ken Shoemake
    https://arxiv.org/abs/1609.04468
    Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
    """
    omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def get_slerp_interp(nb_latents, nb_interp, z_dim):
    """
    modified from: PyTorch inference for "Progressive Growing of GANs" with CelebA snapshot
    https://github.com/ptrblck/prog_gans_pytorch_inference
    """

    latent_interps = np.empty(shape=(0, z_dim), dtype=np.float32)
    for _ in range(nb_latents):
        low = np.random.randn(z_dim)
        high = np.random.randn(z_dim)  # low + np.random.randn(512) * 0.7
        interp_vals = np.linspace(0, 1, num=nb_interp)
        latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                 dtype=np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))

    return latent_interps[:, :, np.newaxis, np.newaxis]


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def load_vgg16(model_dir):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(os.path.join(model_dir, 'vgg16.weight')):
        if not os.path.exists(os.path.join(model_dir, 'vgg16.t7')):
            os.system('wget https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1 -O ' + os.path.join(model_dir, 'vgg16.t7'))
        vgglua = load_lua(os.path.join(model_dir, 'vgg16.t7'))
        vgg = Vgg16()
        for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst.data[:] = src
        torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg16.weight'))
    vgg = Vgg16()
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    return vgg

def load_flood_classifier(ckpt_path):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(ckpt_path))
    return model

class Resnet34_8s(nn.Module):

    def __init__(self, num_classes=1000):
        super(Resnet34_8s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_8s = resnet34(fully_conv=True,
                                      pretrained=True,
                                      output_stride=8,
                                      remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_8s.fc = nn.Conv2d(resnet34_8s.inplanes, num_classes, 1)

        self.resnet34_8s = resnet34_8s

        self._normal_initialization(self.resnet34_8s.fc)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x, feature_alignment=False):
        input_spatial_dim = x.size()[2:]

        if feature_alignment:
            x = adjust_input_image_size_for_proper_feature_alignment(x, output_stride=8)

        x = self.resnet34_8s(x)

        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)

        return x

def load_segmentation_model(ckpt_path):
    model = Resnet34_8s(num_classes=19).to('cuda')
    model.load_state_dict(torch.load(ckpt_path))
    return model


# Define the helper function
def decode_segmap(image,nc=19):
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.
    Returns:
    A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((19, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(nc):
        idx = image == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb




def load_inception(model_path):
    state_dict = torch.load(model_path)
    model = inception_v3(pretrained=False, transform_input=True)
    model.aux_logits = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, state_dict['fc.weight'].size(0))
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = False
    return model

def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean)) # subtract mean
    return batch


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


def pytorch03_to_pytorch04(state_dict_base, trainer_name):
    def __conversion_core(state_dict_base, trainer_name):
        state_dict = state_dict_base.copy()
        if trainer_name == 'MUNIT':
            for key, value in state_dict_base.items():
                if key.endswith(('enc_content.model.0.norm.running_mean',
                                 'enc_content.model.0.norm.running_var',
                                 'enc_content.model.1.norm.running_mean',
                                 'enc_content.model.1.norm.running_var',
                                 'enc_content.model.2.norm.running_mean',
                                 'enc_content.model.2.norm.running_var',
                                 'enc_content.model.3.model.0.model.1.norm.running_mean',
                                 'enc_content.model.3.model.0.model.1.norm.running_var',
                                 'enc_content.model.3.model.0.model.0.norm.running_mean',
                                 'enc_content.model.3.model.0.model.0.norm.running_var',
                                 'enc_content.model.3.model.1.model.1.norm.running_mean',
                                 'enc_content.model.3.model.1.model.1.norm.running_var',
                                 'enc_content.model.3.model.1.model.0.norm.running_mean',
                                 'enc_content.model.3.model.1.model.0.norm.running_var',
                                 'enc_content.model.3.model.2.model.1.norm.running_mean',
                                 'enc_content.model.3.model.2.model.1.norm.running_var',
                                 'enc_content.model.3.model.2.model.0.norm.running_mean',
                                 'enc_content.model.3.model.2.model.0.norm.running_var',
                                 'enc_content.model.3.model.3.model.1.norm.running_mean',
                                 'enc_content.model.3.model.3.model.1.norm.running_var',
                                 'enc_content.model.3.model.3.model.0.norm.running_mean',
                                 'enc_content.model.3.model.3.model.0.norm.running_var',
                                 )):
                    del state_dict[key]
        else:
            def __conversion_core(state_dict_base):
                state_dict = state_dict_base.copy()
                for key, value in state_dict_base.items():
                    if key.endswith(('enc.model.0.norm.running_mean',
                                     'enc.model.0.norm.running_var',
                                     'enc.model.1.norm.running_mean',
                                     'enc.model.1.norm.running_var',
                                     'enc.model.2.norm.running_mean',
                                     'enc.model.2.norm.running_var',
                                     'enc.model.3.model.0.model.1.norm.running_mean',
                                     'enc.model.3.model.0.model.1.norm.running_var',
                                     'enc.model.3.model.0.model.0.norm.running_mean',
                                     'enc.model.3.model.0.model.0.norm.running_var',
                                     'enc.model.3.model.1.model.1.norm.running_mean',
                                     'enc.model.3.model.1.model.1.norm.running_var',
                                     'enc.model.3.model.1.model.0.norm.running_mean',
                                     'enc.model.3.model.1.model.0.norm.running_var',
                                     'enc.model.3.model.2.model.1.norm.running_mean',
                                     'enc.model.3.model.2.model.1.norm.running_var',
                                     'enc.model.3.model.2.model.0.norm.running_mean',
                                     'enc.model.3.model.2.model.0.norm.running_var',
                                     'enc.model.3.model.3.model.1.norm.running_mean',
                                     'enc.model.3.model.3.model.1.norm.running_var',
                                     'enc.model.3.model.3.model.0.norm.running_mean',
                                     'enc.model.3.model.3.model.0.norm.running_var',

                                     'dec.model.0.model.0.model.1.norm.running_mean',
                                     'dec.model.0.model.0.model.1.norm.running_var',
                                     'dec.model.0.model.0.model.0.norm.running_mean',
                                     'dec.model.0.model.0.model.0.norm.running_var',
                                     'dec.model.0.model.1.model.1.norm.running_mean',
                                     'dec.model.0.model.1.model.1.norm.running_var',
                                     'dec.model.0.model.1.model.0.norm.running_mean',
                                     'dec.model.0.model.1.model.0.norm.running_var',
                                     'dec.model.0.model.2.model.1.norm.running_mean',
                                     'dec.model.0.model.2.model.1.norm.running_var',
                                     'dec.model.0.model.2.model.0.norm.running_mean',
                                     'dec.model.0.model.2.model.0.norm.running_var',
                                     'dec.model.0.model.3.model.1.norm.running_mean',
                                     'dec.model.0.model.3.model.1.norm.running_var',
                                     'dec.model.0.model.3.model.0.norm.running_mean',
                                     'dec.model.0.model.3.model.0.norm.running_var',
                                     )):
                        del state_dict[key]
        return state_dict

    state_dict = dict()
    state_dict['a'] = __conversion_core(state_dict_base['a'], trainer_name)
    state_dict['b'] = __conversion_core(state_dict_base['b'], trainer_name)
    return state_dict