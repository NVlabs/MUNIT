"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from torch.utils.serialization import load_lua
from torch.utils.data import DataLoader
from networks import Vgg16
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
from data import ImageFilelist, ImageFolder
import torch
import os
import math
import torchvision.utils as vutils
import yaml
import numpy as np
import torch.nn.init as init

def get_all_data_loaders(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    new_size = conf['new_size']
    height = conf['crop_image_height']
    width = conf['crop_image_width']
    train_loader_a = get_data_loader_list(conf['root_a'], conf['train_list_a'], batch_size, True,
                                          conf['input_dim_a'] == 1, new_size, height, width, num_workers)
    test_loader_a = get_data_loader_list(conf['root_a'], conf['test_list_a'], batch_size, False,
                                         conf['input_dim_a'] == 1, new_size, height, width, num_workers)
    train_loader_b = get_data_loader_list(conf['root_b'], conf['train_list_b'], batch_size, True,
                                          conf['input_dim_b'] == 1, new_size, height, width, num_workers)
    test_loader_b = get_data_loader_list(conf['root_b'], conf['test_list_b'], batch_size, False,
                                         conf['input_dim_b'] == 1, new_size, height, width, num_workers)
    return train_loader_a, train_loader_b, test_loader_a, test_loader_b

def get_data_loader_list(root, file_list, batch_size, train, gray_scale, new_size=256,
                           height=256, width=256, num_workers=4, crop=True):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.Resize(new_size),
                      transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Grayscale()] + transform_list if gray_scale else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFilelist(root, file_list, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader

def get_data_loader_folder(input_folder, batch_size, train, gray_scale, new_size=256,
                           height=256, width=256, num_workers=4, crop=True):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.Resize(new_size),
                      transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Grayscale()] + transform_list if gray_scale else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFolder(input_folder, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader

def get_config(config):
    # type: (str) -> dict
    stream = open(config, 'r')
    docs = yaml.load_all(stream)
    for doc in docs:
        for k, v in doc.items():
            if k == "train":
                return v
    stream.close()

def eformat(f, prec):
    s = "%.*e"%(prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%d"%(mantissa, int(exp))

def write_images(image_outputs, display_image_num, file_name):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs] # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_tensor = image_tensor.data
    image_grid = vutils.make_grid(image_tensor, nrow=display_image_num, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)

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

def write_html(filename, iterations, image_save_iterations, image_directory, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = UnitNet</title>
      <meta content="1" http-equiv="reflesh">
    </head>
    <body>
    ''')
    html_file.write("<h3>current</h3>")
    img_filename = '%s/gen.jpg' % (image_directory)
    html_file.write("""
          <p>
          <a href="%s">
            <img src="%s" style="width:%dpx">
          </a><br>
          <p>
          """ % (img_filename, img_filename, all_size))
    for j in range(iterations, image_save_iterations-1, -1):
        if j % image_save_iterations == 0:
            img_filename = '%s/gen_test%08d.jpg' % (image_directory, j)
            html_file.write("<h3>iteration [%d] (test)</h3>" % j)
            html_file.write("""
                  <p>
                  <a href="%s">
                    <img src="%s" style="width:%dpx">
                  </a><br>
                  <p>
                  """ % (img_filename, img_filename, all_size))
            img_filename = '%s/gen_train%08d.jpg' % (image_directory, j)
            html_file.write("<h3>iteration [%d] (train)</h3>" % j)
            html_file.write("""
                  <p>
                  <a href="%s">
                    <img src="%s" style="width:%dpx">
                  </a><br>
                  <p>
                  """ % (img_filename, img_filename, all_size))
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
    '"""
    omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high

def get_slerp_interp(nb_latents, nb_interp, z_dim):
    """
    modified from: PyTorch inference for "Progressive Growing of GANs" with CelebA snapshot

    https://github.com/ptrblck/prog_gans_pytorch_inference
    '"""

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

def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size())
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
                init.normal(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)

    return init_fun

def instancenorm(x):
    n = x.size(2) * x.size(3)
    t = x.view(x.size(0), x.size(1), n)
    mean = torch.mean(t, dim=2).unsqueeze(2).unsqueeze(2).expand_as(x)
    # Calculate the biased var. torch.var returns unbiased var
    var = torch.var(t, dim=2).unsqueeze(2).unsqueeze(2).expand_as(x) * ((n - 1) / float(n))
    out = (x - mean) / torch.sqrt(var + 1e-9)
    return out
