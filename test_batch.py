"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, get_data_loader_folder
from trainer import MUNIT_Trainer
from optparse import OptionParser
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os

parser = OptionParser()
parser.add_option('--config', type=str, help="net configuration")
parser.add_option('--input_folder', type=str, help="input image folder")
parser.add_option('--output_folder', type=str, help="output image folder")
parser.add_option('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_option('--a2b', type=int, help="1 for a2b and others for b2a", default=1)
parser.add_option('--seed', type=int, default=1, help="random seed")
parser.add_option('--num_style',type=int, default=10, help="number of styles to sample")
parser.add_option('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_option('--output_only', action='store_true', help="whether use synchronized style code or not")

def main(argv):
    (opts, args) = parser.parse_args(argv)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    if not os.path.exists(opts.output_folder):
        os.makedirs(opts.output_folder)

    # Load experiment setting
    config = get_config(opts.config)
    input_dim = config['new_size'] if opts.a2b else config['input_dim_b']
    style_dim = config['gen']['style_dim']

    # Setup model and data loader
    data_loader = get_data_loader_folder(opts.input_folder, 1, False, input_dim == 1, crop=False)
    trainer = MUNIT_Trainer(config,opts)
    state_dict = torch.load(opts.checkpoint)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
    trainer.cuda()
    trainer.eval()
    encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode # encode function
    decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode # decode function

    # Start testing
    style_fixed = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(), volatile=True)
    for i, images in enumerate(data_loader):
        images = Variable(images.cuda(), volatile=True)
        content, _ = encode(images)
        style = style_fixed if opts.synchronized else Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(), volatile=True)
        for j in range(opts.num_style):
            s = style[j].unsqueeze(0)
            outputs = decode(content, s)
            outputs = (outputs + 1) / 2.
            path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
        if not opts.output_only:
            # also save input images
            vutils.save_image(images.data, os.path.join(opts.output_folder, 'input{:03d}.jpg'.format(i)), padding=0, normalize=True)

if __name__ == '__main__':
    main(sys.argv)
