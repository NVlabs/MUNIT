"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, \
    write_html, write_loss, get_config, write_images
from optparse import OptionParser
from torch.autograd import Variable
from itertools import izip
from trainer import MUNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
import os
import sys
from tensorboardX import SummaryWriter
import shutil
parser = OptionParser()
parser.add_option('--config', type=str, help="net configuration")
parser.add_option('--log', type=str, default='logs', help="log path")
parser.add_option('--outputs', type=str, default='outputs', help="outputs path")
parser.add_option("--resume", action="store_true")


def main(argv):
    (opts, args) = parser.parse_args(argv)
    cudnn.benchmark = True
    model_name = os.path.splitext(os.path.basename(opts.config))[0]

    # Load experiment setting
    config = get_config(opts.config)
    max_iter = config['max_iter']
    display_size = config['display_size']

    # Setup model and data loader
    trainer = MUNIT_Trainer(config)
    trainer.cuda()
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
    test_display_images_a = Variable(torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).cuda(), volatile=True)
    test_display_images_b = Variable(torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).cuda(), volatile=True)
    train_display_images_a = Variable(torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).cuda(), volatile=True)
    train_display_images_b = Variable(torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda(), volatile=True)

    # Setup logger and output folders
    train_writer = SummaryWriter(os.path.join(opts.log, model_name))
    output_directory = os.path.join(opts.outputs, model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

    # Start training
    iterations = trainer.resume(checkpoint_directory) if opts.resume else 0
    while True:
        for it, (images_a, images_b) in enumerate(izip(train_loader_a, train_loader_b)):
            trainer.update_learning_rate()
            images_a, images_b = Variable(images_a.cuda()), Variable(images_b.cuda())

            # Main training code
            trainer.dis_update(images_a, images_b, config)
            trainer.gen_update(images_a, images_b, config)

            # Dump training stats in log file
            if (iterations + 1) % config['log_iter'] == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, trainer, train_writer)

            # Write images
            if (iterations + 1) % config['image_save_iter'] == 0:
                # Test set images
                image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                write_images(image_outputs, display_size, '%s/gen_test%08d.jpg' % (image_directory, iterations + 1))
                # Train set images
                image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                write_images(image_outputs, display_size, '%s/gen_train%08d.jpg' % (image_directory, iterations + 1))
                # HTML
                write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')
            if (iterations + 1) % config['image_save_iter'] == 0:
                image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                write_images(image_outputs, display_size, '%s/gen.jpg' % image_directory)

            # Save network weights
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(checkpoint_directory, iterations)

            iterations += 1
            if iterations >= max_iter:
                return


if __name__ == '__main__':
    main(sys.argv)
