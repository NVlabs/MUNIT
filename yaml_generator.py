from utils import get_config, eformat
import yaml

recon_x_w_list = [10, 100]
recon_c_w_list = [1, 10]
recon_s_w_list = [1]
vgg_w_list = [0, 1e-1, 1]
upsample_norm_list = ['ln']

configs = get_config('configs/summer2winter-hd.yaml')
for recon_x_w in recon_x_w_list:
    for recon_c_w in recon_c_w_list:
        for recon_s_w in recon_s_w_list:
            for upsample_norm in upsample_norm_list:
                for vgg_w in vgg_w_list:
                    configs['recon_x_w'] = recon_x_w
                    configs['recon_c_w'] = recon_c_w
                    configs['recon_s_w'] = recon_s_w
                    configs['vgg_w'] = vgg_w
                    configs['init_dis'] = 0
                    configs['gen']['upsample_norm'] = upsample_norm
                    # configs['gen']['pad_type'] = 'zero'
                    configs['dis']['pad_type'] = 'reflect'
                    configs['dis']['norm'] = 'lnna'
                    configs['dis']['num_scales'] = 2
                    file_name = 'configs/configs/summer2winter-hd-refpad2sdislnnagaussian-dec{}-vgg{}-reconx{}-reconc{}-recons{}.yaml'\
                        .format(upsample_norm,
                                eformat(vgg_w, 0),
                                eformat(recon_x_w, 0),
                                eformat(recon_c_w, 0),
                                eformat(recon_s_w, 0))
                    with open(file_name, 'w') as outfile:
                        yaml.dump(configs, outfile, default_flow_style=False)