import os
import yaml
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn


def _load_pickle(path):
    with open(path, 'rb') as file_obj:
        return pickle.load(file_obj)

def parse_configure(model=None, dataset=None):
    parser = argparse.ArgumentParser(description='RLMRec')
    parser.add_argument('--model', type=str, default='LightGCN', help='Model name')
    parser.add_argument('--dataset', type=str, default='amazon', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=None, help='Device number')
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    args, _ = parser.parse_known_args()

    # cuda
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # model name
    if model is not None:
        model_name = model.lower()
    elif args.model is not None:
        model_name = args.model.lower()
    else:
        model_name = 'default'
        # print("Read the default (blank) configuration.")

    # dataset
    if dataset is not None:
        args.dataset = dataset

    # find yml file
    if not os.path.exists('./config/modelconf/{}.yml'.format(model_name)):
        raise Exception(f"Please create the yaml file for your model {model_name} first.")

    # read yml file
    with open('./config/modelconf/{}.yml'.format(model_name), encoding='utf-8') as f:
        config_data = f.read()
        configs = yaml.safe_load(config_data)
        configs['model']['name'] = configs['model']['name'].lower()
        if 'tune' not in configs:
            configs['tune'] = {'enable': False}
        configs['device'] = args.device
        if args.dataset is not None:
            configs['data']['name'] = args.dataset
        if args.seed is not None:
            configs['train']['seed'] = args.seed

        # semantic embeddings
        usrprf_embeds_path = "../data/{}/usr_emb_np.pkl".format(configs['data']['name'])
        itmprf_embeds_path = "../data/{}/itm_emb_np.pkl".format(configs['data']['name'])
        configs['usrprf_embeds'] = _load_pickle(usrprf_embeds_path)
        configs['itmprf_embeds'] = _load_pickle(itmprf_embeds_path)

        # multi-profile embeddings (optional, for ProEx-style models)
        usr_multi_path = "../data/{}/usr_multi_emb_np.pkl".format(configs['data']['name'])
        itm_multi_path = "../data/{}/itm_multi_emb_np.pkl".format(configs['data']['name'])
        if os.path.exists(usr_multi_path):
            configs['usrprf_multi_embeds'] = _load_pickle(usr_multi_path)
        if os.path.exists(itm_multi_path):
            configs['itmprf_multi_embeds'] = _load_pickle(itm_multi_path)

        return configs

configs = parse_configure()
