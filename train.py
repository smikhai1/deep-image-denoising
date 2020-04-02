import pydoc
from src.training import train
from src.utils import get_config, parse_args
from src.data_factory import make_ct_datasets, make_bsd_datasets
import torch
from torch import nn


def main():
    args = parse_args()
    configs = get_config(args.config)
    paths = get_config(args.paths)

    print(f'Configs\n{configs}\n')
    print(f'Paths\n{paths}\n')

    ####### DATA ######
    train_loader, val_loader = make_ct_datasets(configs, paths)

    ####### MODEL ######
    model = pydoc.locate(configs['train_params']['model'])()
    model_name = configs['train_params']['model_name']

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)
    print(f'Current device: {device}')

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f'Number of CUDA devices: {torch.cuda.device_count()}')

    try:
        pretrained = configs['train_params']['pretrained']
        if pretrained:
            model_dumps = torch.load(configs['train_params']['path_weights'], map_location=device)
            model.load_state_dict(model_dumps['model_state_dict'])
            print(f'Weights loaded from model {configs["train_params"]["path_weights"]}')
    except KeyError:
        print('A parameter wasn`t found in the config file')

    ####### OPTIMIZER ######
    optimizer_name = configs['train_params']['optimizer']
    optimizer = pydoc.locate('torch.optim.' + optimizer_name)(model.parameters(),
                                                              **configs['train_params']['optimizer_params'])
    ####### SCHEDULER ######
    scheduler_name = configs['train_params']['scheduler']
    scheduler = pydoc.locate('torch.optim.lr_scheduler.' + scheduler_name)(optimizer,
                                                                           **configs['train_params']['scheduler_params'])
    ####### CRITERION ######
    loss = pydoc.locate(configs['train_params']['loss'])()

    ####### TRAINING ######
    max_epoch = int(configs['train_params']['max_epoch'])

    train(model, optimizer, loss, train_loader, max_epoch, device, val_loader,
          scheduler=scheduler, weights_path=paths['dumps']['weights'], model_name=model_name)

if __name__ == '__main__':
    main()