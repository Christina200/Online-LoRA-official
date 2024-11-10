import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from datasets import build_continual_dataloader
from engine import *
import utils
from lora import LoRA_ViT_timm, LoRA_Swin_timm

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):

    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    set_seed(seed)

    cudnn.benchmark = True

    data_loader, class_mask = build_continual_dataloader(args)

    # Create training model
    vit_model = create_model(args.model, pretrained=args.pretrained)
    # Choose Transformer: ViT or Swin Transformer
    lora_model = LoRA_ViT_timm(vit_model=vit_model, r=args.lora_rank, num_classes=args.nb_classes)
    # lora_model = LoRA_Swin_timm(swin_model=vit_model, r=args.lora_rank, num_classes=args.nb_classes)

    net = lora_model.to(device)
    model = torch.nn.DataParallel(net)

    n_params = sum(p.numel() for p in lora_model.parameters())
    print(f"Total Parameters :\t{n_params}")
    n_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"Learnable Parameters :\t{n_params}")

    print(args)

    optimizer = torch.optim.Adam(model.module.parameters(), lr=args.lr) 
    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model=model.module, criterion=criterion, data_loader=data_loader, optimizer=optimizer, device=device, args=args, class_mask=class_mask)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Online LoRA training and evaluation configs')
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_online_lora':
        from configs.cifar100_online_lora import get_args_parser
        config_parser = subparser.add_parser('cifar100_online_lora', help='Split-CIFAR100 Online LoRA configs')
    elif config == 'core50_online_lora':
        from configs.core50_online_lora import get_args_parser
        config_parser = subparser.add_parser('core50_online_lora', help='CORe50 Online LoRA configs')
    elif config == 'imagenetR_online_lora':
        from configs.imagenetR_online_lora import get_args_parser
        config_parser = subparser.add_parser('imagenetR_online_lora', help='Imagenet-R Online LoRA configs')
    elif config == 'sketch_online_lora':
        from configs.sketch_online_lora import get_args_parser
        config_parser = subparser.add_parser('sketch_online_lora', help='Sketch Online LoRA configs')
    elif config == 'cub200_online_lora':
        from configs.cub200_online_lora import get_args_parser
        config_parser = subparser.add_parser('cub200_online_lora', help='CUB200 Online LoRA configs')
    else:
        raise NotImplementedError
    
    get_args_parser(config_parser)

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

    sys.exit(0)