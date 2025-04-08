
import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from torch import optim
import logging

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from datasets import build_continual_dataloader
# from engine import *
from engine_do import *
import models
import utils

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

#####################
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True
    ###########
    labels = [i for i in range(345*6)]

    class_mask = list()

    for _ in range(6):

        scope = labels[:345]
        labels = labels[345:]

        class_mask.append(scope)
    ##########
    data_loader= torch.zeros(1,1)                   ######################
    args.nb_classes = 345*6
    
    print("NB CLasses: ", args.nb_classes)
    # print("class_mask: ", class_mask)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        num_tasks=args.num_tasks,
        kernel_size=args.kernel_size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
        use_g_prompt=args.use_g_prompt,
        g_prompt_length=args.g_prompt_length,
        g_prompt_layer_idx=args.g_prompt_layer_idx,
        use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
        use_e_prompt=args.use_e_prompt,
        e_prompt_layer_idx=args.e_prompt_layer_idx,
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
        same_key_value=args.same_key_value,
        prompts_per_task=args.num_prompts_per_task,
        args=args
    )
    model.to(device)  

    if args.freeze:
        
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                if n.find('norm1')>=0 or n.find('norm2')>=0:
                    # print(n)
                    pass
                else:
                    p.requires_grad = False
            #         print(n)

        # exit(0)
        
    
    print('args is loaded ')
    # print(model)

    if args.eval:
        
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        path = args.output_dir+'_'+args.dataset

#         checkpoint_path = os.path.join(path, 'checkpoint-fixproj-89.33/task1_checkpoint.pth')
#         if os.path.exists(checkpoint_path):
#             print('Loading checkpoint from:', checkpoint_path)
#             ckpt = torch.load(checkpoint_path)
#             # model.load_state_dict(ckpt['model'])


#         for k, v in ckpt.items():
#             if k == 'model':
#                 print(ckpt[k].keys())

#         return

        for task_id in range(args.num_tasks):
            
            if task_id>0:
                model.head.update(len(class_mask[task_id]))
            
            checkpoint_path = os.path.join(path, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            _ = evaluate_till_now(model, data_loader, device, 
                                            task_id, class_mask, acc_matrix, args,)
        
        return

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0


    criterion = torch.nn.CrossEntropyLoss().to(device)

    milestones = [18] if "CIFAR" in args.dataset else [40]
    lrate_decay = 0.1
    param_list = list(model.parameters())
 

    network_params = [{'params': param_list, 'lr': args.lr, 'weight_decay': args.weight_decay}]
    
    if not args.SLCA:
        optimizer = create_optimizer(args, model)
        if args.sched != 'constant':
            # lr_scheduler, _ = create_scheduler(args, optimizer)
            # Create cosine lr scheduler
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
        elif args.sched == 'constant':
            lr_scheduler = None
    else:
        optimizer = optim.SGD(network_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model,model_without_ddp,
                    criterion, data_loader, lr_scheduler, optimizer,
                    device, class_mask, args)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    print("Started main")
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    print("Parser created: ", parser)
    
    print("Getting config")
    # config = parser.parse_known_args()[-1][0]
#     config = 'cifar100_pgt'
#     config = 'imr_pgt'
    config = 'domainnet_pgt'

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_pgt':
        from configs.cifar100_pgt import get_args_parser
        config_parser = subparser.add_parser('cifar100_pgt', help='Split-CIFAR100 configs for pgt')
    elif config == 'imr_pgt':
        from configs.imr_pgt import get_args_parser
        config_parser = subparser.add_parser('imr_pgt', help='Split-ImageNet-R configs for pgt')
    elif config == 'domainnet_pgt':
        from configs.domainnet_pgt import get_args_parser
        config_parser = subparser.add_parser('domainnet_pgt', help='domainnet configs')
    else:
        raise NotImplementedError
        
    get_args_parser(parser)

    # print("Reached here")
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # print("Reached here")
    main(args)
    
    sys.exit(0)
