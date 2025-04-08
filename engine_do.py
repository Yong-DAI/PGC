
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
import time
from typing import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
from torch.nn import MSELoss

from timm.utils import accuracy
from timm.optim import create_optimizer
import copy
import utils
from torch.distributions.multivariate_normal import MultivariateNormal
import logging


from timm.scheduler import create_scheduler
""
from torchvision import  transforms
from torch.utils.data import Dataset
from PIL import Image

class DummyDataset(Dataset):
    def __init__(self, images,targets, trsf, use_path=False):
        assert len(images) == len(targets), 'Data size error!'
        self.images = images
        self.labels = targets
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        # return idx, image, label
        return  image, label              ### dy

def pil_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        scale = (0.08, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    
    return transforms.Compose(t)

""
def train_one_epoch(model: torch.nn.Module, 
                    criterion, data_loader: Iterable, 
                    device: torch.device, epoch: int, max_norm: float = 0,
                    optimizer=None,
                    old_prompt_matcher = None,
                    old_dap_downsample = None,
                    old_prompt = None,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,
                    old_num_k=5,):

    model.train(set_training_mode)

    s = old_num_k

    # Freezing previous tasks' filters
    for name, param in model.named_parameters():
        if name.find('e_prompt.v_conv_vals') >=0  or name.find('e_prompt.k_conv_vals') >=0:
            for i in range(s):
                if name.find('.{}.weight'.format(i)) >=0 or name.find('.{}.bias'.format(i)) >=0:
                    param.requires_grad = False


    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    
    # metric_logger.add_meter('Lr_head', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    if args.SLCA:
        metric_logger.add_meter('Lr_cls', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Lr_rps', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    else:
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    

    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)
        target = (target+torch.ones_like(target)*345*task_id).to(device, non_blocking=True)

        output = model(input, task_id=task_id, train=set_training_mode)
        logits = output['logits']

        # Masking and computing loss
        known_classes = task_id*len(class_mask[0])
        cur_targets = torch.where(target-known_classes>=0,target-known_classes,-100)
        loss = criterion(logits[:, known_classes:], cur_targets) # base criterion (CrossEntropyLoss)


        if args.use_e_prompt or args.use_g_prompt:
            if task_id > 0:
                # l1_loss = 0.0
                # for old_wt, new_wt in zip(old_prompt_matcher.parameters(), model.e_prompt.prompt_embed_matcher.parameters()):               ##################
                #     l1_loss += torch.norm(old_wt.detach() - new_wt, p=1)
                # for old_wt_, new_wt_ in zip(old_dap_downsample.parameters(), model.dap_downsample.parameters()):
                #     l1_loss += torch.norm(old_wt_.detach() - new_wt_, p=1)
                    
                # loss = loss + 0.01 * l1_loss                      
                
                prompt_loss = torch.norm(old_prompt.detach() - model.e_prompt.prompt, p=1)
                loss = loss + 0.01 * prompt_loss


        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        if args.use_clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        
        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        if args.SLCA:
            metric_logger.update(Lr_cls=optimizer.param_groups[0]["lr"])
            metric_logger.update(Lr_rps=optimizer.param_groups[1]["lr"]) 
        else:
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    logging.info("Averaged stats: {}".format(metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

""
def accuracy_dil(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    # pred = np.remainder(pred, np.ones_like(pred)*50)         ####  dy
    pred =pred%345
    target =target%345
    
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


""

@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, args=None,):
    
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    correct = 0
    total = 0

    # Batchwise Eval time
    start_eval_time = time.time()
    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            output = model(input, task_id=task_id)
            logits = output['logits']

            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)
            # print('Loss')
            predicts = torch.max(logits, dim=1)[1]

            ###########
            predicts =predicts%345
            target =target%345
            #############

            correct += (predicts == target).sum()
            total += len(target)
            # acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            acc1, acc5 = accuracy_dil(logits, target, topk=(1, 5))  #### dy

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    end_eval_time = time.time()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))
    logging.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))
    
    print(f"Batchwise eval time for task {task_id+1} = {(end_eval_time - start_eval_time)/len(data_loader)}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, total, correct


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss

    total, correct = 0, 0
    for i in range(task_id+1):
        logging.info('Evaluating task {}...'.format(i+1))
        test_stats, temp_total, temp_correct = evaluate(model=model, data_loader=data_loader, 
                            device=device, task_id=i, class_mask=class_mask, args=args,)

        total += temp_total
        correct += temp_correct
        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    final_acc = np.divide(correct.cpu(), total)*100.0
    result_str = "[Average accuracy till task{}]\tfinalAcc: {:.4f}\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, final_acc,avg_stat[0], avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)
    logging.info(result_str)
    
    path = args.output_dir+'_'+args.dataset
    Path(os.path.join(path, 'checkpoint')).mkdir(parents=True, exist_ok=True)
    
    with open(os.path.join(path, 'checkpoint/{}_average.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
        f.write(result_str + '\n')

    return test_stats


def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module,
                    criterion, data_loader: Iterable, lr_scheduler, optimizer, device: torch.device, 
                    class_mask=None, args = None,):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    old_num_k = 0

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)
    
#     train_trsf = [        transforms.RandomResizedCrop(224),        transforms.RandomHorizontalFlip(),    ]
#     test_trsf = [        transforms.Resize(256),        transforms.CenterCrop(224),    ]
#     common_trsf = [        transforms.ToTensor(),        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    ]
#     transform_train = transforms.Compose([*train_trsf, *common_trsf])
#     transform_val = transforms.Compose([*test_trsf, *common_trsf])
    
    domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch", ]
    image_list_root = args.data_path
    image_list_paths = [os.path.join(image_list_root, d + "_" + "test" + ".txt") for d in domain_names]
    imgs = []
    for taskid, image_list_path in enumerate(image_list_paths):
        image_list = open(image_list_path).readlines()
        imgs += [(val.split()[0], int(val.split()[1])) for val in image_list]
#         imgs += [(val.split()[0], int(val.split()[1])+ taskid * 345) for val in image_list]
    train_x, train_y = [], []
    for item in imgs:
        train_x.append(os.path.join(image_list_root, item[0]))
        train_y.append(item[1])
    test_img = np.array(train_x)
    test_target = np.array(train_y)
    
    
    test_dataset = DummyDataset( test_img,test_target,  transform_val, True)
        
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers)
     
    image_list_paths = [os.path.join(image_list_root, d + "_" + "train" + ".txt") for d in domain_names]
    ########
    labels = [i for i in range(345*6)]

    class_mask = list()

    for _ in range(6):

        scope = labels[:345]
        labels = labels[345:]

        class_mask.append(scope)
    ########
#     for task_id in range(args.num_tasks):
    
    if args.eval:
        task_id = 1
        with torch.no_grad():
            model.e_prompt.maben.grad.zero_()
            task_key_norm[task_id,:] = model.e_prompt.maben
        
        # np.save('./codebook/codebook_%s.npy'%task_id, task_key_norm.cpu().numpy())
        
        # test_stats, pred_vs_gt_te = evaluate_till_now(model=model, original_model=original_model, data_loader=test_loader, device=device, 
        #                                 task_id=args.num_tasks-1,task_key_norm= task_key_norm, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
        test_stats = evaluate_till_now(model=model, data_loader=test_loader, device=device, 
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
        return

    # for task_id in range(args.num_tasks):
    for task_id, image_list_path in enumerate(image_list_paths):
        imgs = []
        print('loading training data...%s'%task_id)
        image_list = open(image_list_path).readlines()
        imgs += [(val.split()[0], int(val.split()[1]) ) for val in image_list]
        
        train_x, train_y = [], []
        for item in imgs:
            train_x.append(os.path.join(image_list_root, item[0]))
            train_y.append(item[1])
        train_img = np.array(train_x)
        train_target = np.array(train_y)
        
        train_dataset = DummyDataset(train_img,train_target,  transform_train, True)                      ############## .astype(np.uint8)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.num_workers)
        
#         class_mask_dy = [i for i in range(task_id*345,(task_id+1)*345)] 
        ##############################
        if task_id>0:
            model.head.update(len(class_mask[task_id]))
     
        # Create new optimizer for each task to clear optimizer status
        not_n_params = []
        n_params = []
        if args.SLCA:
            milestones = [18] if "CIFAR" in args.dataset else [40]

        lrate_decay = 0.1
        param_list = list(model.parameters())
        if task_id:
            for n, p in model.named_parameters():                
                if n.find('norm1')>=0 or n.find('norm2') >= 0 or n.startswith('norm') or n.find('fc_norm') >= 0:
                    # print(f'Param: {n} Param.requires_grad: {p.requires_grad}')
                    n_params.append(p)
                else:
                    not_n_params.append(p)
            
            network_params = [{'params': not_n_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
                                {'params': n_params, 'lr': 0.005*args.lr, 'weight_decay': args.weight_decay}] 
        else:
            network_params = [{'params': param_list, 'lr': args.lr, 'weight_decay': args.weight_decay}]
        
        if not args.SLCA:
            print("Using adam optimizer")
            print("Reinitialising optimizer")
            optimizer = optim.Adam(network_params, weight_decay=args.weight_decay)
            if args.sched != 'constant':
                # lr_scheduler, _ = create_scheduler(args, optimizer)
                # Create cosine lr scheduler
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
            elif args.sched == 'constant':
                lr_scheduler = None

        else:
            optimizer = optim.SGD(network_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)
        

        if args.use_e_prompt or args.use_g_prompt:
            old_prompt = copy.deepcopy(model.e_prompt.prompt.clone().detach())
            old_prompt_matcher = copy.deepcopy(model.e_prompt.prompt_embed_matcher)
            # old_dap_downsample = copy.deepcopy(model.dap_downsample)
            old_dap_downsample = None

            curr_num_k = 4

            model.e_prompt.process_new_task(old_num_k, old_num_k + curr_num_k)
        else:
            curr_num_k = 0
            old_prompt_matcher = None
            old_prompt = None
        
        print("Task number: ", task_id+1)
        ################                                                                                         ###########################
        if task_id >= 1:
            for k, p in model.named_parameters():
                if "dap_downsample" in k:                       ############  model.e_prompt.prompt_embed_matcher    or "prompt_embed_matcher" 
                    p.requires_grad = False
            for k, p in model.e_prompt.named_parameters():
                if "prompt_embed_matcher" in k:                       ############  model.e_prompt.prompt_embed_matcher    or "prompt_embed_matcher" ,   SE:   model.e_prompt.prompt
                    p.requires_grad = False
                    
        for epoch in range(args.epochs):     
            logging.info('Training for task {} epoch {}/{}'.format(task_id, epoch, args.epochs))     
            train_stats = train_one_epoch(model=model, criterion=criterion, 
                                        data_loader=train_loader, 
                                        optimizer=optimizer, 
                                        device=device, epoch=epoch, max_norm=args.clip_grad,
                                        old_prompt_matcher=old_prompt_matcher,
                                        old_dap_downsample = old_dap_downsample,
                                        old_prompt=old_prompt,
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,
                                        old_num_k=old_num_k,)
            
            if lr_scheduler:
                lr_scheduler.step()


        if args.use_e_prompt or args.use_g_prompt:
            old_num_k += curr_num_k
            
        if task_id >= 5:              #############
            eval_start_time = time.time()
        
            test_stats = evaluate_till_now(model=model, data_loader=test_loader, device=device, 
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
            eval_end_time = time.time()
            print(f"Eval time for task {task_id+1} = {eval_end_time - eval_start_time}")


            if args.output_dir and utils.is_main_process():
                path = args.output_dir+'_'+args.dataset
                Path(os.path.join(path, 'checkpoint')).mkdir(parents=True, exist_ok=True)


                checkpoint_path = os.path.join(path, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
                state_dict = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }
                if args.sched is not None and args.sched != 'constant':
                    state_dict['lr_scheduler'] = lr_scheduler.state_dict()

                model.eval()   #########
                utils.save_on_master(state_dict, checkpoint_path)


            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,}

            if args.output_dir and utils.is_main_process():
                with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                    f.write(json.dumps(log_stats) + '\n')
            
            
        ######################
#         test_stats = evaluate_till_now(model=model, data_loader=test_loader, device=device, 
#                                     task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
#         eval_end_time = time.time()
#         print(f"Eval time for task {task_id+1} = {eval_end_time - eval_start_time}")

        
#         if args.output_dir and utils.is_main_process():
#             path = args.output_dir+'_'+args.dataset
#             Path(os.path.join(path, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
            
#             checkpoint_path = os.path.join(path, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
#             state_dict = {
#                     'model': model_without_ddp.state_dict(),
#                     'optimizer': optimizer.state_dict(),
#                     'epoch': epoch,
#                     'args': args,
#                 }
#             if args.sched is not None and args.sched != 'constant':
#                 state_dict['lr_scheduler'] = lr_scheduler.state_dict()
                
#             model.eval()   #########
#             utils.save_on_master(state_dict, checkpoint_path)
            

#         log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
#             **{f'test_{k}': v for k, v in test_stats.items()},
#             'epoch': epoch,}

#         if args.output_dir and utils.is_main_process():
#             with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
#                 f.write(json.dumps(log_stats) + '\n')
