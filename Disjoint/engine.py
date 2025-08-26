import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch

import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer
from torch.nn import functional as F
import logging

import utils

@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader, device, task_id=-1, args=None):

    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            logits = model.forward(input)
            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, data_loader, device, task_id=-1, acc_matrix=None, args=None):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id+1):
        test_stats = evaluate(model=model, data_loader=data_loader[i]['val'], device=device, task_id=i, args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    if args.all_metrics and task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats

def train_and_evaluate(model: torch.nn.Module, criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, args=None, class_mask=None):
    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    count_updates=0
    batch=[]
    hard_buffer=[]
    # loss dectection
    loss_window=[]
    loss_window_means=[]
    loss_window_variances=[]
    new_peak_detected=True
    omega_As=[] 
    omega_Bs=[]
    ckpt_path = utils.init_ckpt_path(loglevel=logging.ERROR)

    for task_id in range(args.num_tasks):
        for _, data in enumerate(data_loader[task_id]['train']):
                inputs, labels = data
                inputs, labels = (inputs.to(device), labels.to(device))

                # Accumulate batch
                batch.append({'state':inputs, 'trgt':labels})
                if len(batch) > args.nb_batch:
                    del batch[0]
                if len(batch) == args.nb_batch:
                    # Get training data (batch+hard buffer)
                    x=[_['state'] for _ in batch]
                    y=[_['trgt'] for _ in batch]
                    
                    if len(hard_buffer) != 0:
                        xh=[_['state'] for _ in hard_buffer]
                        yh=[_['trgt'] for _ in hard_buffer]
                    for epoch in range(args.epochs): 
                        metric_logger = utils.MetricLogger(delimiter="  ")
                        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
                        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
                        header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'

                        total_loss = (torch.tensor(0.0)).to(device)
                        # Current batch loss
                        current_loss = []
                        for image, label in zip(x, y): 
                            y_pred = model.forward(image) 
                            current_loss.append(criterion(y_pred, label))
                            total_loss += criterion(y_pred, label)
                        
                        # Hard buffer loss
                        hard_loss = []
                        if len(hard_buffer) != 0:
                            # evaluate hard buffer
                            for image_h, label_h in zip(xh, yh): 
                                yh_pred = model.forward(image_h) 
                                hard_loss.append(criterion(yh_pred,label_h))
                                if (args.hard_loss): 
                                    total_loss += criterion(yh_pred,label_h)
                        
                        # keep train loss for loss window
                        if epoch==0: 
                            first_train_loss=total_loss.detach().cpu().numpy()
                            # print('first train loss (for loss window): {0:0.3f}'.format(first_train_loss))
                        
                        wnew_a_params = filter(lambda p: getattr(p, '_is_wnew_a', False), model.parameters())
                        wnew_b_params = filter(lambda p: getattr(p, '_is_wnew_b', False), model.parameters())

                        # Regularization loss
                        if len(omega_As)!=0 and len(omega_As)==len(omega_Bs): # omega_As and omega_Bs should have same length. 
                            mas_loss = 0.
                            for pindex, (p_a, p_b) in enumerate(zip(wnew_a_params, wnew_b_params)):
                                if isinstance(omega_As[pindex], np.ndarray):
                                    product_a = torch.from_numpy(omega_As[pindex]).type(torch.float32).to(device) * ((p_a) ** 2)
                                    product_b = torch.from_numpy(omega_Bs[pindex]).type(torch.float32).to(device) * ((p_b) ** 2)
                                    mas_loss += torch.sum(product_a) + torch.sum(product_b) 
                            print('MAS loss: {}'.format(mas_loss))
                            if (args.regularization): 
                                total_loss+=args.MAS_weight/2.*mas_loss

                        optimizer.zero_grad()
                        torch.sum(total_loss).backward()
                        optimizer.step()

                        metric_logger.update(Loss=total_loss.item())
                        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
                        print("Averaged stats:", metric_logger)

                    # save training accuracy on total batch
                    if len(hard_buffer) != 0:
                        xt=x+xh
                        yt=y+yh
                    else:
                        xt=x[:]
                        yt=y[:]

                    # Update loss_window and detect loss plateaus
                    loss_window.append(np.mean(first_train_loss))
                    if len(loss_window)>args.loss_window_length: del loss_window[0]
                    loss_window_mean=np.mean(loss_window)
                    loss_window_variance=np.var(loss_window)
                    print('loss window mean: {0:0.3f}, loss window variance: {1:0.3f}'.format(loss_window_mean, loss_window_variance))
                    # Check the statistics of the current window
                    if not new_peak_detected and loss_window_mean > last_loss_window_mean+np.sqrt(last_loss_window_variance) :
                        new_peak_detected=True  
                    # Time for updating importance weights    
                    if loss_window_mean < args.loss_window_mean_threshold and loss_window_variance < args.loss_window_variance_threshold and new_peak_detected:
                        count_updates+=1
                        print('importance weights update')
                        last_loss_window_mean=loss_window_mean
                        last_loss_window_variance=loss_window_variance
                        new_peak_detected=False
                        
                        # calculate imporatance based on each sample in the hardbuffer
                        gradients_A = [0 for p in model.parameters() if getattr(p, '_is_wnew_a', False)]
                        gradients_B = [0 for p in model.parameters() if getattr(p, '_is_wnew_b', False)]
                        
                        model.eval()
                        wnew_a_params = filter(lambda p: getattr(p, '_is_wnew_a', False), model.parameters())
                        wnew_b_params = filter(lambda p: getattr(p, '_is_wnew_b', False), model.parameters())
                        for sx in [_['state'] for _ in hard_buffer]:
                            model.zero_grad()
                            output=model.forward(sx).view(1,-1) 
                            label = output.max(1)[1].view(-1)
                            omega_loss = F.nll_loss(F.log_softmax(output, dim=1), label)
                            omega_loss.backward()

                            for pindex, (p_a, p_b) in enumerate(zip(wnew_a_params, wnew_b_params)):
                                g_a=p_a.grad.data.clone().detach().cpu().numpy()
                                g_b=p_b.grad.data.clone().detach().cpu().numpy()
                                gradients_A[pindex]+= np.abs(g_a) ** 2
                                gradients_B[pindex]+= np.abs(g_b) ** 2 
                                
                        # update the running average of the importance weights        
                        omega_As_old = omega_As[:]
                        omega_Bs_old = omega_Bs[:]
                        omega_As=[]
                        omega_Bs=[]
                        wnew_a_params = filter(lambda p: getattr(p, '_is_wnew_a', False), model.parameters())
                        wnew_b_params = filter(lambda p: getattr(p, '_is_wnew_b', False), model.parameters())
                        for pindex, (p_a, p_b) in enumerate(zip(wnew_a_params, wnew_b_params)):
                            if len(omega_As_old) != 0 and len(omega_Bs_old) != 0: # the lengths should be the same. 
                                omega_As.append(1/count_updates*gradients_A[pindex]+(1-1/count_updates)*omega_As_old[pindex])
                                omega_Bs.append(1/count_updates*gradients_B[pindex]+(1-1/count_updates)*omega_Bs_old[pindex])
                            else:
                                omega_As.append(gradients_A[pindex])
                                omega_Bs.append(gradients_B[pindex])
                        
                        # Freeze current LoRA and add a new set of LoRA parameters. 
                        if (args.new_lora): 
                            model.update_and_reset_lora_parameters()
                            model.save_lora_parameters(ckpt_path.replace(".pt", ".safetensors"))
                            model = model.to(device)

                    loss_window_means.append(loss_window_mean)
                    loss_window_variances.append(loss_window_variance)

                    # Update hard_buffer                   
                    if len(hard_buffer) == 0:
                        loss=[l.detach().cpu().numpy() for l in current_loss]
                    else:
                        loss=[l.detach().cpu().numpy() for l in (current_loss+hard_loss)]
                        
                    hard_buffer=[]
                    sorted_inputs=[lx for _,lx in reversed(sorted(zip(loss,xt),key= lambda f:f[0]))]
                    sorted_targets=[ly for _,ly in reversed(sorted(zip(loss,yt),key= lambda f:f[0]))]
                        
                    for i in range(min(args.hard_buffer_size,len(sorted_inputs))):
                        hard_buffer.append({'state':sorted_inputs[i],
                                            'trgt':sorted_targets[i]})
                    batch = []

                    
        test_stats = evaluate_till_now(model=model, data_loader=data_loader, device=device, 
                                    task_id=task_id, acc_matrix=acc_matrix, args=args)

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch,}

        if args.output_dir:
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
