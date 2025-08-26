import os
import sys
import time
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import defaultdict
from randaugment import RandAugment

from models import get_model
from datasets import get_dataset
from utils.augment import Cutout
from utils.memory import Memory
from utils.online_sampler import OnlineSampler, OnlineTestSampler
from utils.indexed_dataset import IndexedDataset
from utils.train_utils import select_optimizer, select_scheduler

from timm.models import create_model
from utils.lora import LoRA_ViT_timm
from utils.lora_utils import init_ckpt_path
from torch.nn import functional as F
import gc
import logging

class Online_LoRA_Trainer():
    def __init__(self, *args, **kwargs) -> None:

        self.mode    = kwargs.get("mode")

        self.n   = kwargs.get("n")
        self.m   = kwargs.get("m")
        self.rnd_NM  = kwargs.get("rnd_NM")

        self.MAS_weight = kwargs.get("MAS_weight")
        self.loss_window_length = kwargs.get("loss_window_length")
        self.loss_window_mean_threshold = kwargs.get("loss_window_mean_threshold")
        self.loss_window_variance_threshold = kwargs.get("loss_window_variance_threshold")
        self.hard_buffer_size = kwargs.get("hard_buffer_size")

        self.n_tasks = kwargs.get("n_tasks")
        self.dataset_name = kwargs.get("dataset")
        self.rnd_seed    = kwargs.get("rnd_seed")

        self.memory_size = kwargs.get("memory_size")
        self.log_path    = kwargs.get("log_path")
        self.model_name  = kwargs.get("model_name")
        self.opt_name    = kwargs.get("opt_name")
        self.sched_name  = kwargs.get("sched_name")
        self.batchsize  = kwargs.get("batchsize")
        self.n_worker    = kwargs.get("n_worker")
        self.lr  = kwargs.get("lr")
        self.init_model  = kwargs.get("init_model")
        self.init_opt    = kwargs.get("init_opt")
        self.topk    = kwargs.get("topk")
        self.use_amp = kwargs.get("use_amp")
        self.transforms  = kwargs.get("transforms")
        self.reg_coef    = kwargs.get("reg_coef")
        self.data_dir    = kwargs.get("data_dir")
        self.debug   = kwargs.get("debug")
        self.note    = kwargs.get("note")
        self.selection_size = kwargs.get("selection_size")

        self.eval_period     = kwargs.get("eval_period")
        self.temp_batchsize  = kwargs.get("temp_batchsize")
        self.online_iter     = kwargs.get("online_iter")
        self.num_gpus    = kwargs.get("num_gpus")
        self.workers_per_gpu     = kwargs.get("workers_per_gpu")
        self.imp_update_period   = kwargs.get("imp_update_period")

        # for distributed training
        self.dist_backend = 'nccl'
        self.dist_url = 'env://'

        self.lr_step     = kwargs.get("lr_step")    # for adaptive LR
        self.lr_length   = kwargs.get("lr_length")  # for adaptive LR
        self.lr_period   = kwargs.get("lr_period")  # for adaptive LR

        self.memory_epoch    = kwargs.get("memory_epoch") # for RM
        self.distilling  = kwargs.get("distilling") # for BiC
        self.agem_batch  = kwargs.get("agem_batch") # for A-GEM
        self.mir_cands   = kwargs.get("mir_cands")  # for MIR

        self.start_time = time.time()
        self.num_updates = 0
        self.train_count = 0

        self.ngpus_per_nodes = torch.cuda.device_count()
        self.world_size = 1
        if "WORLD_SIZE" in os.environ and os.environ["WORLD_SIZE"] != '':
            self.world_size = int(os.environ["WORLD_SIZE"]) * self.ngpus_per_nodes
        else:
            self.world_size = self.world_size * self.ngpus_per_nodes
        self.distributed = self.world_size > 1

        if self.distributed:
            self.batchsize = self.batchsize // self.world_size
        if self.temp_batchsize is None:
            self.temp_batchsize = self.batchsize // 2
        if self.temp_batchsize > self.batchsize:
            self.temp_batchsize = self.batchsize
        self.memory_batchsize = self.batchsize - self.temp_batchsize

        os.makedirs(f"{self.log_path}/logs/{self.dataset_name}/{self.note}", exist_ok=True)

    def setup_distributed_model(self):
        print("Building model...")
        self.model = self.model.to(self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        self.model.to(self.device)
        self.model_without_ddp = self.model
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
            self.model._set_static_graph()
            self.model_without_ddp = self.model.module
        self.criterion = self.model_without_ddp.loss_fn if hasattr(self.model_without_ddp, "loss_fn") else nn.CrossEntropyLoss(reduction="mean")
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)

        n_params = sum(p.numel() for p in self.model_without_ddp.parameters())
        print(f"Total Parameters :\t{n_params}")
        n_params = sum(p.numel() for p in self.model_without_ddp.parameters() if p.requires_grad)
        print(f"Learnable Parameters :\t{n_params}")
        print("")

    def setup_dataset(self):
        # get dataset
        self.train_dataset = self.dataset(root=self.data_dir, train=True, download=True, transform=self.train_transform)
        self.test_dataset = self.dataset(root=self.data_dir, train=False, download=True, transform=self.test_transform)
        self.n_classes = len(self.train_dataset.classes)

        self.exposed_classes = []
        self.mask = torch.zeros(self.n_classes, device=self.device) - torch.inf
        self.seen = 0

    def setup_transforms(self):
        train_transform = []
        self.cutmix = "cutmix" in self.transforms 
        if "autoaug" in self.transforms:
            train_transform.append(lambda x: (x*255).type(torch.uint8))
            if 'cifar' in self.dataset_name:
                train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('cifar10')))
            elif 'imagenet' in self.dataset_name:
                train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('imagenet')))
            elif 'svhn' in self.dataset_name:
                train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('svhn')))
            train_transform.append(lambda x: x.type(torch.float32)/255)
            
        if "cutout" in self.transforms:
            train_transform.append(Cutout(size=16))
        if "randaug" in self.transforms:
            train_transform.append(RandAugment())

        self.train_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
                transforms.RandomCrop(self.inp_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(self.mean, self.std),])
        self.test_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),])

    def run(self):
        # Distributed Launch
        if self.ngpus_per_nodes > 1:
            mp.spawn(self.main_worker, nprocs=self.ngpus_per_nodes, join=True)
        else:
            self.main_worker(0)
    
    def main_worker(self, gpu) -> None:
        self.gpu    = gpu % self.ngpus_per_nodes
        self.device = torch.device(self.gpu)
        if self.distributed:
            self.local_rank = self.gpu
            if 'SLURM_PROCID' in os.environ.keys():
                self.rank = int(os.environ['SLURM_PROCID']) * self.ngpus_per_nodes + self.gpu
                print(f"| Init Process group {os.environ['SLURM_PROCID']} : {self.local_rank}")
            elif 'WORLD_SIZE' in os.environ.keys():
                self.rank = int(os.environ['RANK']) * self.ngpus_per_nodes + self.gpu
                print(f"| Init Process group {os.environ['RANK']} : {self.local_rank}")
            else :
                self.rank = self.gpu
                print(f"| Init Process group 0 : {self.local_rank}")
            if 'MASTER_ADDR' not in os.environ.keys():
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '12701'
            torch.cuda.set_device(self.gpu)
            time.sleep(self.rank * 0.1) # prevent port collision
            dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
                                    world_size=self.world_size, rank=self.rank)
            torch.distributed.barrier()
            self.setup_for_distributed(self.is_main_process())
        else:
            pass
        
        if self.rnd_seed is not None:
            random.seed(self.rnd_seed)
            np.random.seed(self.rnd_seed)
            torch.manual_seed(self.rnd_seed)
            torch.cuda.manual_seed(self.rnd_seed)
            torch.cuda.manual_seed_all(self.rnd_seed) # if use multi-GPU
            cudnn.deterministic = True
            print('You have chosen to seed training. '
                'This will turn on the CUDNN deterministic setting, '
                'which can slow down your training considerably! '
                'You may see unexpected behavior when restarting '
                'from checkpoints.')
        cudnn.benchmark = False

        print(f"Select a CIL method ({self.mode})")

        self.dataset, self.mean, self.std, self.n_classes = get_dataset(self.dataset_name)
        print(f"Building model ({self.model_name})")
        vit_model = create_model('vit_base_patch16_224', pretrained=True)
        lora_model = LoRA_ViT_timm(vit_model=vit_model, r=4, num_classes=self.n_classes)
        net = lora_model.to(self.device)
        self.model = torch.nn.DataParallel(net)
        self.inp_size = 224
        self.setup_transforms()
        self.setup_dataset()
        self.setup_distributed_model()
        self.memory = Memory()
        self.total_samples = len(self.train_dataset)

        train_dataset = IndexedDataset(self.train_dataset)
        self.train_sampler = OnlineSampler(train_dataset, self.n_tasks, self.m, self.n, self.rnd_seed, self.rnd_NM, self.selection_size)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batchsize, sampler=self.train_sampler, num_workers=self.n_worker, pin_memory=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batchsize, shuffle=False, num_workers=self.n_worker, pin_memory=True)

        print(f"Incrementally training {self.n_tasks} tasks")
        task_records = defaultdict(list)
        eval_results = defaultdict(list)
        samples_cnt = 0

        num_eval = self.eval_period

        optimizer = torch.optim.Adam(self.model.module.parameters(), lr=self.lr) 
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

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
        ckpt_path = init_ckpt_path(loglevel=logging.ERROR)   
        
        for task_id in range(self.n_tasks):
            if self.mode == "joint" and task_id > 0:
                return

            print("\n" + "#" * 50)
            print(f"# Task {task_id} Session")
            print("#" * 50 + "\n")
            print("[2-1] Prepare a datalist for the current task")
            
            self.train_sampler.set_task(task_id)
            self.online_before_task(task_id)       

            for i, (images, labels, idx) in enumerate(self.train_dataloader):
                if self.debug and (i+1) * self.temp_batchsize >= 500:
                    break
                samples_cnt += images.size(0) * self.world_size
                self.add_new_class(labels)
                for j in range(len(labels)):
                    labels[j] = self.exposed_classes.index(labels[j].item())

                x = images.to(self.device)
                y = labels.to(self.device)
                if len(hard_buffer) != 0:
                    xh = [_['state'] for _ in hard_buffer]
                    yh = [_['trgt'] for _ in hard_buffer]

                # only 1 online epoch
                total_loss = (torch.tensor(0.0)).to(self.device)
                # Current batch loss
                current_loss = [] 
                y_pred = self.model.module.forward(x) 
                current_loss.append(criterion(y_pred, y))
                total_loss += criterion(y_pred, y)
                
                # Hard buffer loss
                hard_loss = []
                if len(hard_buffer) != 0:
                    # evaluate hard buffer
                    for image_h, label_h in zip(xh, yh):
                        yh_pred = self.model.module.forward(image_h) 
                        hard_loss.append(criterion(yh_pred,label_h))
                        total_loss += criterion(yh_pred,label_h)
                
                # keep train loss for loss window
                first_train_loss=total_loss.detach().cpu().numpy()
                
                wnew_a_params = filter(lambda p: getattr(p, '_is_wnew_a', False), self.model.module.parameters())
                wnew_b_params = filter(lambda p: getattr(p, '_is_wnew_b', False), self.model.module.parameters())

                # Regularization loss
                if len(omega_As)!=0 and len(omega_As)==len(omega_Bs): # omega_As and omega_Bs should have same length. 
                    mas_loss = 0.
                    for pindex, (p_a, p_b) in enumerate(zip(wnew_a_params, wnew_b_params)):
                        product_a = torch.from_numpy(omega_As[pindex]).type(torch.float32).to(self.device) * ((p_a) ** 2)
                        product_b = torch.from_numpy(omega_Bs[pindex]).type(torch.float32).to(self.device) * ((p_b) ** 2)
                        mas_loss += torch.sum(product_a) + torch.sum(product_b) 
                    print('MAS loss: {}'.format(mas_loss))
                    total_loss+=self.MAS_weight/2.*mas_loss

                optimizer.zero_grad()
                torch.sum(total_loss).backward()
                optimizer.step()

                # save training accuracy on total batch
                if len(hard_buffer) != 0:
                    xt=xh + [x]
                    yt=yh + [y]
                else:
                    xt=[x]
                    yt=[y]

                # Update loss_window and detect loss plateaus
                loss_window.append(np.mean(first_train_loss))
                if len(loss_window)>self.loss_window_length: del loss_window[0]
                loss_window_mean=np.mean(loss_window)
                loss_window_variance=np.var(loss_window)
                print('loss window mean: {0:0.3f}, loss window variance: {1:0.3f}'.format(loss_window_mean, loss_window_variance))
                # Check the statistics of the current window
                if not new_peak_detected and loss_window_mean > last_loss_window_mean+np.sqrt(last_loss_window_variance) : #and loss_window_variance > 0.1:
                    new_peak_detected=True  
                # Time for updating importance weights    
                if loss_window_mean < self.loss_window_mean_threshold and loss_window_variance < self.loss_window_variance_threshold and new_peak_detected:
                    new_task = False
                    count_updates+=1
                    print('importance weights update')
                    last_loss_window_mean=loss_window_mean
                    last_loss_window_variance=loss_window_variance
                    new_peak_detected=False
                    
                    # calculate imporatance based on each sample in the hardbuffer
                    gradients_A = [0 for p in self.model.module.parameters() if getattr(p, '_is_wnew_a', False)]
                    gradients_B = [0 for p in self.model.module.parameters() if getattr(p, '_is_wnew_b', False)]
                    
                    self.model.module.eval()
                    wnew_a_params = filter(lambda p: getattr(p, '_is_wnew_a', False), self.model.module.parameters())
                    wnew_b_params = filter(lambda p: getattr(p, '_is_wnew_b', False), self.model.module.parameters())
                    for sx in [_['state'] for _ in hard_buffer]:
                        self.model.module.zero_grad()
                        output=self.model.module.forward(sx).view(1,-1) 
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
                    wnew_a_params = filter(lambda p: getattr(p, '_is_wnew_a', False), self.model.module.parameters())
                    wnew_b_params = filter(lambda p: getattr(p, '_is_wnew_b', False), self.model.module.parameters())
                    for pindex, (p_a, p_b) in enumerate(zip(wnew_a_params, wnew_b_params)):
                        if len(omega_As_old) != 0 and len(omega_Bs_old) != 0: # the lengths should be the same. 
                            omega_As.append(1/count_updates*gradients_A[pindex]+(1-1/count_updates)*omega_As_old[pindex])
                            omega_Bs.append(1/count_updates*gradients_B[pindex]+(1-1/count_updates)*omega_Bs_old[pindex])
                        else:
                            omega_As.append(gradients_A[pindex])
                            omega_Bs.append(gradients_B[pindex])
                    
                    # Added: freeze current LoRA and create new set of LoRA parameters. 
                    self.model.module.update_and_reset_lora_parameters()
                    self.model.module.save_lora_parameters(ckpt_path.replace(".pt", ".safetensors"))
                    self.model.module = self.model.module.to(self.device)

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
                    
                for i in range(min(self.hard_buffer_size, len(sorted_inputs))):
                    hard_buffer.append({'state':sorted_inputs[i],
                                        'trgt':sorted_targets[i]})

                if samples_cnt > num_eval:
                    with torch.no_grad():
                        test_sampler = OnlineTestSampler(self.test_dataset, self.exposed_classes)
                        test_dataloader = DataLoader(self.test_dataset, batch_size=self.batchsize*2, sampler=test_sampler, num_workers=self.n_worker)
                        eval_dict = self.online_evaluate(test_dataloader) # combined_dataloader
                        if self.distributed:
                            eval_dict =  torch.tensor([eval_dict['avg_loss'], eval_dict['avg_acc'], *eval_dict['cls_acc']], device=self.device)
                            dist.reduce(eval_dict, dst=0, op=dist.ReduceOp.SUM)
                            eval_dict = eval_dict.cpu().numpy()
                            eval_dict = {'avg_loss': eval_dict[0]/self.world_size, 'avg_acc': eval_dict[1]/self.world_size, 'cls_acc': eval_dict[2:]/self.world_size}
                        if self.is_main_process():
                            eval_results["test_acc"].append(eval_dict['avg_acc'])
                            eval_results["avg_acc"].append(eval_dict['cls_acc'])
                            eval_results["data_cnt"].append(num_eval)
                            self.report_test(samples_cnt, eval_dict["avg_loss"], eval_dict['avg_acc'])
                        num_eval += self.eval_period
                sys.stdout.flush()
            self.report_test(samples_cnt, eval_dict["avg_loss"], eval_dict['avg_acc'])
            self.online_after_task(task_id)
            
            test_sampler = OnlineTestSampler(self.test_dataset, self.exposed_classes)
            test_dataloader = DataLoader(self.test_dataset, batch_size=self.batchsize*2, sampler=test_sampler, num_workers=self.n_worker)
            eval_dict = self.online_evaluate(test_dataloader) #  combined_dataloader
            #! after training done
            # self.report_test(num_eval, eval_dict["avg_loss"], eval_dict['avg_acc'])
            
            if self.distributed:
                eval_dict =  torch.tensor([eval_dict['avg_loss'], eval_dict['avg_acc'], *eval_dict['cls_acc']], device=self.device)
                dist.reduce(eval_dict, dst=0, op=dist.ReduceOp.SUM)
                eval_dict = eval_dict.cpu().numpy()
                eval_dict = {'avg_loss': eval_dict[0]/self.world_size, 'avg_acc': eval_dict[1]/self.world_size, 'cls_acc': eval_dict[2:]/self.world_size}
            task_acc = eval_dict['avg_acc']

            print("[2-4] Update the information for the current task")
            task_records["task_acc"].append(task_acc)
            task_records["cls_acc"].append(eval_dict["cls_acc"])

            print("[2-5] Report task result")
        if self.is_main_process():        
            np.save(f"{self.log_path}/logs/{self.dataset_name}/{self.note}/seed_{self.rnd_seed}.npy", task_records["task_acc"])

            if self.eval_period is not None:
                np.save(f'{self.log_path}/logs/{self.dataset_name}/{self.note}/seed_{self.rnd_seed}_eval.npy', eval_results['test_acc'])
                np.save(f'{self.log_path}/logs/{self.dataset_name}/{self.note}/seed_{self.rnd_seed}_eval_time.npy', eval_results['data_cnt'])
    
            # Accuracy (A)
            A_auc = np.mean(eval_results["test_acc"])
            A_avg = np.mean(task_records["task_acc"])
            A_last = task_records["task_acc"][self.n_tasks - 1]

            # Forgetting (F)
            cls_acc = np.array(task_records["cls_acc"])
            acc_diff = []
            for j in range(self.n_classes):
                if np.max(cls_acc[:-1, j]) > 0:
                    acc_diff.append(np.max(cls_acc[:-1, j]) - cls_acc[-1, j])
            F_last = np.mean(acc_diff)

            print(f"======== Summary =======")
            print(f"A_auc {A_auc} | A_avg {A_avg} | A_last {A_last} | F_last {F_last}")
            # for i in range(len(cls_acc)):
            #     print(f"Task {i}")
            #     print(cls_acc[i])
            print(f"="*24)

    def add_new_class(self, class_name):
        exposed_classes = []
        for label in class_name:
            if label.item() not in self.exposed_classes:
                self.exposed_classes.append(label.item())
        if self.distributed:
            exposed_classes = torch.cat(self.all_gather(torch.tensor(self.exposed_classes, device=self.device))).cpu().tolist()
            self.exposed_classes = []
            for cls in exposed_classes:
                if cls not in self.exposed_classes:
                    self.exposed_classes.append(cls)
        self.memory.add_new_class(cls_list=self.exposed_classes)
        self.mask[:len(self.exposed_classes)] = 0
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def online_step(self, images, labels, idx):
        pass

    def online_before_task(self, task_id):
        pass

    def online_after_task(self, task_id):
        pass
    
    def online_evaluate(self, test_loader):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model.module.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)

                logit = self.model.module.forward(x)
                logit = logit + self.mask
                loss = self.criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        
        eval_dict = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
        return eval_dict

            
    def is_dist_avail_and_initialized(self):
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True

    def get_world_size(self):
        if not self.is_dist_avail_and_initialized():
            return 1
        return dist.get_world_size()

    def get_rank(self):
        if not self.is_dist_avail_and_initialized():
            return 0
        return dist.get_rank()

    def is_main_process(self):
        return self.get_rank() == 0

    def setup_for_distributed(self, is_master):
        """
        This function disables printing when not in master process
        """
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)
        __builtin__.print = print

    def report_training(self, sample_num, train_loss, train_acc):
        print(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"Num_Classes {len(self.exposed_classes)} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )

    def report_test(self, sample_num, avg_loss, avg_acc):
        print(
            f"Test | Sample # {sample_num} | test_loss {avg_loss:.4f} | test_acc {avg_acc:.4f} | "
        )
    
    def _interpret_pred(self, y, pred):
        # xlable is batch
        ret_num_data = torch.zeros(self.n_classes)
        ret_corrects = torch.zeros(self.n_classes)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        correct_xlabel = y.masked_select(y == pred)
        correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer)

    def all_gather(self, item):
        local_size = torch.tensor(item.size(0), device=self.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                dist.gather(local_size, all_sizes, dst=i)
            else:
                dist.gather(local_size, dst=i)
        # dist.all_gather(all_sizes, local_size, async_op=False)
        max_size = max(all_sizes)

        size_diff = max_size.item() - local_size.item()
        if size_diff:
            padding = torch.zeros(size_diff, device=self.device, dtype=item.dtype)
            item = torch.cat((item, padding))

        all_qs_padded = [torch.zeros_like(item) for _ in range(dist.get_world_size())]

        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                dist.gather(item, all_qs_padded, dst=i)
            else:
                dist.gather(item, dst=i)

        # dist.all_gather(all_qs_padded, item)
        all_qs = []
        for q, size in zip(all_qs_padded, all_sizes):
            all_qs.append(q[:size])
        return all_qs
    
    def train_data_config(self, n_task, train_dataset,train_sampler):
        for t_i in range(n_task):
            train_sampler.set_task(t_i)
            train_dataloader = DataLoader(train_dataset,batch_size=self.batchsize,sampler=train_sampler,num_workers=4)
            data_info={}
            for i,data in enumerate(train_dataloader):
                _,label = data
                label = label.to(self.device)
                for b in range(len(label)):
                    if 'Class_'+str(label[b].item()) in data_info.keys():
                        data_info['Class_'+str(label[b].item())] += 1
                    else:
                        data_info['Class_'+str(label[b].item())] = 1
            print(f"[Train] Task{t_i} Data Info")
            print(data_info);print()
            convert_data_info = self.convert_class_label(data_info)
            np.save(f"{self.log_path}/logs/{self.dataset_name}/{self.note}/seed_{self.rnd_seed}_task{t_i}_train_data.npy", convert_data_info)
            print(convert_data_info)
            
            print()
            
    def test_data_config(self, test_dataloader,task_id):
        data_info={}
        for i,data in enumerate(test_dataloader):
            _,label = data
            label = label.to(self.device)
            
            for b in range(len(label)):
                if 'Class_'+str(label[b].item()) in data_info.keys():
                    data_info['Class_'+str(label[b].item())]+=1
                else:
                    data_info['Class_'+str(label[b].item())]=1
        
        print("<<Exposed Class>>")
        print(self.exposed_classes)
        
        print(f"[Test] Task {task_id} Data Info")
        print(data_info)
        print("<<Convert>>")
        convert_data_info = self.convert_class_label(data_info)
        print(convert_data_info)
        print()
        
    def convert_class_label(self,data_info):
        #* self.class_list => original class label
        self.class_list = self.train_dataset.classes
        for key in list(data_info.keys()):
            old_key= int(key[6:])
            data_info[self.class_list[old_key]] = data_info.pop(key)
            
        return data_info
    
    def current_task_data(self,train_loader):
        data_info={}
        for i,data in enumerate(train_loader):
            _,label = data
            
            for b in range(label.shape[0]):
                if 'Class_'+str(label[b].item()) in data_info.keys():
                    data_info['Class_'+str(label[b].item())] +=1
                else:
                    data_info['Class_'+str(label[b].item())] =1
        
        print("Current Task Data Info")
        print(data_info)
        print("<<Convert to str>>")
        convert_data_info = self.convert_class_label(data_info)
        print(convert_data_info)
        print()