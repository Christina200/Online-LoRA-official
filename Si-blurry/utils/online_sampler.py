import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from typing import Optional, Sized, Iterable, Tuple

class OnlineSampler(Sampler):
    def __init__(self, data_source: Optional[Sized], num_tasks: int, m: int, n: int,  rnd_seed: int, cur_iter: int= 0, varing_NM: bool= False, num_replicas: int=None, rank: int=None) -> None:

        self.data_source    = data_source
        self.classes    = self.data_source.classes
        self.targets    = self.data_source.targets
        self.generator  = torch.Generator().manual_seed(rnd_seed)
        
        self.n  = n
        self.m  = m
        self.varing_NM = varing_NM
        self.task = cur_iter

        if num_replicas is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            num_replicas = dist.get_world_size()
        if rank is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            rank = dist.get_rank()

        self.distributed = num_replicas is not None and rank is not None
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0

        self.disjoint_num   = len(self.classes) * n // 100
        self.disjoint_num   = int(self.disjoint_num // num_tasks) * num_tasks
        self.blurry_num     = len(self.classes) - self.disjoint_num
        self.blurry_num     = int(self.blurry_num // num_tasks) * num_tasks

        if not self.varing_NM:
            # Divide classes into N% of disjoint and (100 - N)% of blurry
            class_order         = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes   = class_order[:self.disjoint_num]
            self.disjoint_classes   = self.disjoint_classes.reshape(num_tasks, -1).tolist()
            self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
            self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

            print("disjoint classes: ", self.disjoint_classes)
            print("blurry classes: ", self.blurry_classes)
            # Get indices of disjoint and blurry classes
            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)
                        break

            # Randomly shuffle M% of blurry indices
            blurred = []
            for i in range(num_tasks):
                blurred += self.blurry_indices[i][:len(self.blurry_indices[i]) * m // 100]
                self.blurry_indices[i] = self.blurry_indices[i][len(self.blurry_indices[i]) * m // 100:]
            blurred = torch.tensor(blurred)
            blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
            print("blurry indices: ", len(blurred))
            num_blurred = len(blurred) // num_tasks
            for i in range(num_tasks):
                self.blurry_indices[i] += blurred[:num_blurred]
                blurred = blurred[num_blurred:]
            
            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                print("task %d: disjoint %d, blurry %d" % (i, len(self.disjoint_indices[i]), len(self.blurry_indices[i])))
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()
        else:
            # Divide classes into N% of disjoint and (100 - N)% of blurry
            class_order = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes = class_order[:self.disjoint_num].tolist()
            if self.disjoint_num > 0:
                self.disjoint_slice = [0] + torch.randint(0, self.disjoint_num, (num_tasks - 1,), generator=self.generator).sort().values.tolist() + [self.disjoint_num]
                self.disjoint_classes = [self.disjoint_classes[self.disjoint_slice[i]:self.disjoint_slice[i + 1]] for i in range(num_tasks)]
            else:
                self.disjoint_classes = [[] for _ in range(num_tasks)]

            if self.blurry_num > 0:
                self.blurry_slice = [0] + torch.randint(0, self.blurry_num, (num_tasks - 1,), generator=self.generator).sort().values.tolist() + [self.blurry_num]
                self.blurry_classes = [class_order[self.disjoint_num + self.blurry_slice[i]:self.disjoint_num + self.blurry_slice[i + 1]].tolist() for i in range(num_tasks)]
            else:
                self.blurry_classes = [[] for _ in range(num_tasks)]
            # self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
            # self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

            print("disjoint classes: ", self.disjoint_classes)
            print("blurry classes: ", self.blurry_classes)
            
            # Get indices of disjoint and blurry classes
            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            num_blurred = 0
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)
                        num_blurred += 1
                        break

            # Randomly shuffle M% of blurry indices
            blurred = []
            num_blurred = num_blurred * m // 100
            if num_blurred > 0:
                num_blurred = [0] + torch.randint(0, num_blurred, (num_tasks-1,), generator=self.generator).sort().values.tolist() + [num_blurred]

                for i in range(num_tasks):
                    blurred += self.blurry_indices[i][:num_blurred[i + 1] - num_blurred[i]]
                    self.blurry_indices[i] = self.blurry_indices[i][num_blurred[i + 1] - num_blurred[i]:]
                blurred = torch.tensor(blurred)
                blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
                print("blurry indices: ", len(blurred))
                # num_blurred = len(blurred) // num_tasks
                for i in range(num_tasks):
                    self.blurry_indices[i] += blurred[:num_blurred[i + 1] - num_blurred[i]]
                    blurred = blurred[num_blurred[i + 1] - num_blurred[i]:]
            
            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                print("task %d: disjoint %d, blurry %d" % (i, len(self.disjoint_indices[i]), len(self.blurry_indices[i])))
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()

        if self.distributed:
            self.num_samples = int(len(self.indices[self.task]) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas  
            self.num_selected_samples = int(len(self.indices[self.task]) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices[self.task]))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices[self.task]))

    def __iter__(self) -> Iterable[int]:
        if self.distributed:
            # subsample
            indices = self.indices[self.task][self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            return iter(indices[:self.num_selected_samples])
        else:
            return iter(self.indices[self.task])

    def __len__(self) -> int:
        return self.num_selected_samples

    def set_task(self, cur_iter: int)-> None:
        if cur_iter >= len(self.indices) or cur_iter < 0:
            raise ValueError("task out of range")
        self.task = cur_iter

        if self.distributed:
            self.num_samples = int(len(self.indices[self.task]) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas  
            self.num_selected_samples = int(len(self.indices[self.task]) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices[self.task]))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices[self.task]))
    
    def get_task(self, cur_iter: int)-> Iterable[int]:
        indices = self.indices[cur_iter][self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return indices[:self.num_selected_samples]

class OnlineBatchSampler(Sampler):
    def __init__(self, data_source: Optional[Sized], num_tasks: int, m: int, n: int, rnd_seed: int, batchsize: int=16, online_iter: int=1, cur_iter: int=0, varing_NM: bool=False, num_replicas: int=None, rank: int=None) -> None:
        super().__init__(data_source)
        self.data_source    = data_source
        self.classes    = self.data_source.classes
        self.targets    = self.data_source.targets
        self.num_tasks  = num_tasks
        self.m      = m
        self.n      = n
        self.rnd_seed   = rnd_seed
        self.batchsize  = batchsize
        self.online_iter    = online_iter
        self.cur_iter   = cur_iter
        self.varing_NM  = varing_NM

        if num_replicas is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            num_replicas = dist.get_world_size()
        if rank is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            rank = dist.get_rank()

        self.distributed = num_replicas is not None and rank is not None
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0

        self.disjoint_num   = len(self.classes) * n // 100
        self.disjoint_num   = int(self.disjoint_num // num_tasks) * num_tasks
        self.blurry_num     = len(self.classes) - self.disjoint_num
        self.blurry_num     = int(self.blurry_num // num_tasks) * num_tasks

        if not self.varing_NM:
            # Divide classes into N% of disjoint and (100 - N)% of blurry
            class_order         = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes   = class_order[:self.disjoint_num]
            self.disjoint_classes   = self.disjoint_classes.reshape(num_tasks, -1).tolist()
            self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
            self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

            print("disjoint classes: ", self.disjoint_classes)
            print("blurry classes: ", self.blurry_classes)
            # Get indices of disjoint and blurry classes
            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)
                        break

            # Randomly shuffle M% of blurry indices
            blurred = []
            for i in range(num_tasks):
                blurred += self.blurry_indices[i][:len(self.blurry_indices[i]) * m // 100]
                self.blurry_indices[i] = self.blurry_indices[i][len(self.blurry_indices[i]) * m // 100:]
            blurred = torch.tensor(blurred)
            blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
            print("blurry indices: ", len(blurred))
            num_blurred = len(blurred) // num_tasks
            for i in range(num_tasks):
                self.blurry_indices[i] += blurred[:num_blurred]
                blurred = blurred[num_blurred:]
            
            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                print("task %d: disjoint %d, blurry %d" % (i, len(self.disjoint_indices[i]), len(self.blurry_indices[i])))
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)]
                num_batches     = int(self.indices[i].size(0) // self.batchsize)
                rest            = self.indices[i].size(0) % self.batchsize
                self.indices[i] = self.indices[i][:num_batches * self.batchsize].reshape(-1, self.batchsize).repeat(self.online_iter, 1).flatten().tolist() + self.indices[i][-rest:].tolist()
        else:
            # Divide classes into N% of disjoint and (100 - N)% of blurry
            class_order         = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes   = class_order[:self.disjoint_num].tolist()
            if self.disjoint_num > 0:
                self.disjoint_slice = [0] + torch.randint(0, self.disjoint_num, (num_tasks - 1,), generator=self.generator).sort().values.tolist() + [self.disjoint_num]
                self.disjoint_classes = [self.disjoint_classes[self.disjoint_slice[i]:self.disjoint_slice[i + 1]] for i in range(num_tasks)]
            else:
                self.disjoint_classes = [[] for _ in range(num_tasks)]

            self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
            self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

            print("disjoint classes: ", self.disjoint_classes)
            print("blurry classes: ", self.blurry_classes)
            
            # Get indices of disjoint and blurry classes
            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            num_blurred = 0
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)
                        num_blurred += 1
                        break

            # Randomly shuffle M% of blurry indices
            blurred = []
            num_blurred = num_blurred * m // 100
            num_blurred = [0] + torch.randint(0, num_blurred, (num_tasks-1,), generator=self.generator).sort().values.tolist() + [num_blurred]

            for i in range(num_tasks):
                blurred += self.blurry_indices[i][:num_blurred[i + 1] - num_blurred[i]]
                self.blurry_indices[i] = self.blurry_indices[i][num_blurred[i + 1] - num_blurred[i]:]
            blurred = torch.tensor(blurred)
            blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
            print("blurry indices: ", len(blurred))
            # num_blurred = len(blurred) // num_tasks
            for i in range(num_tasks):
                self.blurry_indices[i] += blurred[:num_blurred[i + 1] - num_blurred[i]]
                blurred = blurred[num_blurred[i + 1] - num_blurred[i]:]
            
            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                print("task %d: disjoint %d, blurry %d" % (i, len(self.disjoint_indices[i]), len(self.blurry_indices[i])))
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()
                num_batches     = int(self.indices[i].size(0) // self.batchsize)
                rest            = self.indices[i].size(0) % self.batchsize
                self.indices[i] = self.indices[i][:num_batches * self.batchsize].reshape(-1, self.batchsize).repeat(self.online_iter, 1).flatten().tolist() + self.indices[i][-rest:].tolist()

    def __iter__(self) -> Iterable[int]:
        if self.distributed:
            # subsample
            indices = self.indices[self.task][self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            return iter(indices[:self.num_selected_samples])
        else:
            return iter(self.indices[self.task])

    def __len__(self) -> int:
        return self.num_selected_samples

    def set_task(self, cur_iter: int) -> None:

        if cur_iter >= len(self.indices) or cur_iter < 0:
            raise ValueError("task out of range")
        self.task = cur_iter

        if self.distributed:
            self.num_samples = int(len(self.indices[self.task]) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas  
            self.num_selected_samples = int(len(self.indices[self.task]) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices[self.task]))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices[self.task]))
    
    def get_task(self, cur_iter : int) -> Iterable[int]:
        indices = self.indices[cur_iter][self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return indices[:self.num_selected_samples]

    def get_task_classes(self, cur_iter : int) -> Iterable[int]:
        return list(set(self.classes[self.indices[cur_iter]]))

class OnlineTestSampler(Sampler):
    def __init__(self, data_source: Optional[Sized], exposed_class : Iterable[int], num_replicas: int=None, rank: int=None) -> None:
        self.data_source    = data_source
        self.classes    = self.data_source.classes
        self.targets    = self.data_source.targets
        self.exposed_class  = exposed_class
        self.indices    = [i for i in range(self.data_source.__len__()) if self.targets[i] in self.exposed_class]

        if num_replicas is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            num_replicas = dist.get_world_size()
        if rank is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            rank = dist.get_rank()

        self.distributed = num_replicas is not None and rank is not None
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0

        if self.distributed:
            self.num_samples = int(len(self.indices) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
            self.num_selected_samples = int(len(self.indices) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices))

    def __iter__(self) -> Iterable[int]:
        if self.distributed:
            # subsample
            indices = self.indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            return iter(indices[:self.num_selected_samples])
        else:
            return iter(self.indices)

    def __len__(self) -> int:
        return self.num_selected_samples