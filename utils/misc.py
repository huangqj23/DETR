import os
import subprocess
import time
from collections import defaultdict, deque
import pickle
from packaging import version
from typing import Optional, List
import torch
import torch.distributed as dist
from torch import Tensor
import torch.jit.unsupported_tensor_ops
import torchvision
import datetime

# 由于pytorch和torchvision 0.5版本有empty tensor的bug，所以需要判断torchvision的版本 
if version.parse(torchvision.__version__) < version.parse('0.7'):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size

# 用于在分布式计算环境中控制打印行为。具体来说，该函数会在不是主进程的情况下禁用 print 函数的输出，以减少不必要的日志信息。
# 在 Python 中，内建的函数（如 print）是可以在运行时被替换的。Python 的灵活性允许对内建函数进行重定义或替换，这在某些情况下（如调试、测试或特殊环境设置）是非常有用的。
# builtins 是一个特殊模块，包含所有的内建函数和异常。通过导入这个模块，可以访问和修改内建函数。
def setup_for_distributed(is_master):
    """
    当不在主进程时，此功能禁用打印。
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print
    
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print # 将内建的 print 函数替换为新定义的 print 函数。
    
# 检查分布式计算环境是否可用并已初始化。它通常用于涉及分布式训练的机器学习框架中，例如 PyTorch 的分布式包 
def is_dist_avail_and_initialized():
    if not dist.is_available(): # check if the distributed package is available
        return False
    # Checking if the default process group has been initialized
    if not dist.is_initialized(): # if distributed package is initialized, check if the distributed environment has been properly initialized
        return False
    return True

# 它返回当前默认进程组中的总进程数，也就是“世界大小”（world size）。这表示参与分布式训练或计算的总进程数量。
# 在分布式训练中，多个进程通常在多个计算节点上运行，每个进程可能负责处理一部分数据或模型参数。通过 dist.get_world_size()，
# 你可以获取到这些进程的总数，从而在代码中进行相应的逻辑处理，比如计算全局平均值时需要除以世界大小。
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

# 用于获取当前进程在分布式计算环境中的“排名”（rank）。在分布式计算中，排名用于标识每个进程的唯一身份。
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0 # 如果分布式环境不可用或未初始化，函数返回 0。这通常表示当前进程不是在分布式环境中运行，或者初始化失败。
    return dist.get_rank() # 在分布式环境中，dist.get_rank() 返回 0 表示当前进程是“主进程”或“根进程”。这通常是第一个启动的进程，用于执行某些特定任务，如日志记录或模型检查点保存。

def is_main_process():
    return get_rank() == 0  # 判断是不是主进程

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs) # 在主进程中保存模型或数据到指定的文件路径
        
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank==0)
        

# 用于跟踪一系列数值，并提供对这些数值的平滑统计信息的访问，如中位数、平均值、全局平均值等。它还提供了在分布式计算环境中同步这些数值的方法。
class SmoothedValue(object):
    """跟踪一系列值，并提供对窗口内或全局系列平均值的平滑值的访问。"""
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)  # 双端队列
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
    
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
    
    def synchronize_between_processes(self):
        """
        用于在分布式计算环境中同步多个进程之间的某个平均值。具体来说，它使用 all_reduce 操作来确保所有进程共享相同的计数和总和。这在分布式训练中非常有用，例如在计算全局平均损失或准确率时。
        通过all_reduce在所有进程中同步平均值。这需要所有进程在调用此函数时都到达这一点。
        注意：不对deque进行同步
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier() # 用于同步所有进程，确保所有进程都在同一时间点调用此方法。
        dist.all_reduce(t) # 将 t 张量在所有进程间进行求和操作，以确保每个进程得到相同的结果。
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]
        
    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()
    
    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()
    
    @property
    def global_avg(self):
        return self.total / self.count  # 总和 / 计数
    
    @property
    def max(self):
        return max(self.deque)
    
    @property
    def value(self): # 返回最近添加的数
        return self.deque[-1]
    
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )
        

# all_gather 操作用于将每个参与进程中的数据收集到所有进程中。每个进程都会得到所有其他进程的数据副本。
# all_reduce 操作用于对每个参与进程的数据进行归约操作（如求和、求平均等），然后将结果分发给所有进程。每个进程会得到相同的归约结果。
def all_gather(data):
    """在任意可序列化的数据（不一定是张量）上运行 all_gather
    Args:
        data: any pickable object.
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    
    # serialized to a Tensor
    buffer = pickle.dumps(data) # 使用 pickle 序列化数据，将其转换为字节流。
    storage = torch.ByteStorage.from_buffer(buffer) # 将字节流转换为 torch.ByteStorage。
    tensor = torch.ByteTensor(storage).to('cuda') # 将存储对象转换为 torch.ByteTensor，并移动到 GPU 上（使用 CUDA）。
    
    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device='cuda') # 计算当前进程中张量的元素数量。
    size_list = [torch.tensor([0], device='cuda') for _ in range(world_size)] # 初始化一个列表，用于存储每个进程的张量大小。
    dist.all_gather(size_list, local_size) #  使用 all_gather 收集每个进程的张量大小。
    size_list = [int(size.item() for size in size_list)] # 将张量大小转换为整数列表。
    max_size = max(size_list) # 找出最大的张量大小，确保所有进程都能接收相同大小的数据。
    
    # receiving Tensor from all ranks
    # 我们对张量进行填充，因为 torch all_gather 不支持收集不同形状的张量
    tensor_list = [] #  初始化一个列表，用于存储从所有进程接收到的张量。
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.unit8, device='cuda')) # 为每个进程创建一个空的张量 torch.empty((max_size,), dtype=torch.uint8, device="cuda")，以便接收数据。
    if local_size != max_size: # 如果当前进程的张量大小小于最大大小，则进行填充 padding，以确保所有张量大小一致。
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.unit8, device='cuda')
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor) # 收集所有进程的张量到 tensor_list 中。将反序列化后的数据添加到 data_list 中。
    
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size] # 遍历 size_list 和 tensor_list，根据每个进程的实际大小 size，从接收到的张量中提取有效字节。
        data_list.append(pickle.loads(buffer)) # 使用 pickle.loads(buffer) 将字节流反序列化为原始数据结构。
    # 返回 data_list，其中包含了从每个进程收集到的原始数据。
    return data_list  

'''
在多进程分布式计算中，`dist.all_gather` 是一个同步操作，即所有参与的进程都会执行 `dist.all_gather`。这意味着每个进程都需要调用 `dist.all_gather`，并且该函数会在所有进程都到达这一步时才继续执行。这种同步机制确保了数据在所有进程之间的一致性和同步性。

### 具体执行过程：

1. **计算和初始化**:
   - 每个进程都会计算自身的 `local_size`，即当前进程中张量的元素数量。
   - 每个进程都会初始化一个 `size_list`，用于存储来自所有进程的张量大小。

2. **数据收集**:
   - 每个进程调用 `dist.all_gather(size_list, local_size)`，这会将所有进程的 `local_size` 收集到每个进程的 `size_list` 中。
   - 由于 `all_gather` 是一个同步操作，所有进程在调用此函数时都会等待，直到所有进程都准备好进行数据交换。

3. **后续处理**:
   - 在 `all_gather` 完成后，每个进程的 `size_list` 都会包含来自所有进程的张量大小。
   - 然后，每个进程都会计算 `max_size`，以便后续操作中使用统一的张量大小进行数据处理。

因此，`dist.all_gather` 不仅仅是主进程执行，而是所有参与的进程都必须执行。这样做的目的是确保每个进程都能获得所有其他进程的信息，实现全局同步和一致的数据视图。
'''

def reduce_dict(input_dict, average=True):
    '''归约所有进程字典中的值，以便所有进程都有平均结果。返回一个与输入字典具有相同字段的字典，经过归约处理。
    Args:
        input_dict: all the values will be reduced
        average(bool): whether to do average or sum
    '''
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    
    with torch.no_grad():
        names = []
        values = []
        # 对key进行排序，以确保它们在各个进程中保持一致。
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
    
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue) # defaultdict 在访问不存在的键时不会抛出 KeyError，而是自动调用一个工厂函数为这个键生成一个默认值。
        self.delimiter = delimiter
        
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
            
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()
    
    def add_meter(self, name, meter):
        self.meters[name] = meter
        
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB
                    ))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time)
                    ))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        
# 用于获取当前 Git 仓库的状态信息，包括当前提交的 SHA 值、工作目录的修改状态以及当前分支名称。
def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = 'clean'
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD']) # 获取当前提交的 SHA 值。
        subprocess.check_output(['git', 'diff'], cwd=cwd) # 检查工作目录是否有未提交的更改，虽然这行代码没有直接使用输出，但它会在有更改时返回非空结果。这里相当于没用
        diff = _run(['git', 'diff-index', 'HEAD']) # 检查与 HEAD 的差异。如果有差异，则 diff 将包含输出，否则为空。
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD']) # 获取当前分支名称。
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message

def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)

# 传入的是一个列表，每个元素是一个图片的shape信息。统计所有图片中最大的HWC
def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

class NestedTensor(object): # 在 Python 3 中，继承自 object 是隐含的，所以可以省略。
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask
    
    def to(self, device):
        # type: (Device) -> NestedTensor
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask =None
        # 返回一个新的 NestedTensor 对象，包含移动（设备gpu）后的 tensors 和 mask。
        return NestedTensor(cast_tensor, cast_mask)
    
    def decompose(self):
        return self.tensors, self.mask
    
    def __repr__(self):
        return str(self.tensors)
    
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    #TODO 可以写的更通用
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)
    
        # TODO make it support different-sized images可以进一步改进适配不同大小的图片
        max_size = _max_by_axis([list(img.shape) for img in tensor_list]) # 返回最大的[H, W, C]
        batch_shape = [len(tensor_list)] + max_size # batch_size + [H,W,C]
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        # 如果图片大小小于HW则需要进行填充
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]] = False # 有图片的地方设置为False
    
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)
    
    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)
        
        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))
        
    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)
    
    return NestedTensor(tensor, mask=mask)

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if version.parse(torchvision.__version__) < version.parse('0.7'):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )
        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)