import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP

# 模型定义
class LeNet(nn.Module):
    def __init__(self, num_classes=100):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)  # CIFAR100 has 100 classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    # 销毁进程组
    dist.destroy_process_group()

def get_model():
    model = LeNet(100).cuda()
    model = DDP(model, device_ids=[torch.cuda.current_device()])
    return model

def get_dataloader(train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    rank = dist.get_rank()
    # 每个进程创建其独立的数据目录，避免I/O冲突
    # 这里使用rank来创建独立目录，例如：'./data_0'，'./data_1'等
    # 这种方法避免了多个进程同时写入同一个文件所导致的冲突
    # 注：这是一种简单的解决方案，但在需要大量磁盘空间的情况下并不高效，因为每个进程都需要存储数据集的一个完整副本。
    dataset = datasets.CIFAR100(root=f'./data_{rank}', train=train, download=True, transform=transform)
    sampler = DistributedSampler(dataset, shuffle=train)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)
    return loader

def train(model, loader, optimizer, criterion, epoch, rank):
    model.train()
    # 设置DistributedSampler的epoch
    loader.sampler.set_epoch(epoch)
    for batch_idx, (data, targets) in enumerate(loader):
        data, targets = data.cuda(rank), targets.cuda(rank)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 每100个batch计算当前的损失，并在所有进程中进行聚合然后打印
        if (batch_idx + 1) % 100 == 0:
            # 将当前的loss转换为tensor，并在所有进程间进行求和
            loss_tensor = torch.tensor([loss.item()]).cuda(rank)
            dist.all_reduce(loss_tensor)

            # 计算所有进程的平均损失
            mean_loss = loss_tensor.item() / dist.get_world_size()  # 平均损失

            # 如果是rank 0，则打印平均损失
            if rank == 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx + 1}, Mean Loss: {mean_loss}")

def evaluate(model, dataloader, device):
    model.eval()
    local_preds = []
    local_targets = []

    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            local_preds.append(preds)
            local_targets.append(targets)

    # 将本地预测和目标转换为全局列表
    local_preds = torch.cat(local_preds)
    local_targets = torch.cat(local_targets)

    # 使用all_gather收集所有进程的预测和目标
    world_size = dist.get_world_size()
    gathered_preds = [torch.zeros_like(local_preds) for _ in range(world_size)]
    gathered_targets = [torch.zeros_like(local_targets) for _ in range(world_size)]
    
    dist.all_gather(gathered_preds, local_preds)
    dist.all_gather(gathered_targets, local_targets)
    
    # 只在rank 0进行计算和输出
    if dist.get_rank() == 0:
        gathered_preds = torch.cat(gathered_preds)
        gathered_targets = torch.cat(gathered_targets)
        accuracy = (gathered_preds == gathered_targets).float().mean()
        print(f"Global Test Accuracy: {accuracy.item()}")

def main_worker(rank, world_size, num_epochs):
    setup(rank, world_size)
    model = get_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_loader = get_dataloader(train=True)
    test_loader = get_dataloader(train=False)
    start_time = time.time()
    for epoch in range(num_epochs):  # num of epochs
        train(model, train_loader, optimizer, criterion, epoch, rank)
        evaluate(model, test_loader, rank)
    # 计时结束前同步所有进程，确保所有进程已经完成训练
    dist.barrier()
    duration = time.time() - start_time
    
    if rank == 0:
        print(f"Training completed in {duration:.2f} seconds")
    cleanup()

if __name__ == "__main__":
    world_size = 4 # 4块GPU
    num_epochs = 10 # 总共训练10轮
    # 采用mp.spawn启动
    mp.spawn(main_worker, args=(world_size,num_epochs), nprocs=world_size, join=True)

