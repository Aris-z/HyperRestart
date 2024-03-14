import haienv
haienv.set_env("hsgdr")
import numpy as np
import torch
from torch.utils.data import Sampler, DataLoader

class MultiTaskBatchSampler(Sampler):
    def __init__(self, dataset_sizes, batch_size, temperature, shuffle=True):
        self.dataset_sizes = dataset_sizes
        self.batch_size = batch_size
        self.temperature = temperature
        self.shuffle = shuffle

    def generate_tasks_distribution(self):
        total_size = sum(self.dataset_sizes)
        weights = np.array([(size / total_size) ** (1.0 / self.temperature) for size in self.dataset_sizes])
        weights = weights / np.sum(weights)
        return torch.as_tensor(weights, dtype=torch.double)

    def __iter__(self):
        indices = []
        for dataset_size in self.dataset_sizes:
            if self.shuffle:
                indices.append(torch.randperm(dataset_size).tolist())
            else:
                indices.append(list(range(dataset_size)))

        tasks_distribution = self.generate_tasks_distribution()
        num_batches_per_epoch = sum(self.dataset_sizes) // self.batch_size
        batch_task_assignments = torch.multinomial(tasks_distribution, num_batches_per_epoch, replacement=True)

        for batch_task in batch_task_assignments:
            num_task_samples = self.dataset_sizes[batch_task]
            selected_indices = torch.randint(low=0, high=num_task_samples, size=(self.batch_size,)).tolist()
            yield (self.dataset_sizes[batch_task] + torch.tensor(indices[batch_task])[selected_indices]).tolist()

    def __len__(self):
        return sum(self.dataset_sizes) // self.batch_size


if __name__ == '__main__':
    
    dataset_sizes = [100, 100, 100]
    batch_size = 10
    temperature = 1

    sampler = MultiTaskBatchSampler(dataset_sizes, batch_size, temperature)
    data_loader = DataLoader(range(sum(dataset_sizes)), batch_sampler=sampler)

    for batch in data_loader:
        print(batch)
