from graphormer.data import register_dataset
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset, DataLoader
from graphormer.data.wrapper import preprocess_item
import copy
import itertools
### Physical world data generater ###
def generate_electrons_iter_1(n_electrons):
    positions = np.random.uniform(-5, 5, (n_electrons, 3)) # 均匀分布
    charges = np.random.randint(1, 10, size=(n_electrons))  # 每个原子带电1-5
    return positions, charges

def calculate_forces_and_potential_iter_1(positions, charges):
    k_e = 1  # 库伦常数，用了原子单位制
    n_electrons, _ = positions.shape
    forces = np.zeros_like(positions)
    potential_energy = np.float32(0.0)

    for i in range(n_electrons):
        for j in range(i + 1, n_electrons):
            r_vec = positions[j] - positions[i]
            r_mag = np.linalg.norm(r_vec)
            if r_mag == 0:
                continue  # 避免两个随机出来的坐标恰好完全相同
            r_hat = r_vec / r_mag # 从i指向j的向量

            force_magnitude = k_e * charges[i] * charges[j] / r_mag ** 2
            forces[i] -= force_magnitude * r_hat # 库伦排斥力 所以是反方向
            forces[j] += force_magnitude * r_hat # 
            potential_energy += k_e * charges[i] * charges[j] / r_mag

    return forces, potential_energy

class RandomElecDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        n_electrons,
        batch_size,
        max_length=None,
        seed: int = 0,
        train_idx=None,
        valid_idx=None,
        test_idx=None,
        train_set=None,
        valid_set=None,
        test_set=None,
        transform=None,
        pre_transform=None,
    ):
        self.n_electrons = n_electrons
        self.max_length = max_length
        self.batch_size = batch_size
        self.seed = seed
        super().__init__(root, transform, pre_transform)
        self.process()
        # import pudb; pudb.set_trace()
        if train_idx is None and train_set is None:
            
            train_valid_idx, test_idx = train_test_split(
                np.arange(self.num_data),
                test_size=self.num_data // 10,
                random_state=seed,
            )
            train_idx, valid_idx = train_test_split(
                train_valid_idx, test_size=self.num_data // 10, random_state=seed
            )
            self.train_idx = torch.from_numpy(train_idx)
            self.valid_idx = torch.from_numpy(valid_idx)
            self.test_idx = torch.from_numpy(test_idx)
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)
            # 1d find: train_data居然也是RandomElecDataset(64)
        elif train_set is not None:
            self.num_data = len(train_set) + len(valid_set) + len(test_set)
            self.train_data = self.create_subset(train_set)
            self.valid_data = self.create_subset(valid_set)
            self.test_data = self.create_subset(test_set)
            self.train_idx = None
            self.valid_idx = None
            self.test_idx = None
        else:
            self.num_data = len(train_idx) + len(valid_idx) + len(test_idx)
            self.train_idx = train_idx
            self.valid_idx = valid_idx
            self.test_idx = test_idx
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)
        self.__indices__ = None

    def generate_dataset(self):
        data_list = []
        for _ in range(self.batch_size):
            positions, charges = generate_electrons_iter_1(self.n_electrons)
            forces, potential_energy = calculate_forces_and_potential_iter_1(positions, charges)
            # Convert data to Tensor
            charges_tensor = torch.tensor(charges, dtype=torch.long).unsqueeze(-1)
            positions_tensor = torch.tensor(positions, dtype=torch.float)

            energy_tensor = torch.tensor([potential_energy], dtype=torch.float)
            edge_index = list(itertools.permutations(range(self.n_electrons), 2))
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.ones(edge_index.size(1), 1, dtype=torch.long)
            data = Data(x=charges_tensor,
                        charges=charges_tensor,
                        edge_index=edge_index, 
                        edge_attr=edge_attr, 
                        y=energy_tensor, 
                        pos=positions_tensor, 
                        tags=torch.ones_like(charges_tensor).squeeze(), 
                        real_mask=torch.ones_like(charges_tensor, dtype=torch.bool).squeeze(),
                        forces=torch.tensor(forces, dtype=torch.float))
            data_list.append(data)

        return data_list

    def process(self):
        print("processing the data")
        data_list = self.generate_dataset()
        self.data_list = data_list
        data, slices = self.collate(data_list) # 64*90 = 5760 = edge_attr个数
        torch.save((data, slices), self.processed_paths[0])
        self.data, self.slices = data, slices
        self.num_data = len(data_list)
        print("finish processing")

    def len(self):
        return self.num_data

    def __len__(self) -> np.int:
        return self.num_data

    def get(self, idx):
        data = self.data.__class__()
        if hasattr(self.data, '__num_nodes__'):
            data.__num_nodes__ = self.data.__num_nodes__[self.slices['x'][idx]:self.slices['x'][idx + 1]]
        for key in self.data.keys:
            item = self.data[key][self.slices[key][idx]:self.slices[key][idx + 1]]
            if torch.is_tensor(item):
                item = item.clone()
            data[key] = item
        return data

    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.data_list[idx]
            item.idx = idx
            item.y = item.y.reshape(-1)
            return preprocess_item(item)
        else:
            raise TypeError("index to a RandomElecDataset can only be an integer.")


    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass


@register_dataset("streaming_elec_dataset")
def create_customized_dataset():
    n_electrons = 10  # 假设为10个电子
    batch_size = 409600 # 你得骗
    
    dataset = RandomElecDataset(
        root='/tmp/RandomElecDataset',
        n_electrons=n_electrons,
        batch_size=batch_size
    )
    num_graphs = len(dataset)

    # Enhan changed here.
    train_idx = dataset.train_idx
    valid_idx = dataset.valid_idx
    test_idx = dataset.test_idx
    return {
        "dataset": dataset,
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "source": "pyg"
    }

### for debug ###
# if __name__ == "__main__":
#     # 定义并加载数据集
#     dataset = RandomElecDataset(n_electrons=10, max_length=1000)
#     import pdb; pdb.set_trace()
#     data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

#     # 确认数据加载器工作正常
#     for batch in data_loader:
        
#         print(batch)
