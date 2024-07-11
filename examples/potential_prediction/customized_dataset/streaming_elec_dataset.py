from graphormer.data import register_dataset
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset, DataLoader
from graphormer.data.wrapper import preprocess_item
import copy
### Physical world data generater ###
def generate_electrons_iter_1(n_electrons):
    positions = np.random.uniform(-5, 5, (n_electrons, 3)) # 均匀分布
    charges = -np.random.randint(1, 6, size=(n_electrons))  # 每个原子带电1-5
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

class RandomElecDataset(Dataset):
    def __init__(
        self,
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
    ):
        self.n_electrons = n_electrons
        self.max_length = max_length
        self.batch_size = batch_size
        ### modify ###
        # positions, charges = generate_electrons_iter_1(self.n_electrons)
        # _, potential_energy = calculate_forces_and_potential_iter_1(positions, charges)
        # # 将数据转换为Tensor
        # charges_tensor = torch.tensor(charges, dtype=torch.long).unsqueeze(-1)
        # positions_tensor = torch.tensor(positions, dtype=torch.float)
        # energy_tensor = torch.tensor([potential_energy], dtype=torch.float)
        # self.dataset = Data(x=charges_tensor, pos=positions_tensor, y=energy_tensor)
        self.dataset = self.generate_dataset()
        # import pdb; pdb.set_trace()
        if self.dataset is not None:
            self.num_data = len(self.dataset.y)
        self.seed = seed
        if train_idx is None and train_set is None:
            train_valid_idx, test_idx = train_test_split(
                np.arange(self.num_data),
                test_size=self.num_data // 10,
                random_state=seed,
            )
            train_idx, valid_idx = train_test_split(
                train_valid_idx, test_size=self.num_data // 5, random_state=seed
            )
            self.train_idx = torch.from_numpy(train_idx)
            self.valid_idx = torch.from_numpy(valid_idx)
            self.test_idx = torch.from_numpy(test_idx)
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)
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
        all_charges = []
        all_positions = []
        all_energies = []
        data_list = []
        
        for _ in range(self.batch_size):
            positions, charges = generate_electrons_iter_1(self.n_electrons)
            _, potential_energy = calculate_forces_and_potential_iter_1(positions, charges)
            # Convert data to Tensor
            charges_tensor = torch.tensor(charges, dtype=torch.long).unsqueeze(-1)
            positions_tensor = torch.tensor(positions, dtype=torch.float)
            energy_tensor = torch.tensor([potential_energy], dtype=torch.float)
            
            data_list.append(Data(x=charges_tensor, pos=positions_tensor, y=energy_tensor))
        
        # Concatenate all the tensors along the batch dimension
        
        
        return Dataset(data_list)

        
    def index_select(self, idx):
        dataset = copy.copy(self)
        dataset.dataset = self.dataset.index_select(idx)
        if isinstance(idx, torch.Tensor):
            dataset.num_data = idx.size(0)
        else:
            dataset.num_data = idx.shape[0]
        dataset.__indices__ = idx
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_idx = None
        dataset.valid_idx = None
        dataset.test_idx = None
        return dataset

    def create_subset(self, subset):
        dataset = copy.copy(self)
        dataset.dataset = subset
        dataset.num_data = len(subset)
        dataset.__indices__ = None
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_idx = None
        dataset.valid_idx = None
        dataset.test_idx = None
        return dataset

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.dataset[idx]
            item.idx = idx
            item.y = item.y.reshape(-1)
            return preprocess_item(item)
        else:
            raise TypeError("index to a GraphormerPYGDataset can only be an integer.")

    def __len__(self):
        return self.num_data
    # def __init__(self, n_electrons, max_length=None):
    #     self.n_electrons = n_electrons
    #     self.max_length = max_length

    # def __len__(self):
    #     if self.max_length is not None:
    #         return self.max_length
    #     return 64
    
    # def __getitem__(self, index):
    #     positions, charges = generate_electrons_iter_1(self.n_electrons)
    #     _, potential_energy = calculate_forces_and_potential_iter_1(positions, charges)
    #     # 将数据转换为Tensor
    #     charges_tensor = torch.tensor(charges, dtype=torch.long).unsqueeze(-1)
    #     positions_tensor = torch.tensor(positions, dtype=torch.float)
    #     energy_tensor = torch.tensor([potential_energy], dtype=torch.float)

    #     return Data(x=charges_tensor, pos=positions_tensor, y=energy_tensor)




    

@register_dataset("streaming_elec_dataset")
def create_customized_dataset():
    n_electrons = 10  # 假设为10个电子
    batch_size = 64 # 你得骗
    dataset = RandomElecDataset(n_electrons, batch_size)
    import pdb; pdb.set_trace()
    num_graphs = len(dataset)

    # train_idx = np.arange(num_graphs)
    # valid_idx = []  # 空列表
    # test_idx = []  # 空列表
    train_valid_idx, test_idx = train_test_split(
        np.arange(num_graphs), test_size=num_graphs // 10, random_state=0
    )
    train_idx, valid_idx = train_test_split(
        train_valid_idx, test_size=num_graphs // 5, random_state=0
    )

    return {
        "dataset": dataset,
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "source": "pyg"
    }

### for debug ###
if __name__ == "__main__":
    # 定义并加载数据集
    dataset = RandomElecDataset(n_electrons=10, max_length=1000)
    import pdb; pdb.set_trace()
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 确认数据加载器工作正常
    for batch in data_loader:
        
        print(batch)
