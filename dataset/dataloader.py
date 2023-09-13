"""
Module for dataloader for SELD
"""
from torch.utils.data import Dataset

class SeldDataset(Dataset):
    """
    Chunk dataset for SELD task
    """
    def __init__(self, db_data, joint_transform=None, transform=None):
        super().__init__()
        self.features = db_data['features']
        self.sed_targets = db_data['sed_targets']
        self.doa_targets = db_data['doa_targets']
        self.chunk_idxes = db_data['feature_chunk_idxes']
        self.gt_chunk_idxes = db_data['gt_chunk_idxes']
        self.filename_list = db_data['filename_list']
        self.chunk_len = db_data['feature_chunk_len']
        self.gt_chunk_len = db_data['gt_chunk_len']
        self.joint_transform = joint_transform  # transform that change label
        self.transform = transform  # transform that does not change label
        self.n_samples = len(self.chunk_idxes)

    def __len__(self):
        """
        Total of training samples.
        """
        return self.n_samples

    def __getitem__(self, index):
        """
        Generate one sample of data
        """
        # Select sample
        chunk_idx = self.chunk_idxes[index]
        gt_chunk_idx = self.gt_chunk_idxes[index]

        # get filename
        filename = self.filename_list[index]

        # Load data and get label
        # (n_channels, n_timesteps, n_mels)
        X = self.features[:, chunk_idx: chunk_idx + self.chunk_len, :]
        # (n_timesteps, n_classes)
        sed_labels = self.sed_targets[gt_chunk_idx:gt_chunk_idx + self.gt_chunk_len]
        # (n_timesteps, x*n_classes) or (n_timesteps, x*n_classes, 2)
        doa_labels = self.doa_targets[gt_chunk_idx:gt_chunk_idx + self.gt_chunk_len]

        if self.joint_transform is not None:
            X, sed_labels, doa_labels = self.joint_transform(X, sed_labels, doa_labels)

        if self.transform is not None:
            X = self.transform(X)

        return X, sed_labels, doa_labels, filename
