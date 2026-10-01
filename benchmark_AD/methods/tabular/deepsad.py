from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim

from benchmark_AD.methods.tabular.prepare_traces_tabular import prepare_traces_tabular
from benchmark_AD.utils import log_results, predict_scores

class BaseTrainer(ABC):
    """Trainer base class."""

    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: tuple, batch_size: int,
                 weight_decay: float, device: str, n_jobs_dataloader: int):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader

    @abstractmethod
    def train(self, dataset, net):
        """
        Implement train method that trains the given network using the train_set of dataset.
        :return: Trained net
        """
        pass

    @abstractmethod
    def test(self, dataset, net):
        """
        Implement test method that evaluates the test_set of dataset on the given network.
        """
        pass

class AETrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Results
        self.train_time = None
        self.test_aucroc = None; self.test_aucpr = None
        self.test_time = None

    def train(self, dataset, ae_net):
        # logger = logging.getLogger()

        # Get train data loader
        train_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Set device
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        # logger.info('Starting pretraining...')
        # start_time = time.time()
        ae_net.train()
        for epoch in range(self.n_epochs):

            epoch_loss = 0.0
            n_batches = 0
            # epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                rec = ae_net(inputs)
                rec_loss = criterion(rec, inputs)
                loss = torch.mean(rec_loss)
                loss.backward()
                optimizer.step()
                scheduler.step()
                if epoch in self.lr_milestones:
                    # logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
                    pass

                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            # epoch_train_time = time.time() - epoch_start_time
            # logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        # f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        # self.train_time = time.time() - start_time
        # logger.info('Pretraining Time: {:.3f}s'.format(self.train_time))
        # logger.info('Finished pretraining.')

        return ae_net

    def test(self, dataset, ae_net):
        # logger = logging.getLogger()

        # Get test data loader
        test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Set device for network
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # Testing
        # logger.info('Testing autoencoder...')
        epoch_loss = 0.0
        n_batches = 0
        # start_time = time.time()
        idx_label_score = []
        ae_net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, _, idx = data
                inputs, labels, idx = inputs.to(self.device), labels.to(self.device), idx.to(self.device)

                rec = ae_net(inputs)
                rec_loss = criterion(rec, inputs)
                scores = torch.mean(rec_loss, dim=tuple(range(1, rec.dim())))

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss = torch.mean(rec_loss)
                epoch_loss += loss.item()
                n_batches += 1

        # self.test_time = time.time() - start_time

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_aucroc = roc_auc_score(labels, scores)
        # self.test_aucpr = average_precision_score(labels, scores, pos_label=1)

        # Log results
        # logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        # logger.info('Test AUCROC: {:.2f}%'.format(100. * self.test_aucroc))
        # logger.info('Test AUCPR: {:.2f}%'.format(100. * self.test_aucpr))
        # logger.info('Test Time: {:.3f}s'.format(self.test_time))
        # logger.info('Finished testing autoencoder.')


class BaseADDataset(ABC):
    """Anomaly detection dataset base class."""

    def __init__(self, root: str):
        super().__init__()
        self.root = root  # root path to data

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = None  # tuple with original class labels that define the normal class
        self.outlier_classes = None  # tuple with original class labels that define the outlier class

        self.train_set = None  # must be of type torch.utils.data.Dataset
        self.test_set = None  # must be of type torch.utils.data.Dataset

    @abstractmethod
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) ->tuple[DataLoader, DataLoader]:
        """Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set."""
        pass

    def __repr__(self):
        return self.__class__.__name__

class BaseTrainer(ABC):
    """Trainer base class."""

    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: tuple, batch_size: int,
                 weight_decay: float, device: str, n_jobs_dataloader: int):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader

class DeepSADTrainer(BaseTrainer):

    def __init__(self, c, eta: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Deep SAD parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.eta = eta

        # Optimization parameters
        self.eps = 1e-6

        # Results
        self.train_time = None
        self.test_aucroc = None; self.test_aucpr = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset: BaseADDataset, net):
        # logger = logging.getLogger()

        # Get train data loader
        train_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            # logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            # logger.info('Center c initialized.')

        # Training
        # logger.info('Starting training...')
        # start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):
            print(f"epoch {epoch+1}/{self.n_epochs}")
            epoch_loss = 0.0
            n_batches = 0
            # epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, semi_targets, _ = data
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

                # transfer the label "1" to "-1" for the inverse loss
                semi_targets[semi_targets==1] = -1

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                loss = torch.mean(losses)
                print(f"batch loss: {loss.item()}")
                loss.backward()
                optimizer.step()
                scheduler.step()
                if epoch in self.lr_milestones:
                    # logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
                    pass

                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            # epoch_train_time = time.time() - epoch_start_time
            # logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        # f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        # self.train_time = time.time() - start_time
        # logger.info('Training Time: {:.3f}s'.format(self.train_time))
        # logger.info('Finished training.')

        return net
    
    def test(self, dataset: BaseADDataset, net):
        # logger = logging.getLogger()

        # Get test data loader
        test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Testing
        # logger.info('Starting testing...')
        epoch_loss = 0.0
        n_batches = 0
        # start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, semi_targets, idx = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)
                idx = idx.to(self.device)

                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                loss = torch.mean(losses)
                scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        # self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        # labels = np.array(labels)
        scores = np.array(scores)
        # self.test_aucroc = roc_auc_score(labels, scores)
        # self.test_aucpr = average_precision_score(labels, scores, pos_label = 1)

        # Log results
        # logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        # logger.info('Test AUCROC: {:.2f}%'.format(100. * self.test_aucroc))
        # logger.info('Test AUCPR: {:.2f}%'.format(100. * self.test_aucpr))
        # logger.info('Test Time: {:.3f}s'.format(self.test_time))
        # logger.info('Finished testing.')

        return scores

    def init_center_c(self, train_loader: DataLoader, net, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

class Linear_BN_leakyReLU(nn.Module):
    """
    A nn.Module that consists of a Linear layer followed by BatchNorm1d and a leaky ReLu activation
    """

    def __init__(self, in_features, out_features, bias=False, eps=1e-04):
        super(Linear_BN_leakyReLU, self).__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features, eps=eps, affine=bias)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.linear(x)))

class BaseNet(nn.Module):
    """Base class for all neural networks."""

    def __init__(self):
        super().__init__()
        # self.logger = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None  # representation dimensionality, i.e. dim of the code layer or last layer

    def forward(self, *input):
        """
        Forward pass logic
        :return: Network output
        """
        raise NotImplementedError

    def summary(self):
        """Network summary."""
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        # self.logger.info('Trainable parameters: {}'.format(params))
        # self.logger.info(self)

class MLP(BaseNet):

    def __init__(self, x_dim, h_dims=[128, 64], rep_dim=32, bias=False):
        super().__init__()

        self.rep_dim = rep_dim

        neurons = [x_dim, *h_dims]
        layers = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=bias) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(layers)
        self.code = nn.Linear(h_dims[-1], rep_dim, bias=bias)

    def forward(self, x):
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        return self.code(x)


class MLP_Decoder(BaseNet):

    def __init__(self, x_dim, h_dims=[64, 128], rep_dim=32, bias=False):
        super().__init__()

        self.rep_dim = rep_dim

        neurons = [rep_dim, *h_dims]
        layers = [Linear_BN_leakyReLU(neurons[i - 1], neurons[i], bias=bias) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(layers)
        self.reconstruction = nn.Linear(h_dims[-1], x_dim, bias=bias)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)
        x = self.reconstruction(x)
        return self.output_activation(x)


def build_network(net_name, input_size ,ae_net=None):
    net = MLP(x_dim=input_size, h_dims=[100, 20], rep_dim=10, bias=False)

    return net

class MLP_Autoencoder(BaseNet):

    def __init__(self, x_dim, h_dims=[128, 64], rep_dim=32, bias=False):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = MLP(x_dim, h_dims, rep_dim, bias)
        self.decoder = MLP_Decoder(x_dim, list(reversed(h_dims)), rep_dim, bias)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def build_autoencoder(net_name, input_size):
    """Builds the corresponding autoencoder network."""
    
    ae_net = MLP_Autoencoder(x_dim=input_size, h_dims=[100, 20], rep_dim=10, bias=False)

    return ae_net

class deepsad(object):
    """A class for the Deep SAD method.

    Attributes:
        eta: Deep SAD hyperparameter eta (must be 0 < eta).
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network phi.
        trainer: DeepSADTrainer to train a Deep SAD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SAD network.
        ae_net: The autoencoder network corresponding to phi for network weights pretraining.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
        ae_results: A dictionary to save the autoencoder results.
    """

    def __init__(self, eta: float = 1.0):
        """Inits DeepSAD with hyperparameter eta."""

        self.eta = eta
        self.c = None  # hypersphere center c

        self.net_name = None
        self.net = None  # neural network phi

        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.results = {
            'train_time': None,
            'test_aucroc': None,
            'test_aucpr': None,
            'test_time': None,
            'test_scores': None,
        }

        self.ae_results = {
            'train_time': None,
            'test_aucroc': None,
            'test_aucpr': None,
            'test_time': None
        }

    def set_network(self, net_name, input_size):
        """Builds the neural network phi."""
        self.net_name = net_name
        self.net = build_network(net_name, input_size)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        """Trains the Deep SAD model on the training data."""

        self.optimizer_name = optimizer_name
        self.trainer = DeepSADTrainer(self.c, self.eta, optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs,
                                      lr_milestones=lr_milestones, batch_size=batch_size, weight_decay=weight_decay,
                                      device=device, n_jobs_dataloader=n_jobs_dataloader)
        # Get the model
        self.net = self.trainer.train(dataset, self.net)
        self.results['train_time'] = self.trainer.train_time
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get as list

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Deep SAD model on the test data."""

        if self.trainer is None:
            self.trainer = DeepSADTrainer(self.c, self.eta, device=device, n_jobs_dataloader=n_jobs_dataloader)

        score = self.trainer.test(dataset, self.net)

        # Get results
        # self.results['test_aucroc'] = self.trainer.test_aucroc
        # self.results['test_aucpr'] = self.trainer.test_aucpr
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

        return score

    def pretrain(self, dataset: BaseADDataset, input_size ,optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        """Pretrains the weights for the Deep SAD network phi via autoencoder."""

        # Set autoencoder network
        self.ae_net = build_autoencoder(self.net_name, input_size)

        # Train
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(dataset, self.ae_net)

        # Get train results
        self.ae_results['train_time'] = self.ae_trainer.train_time

        # Test
        self.ae_trainer.test(dataset, self.ae_net)

        # Get test results
        self.ae_results['test_aucroc'] = self.ae_trainer.test_aucroc
        self.ae_results['test_aucpr'] = self.ae_trainer.test_aucpr
        self.ae_results['test_time'] = self.ae_trainer.test_time

        # Initialize Deep SAD network weights from pre-trained encoder
        self.init_network_weights_from_pretraining()

    def init_network_weights_from_pretraining(self):
        """Initialize the Deep SAD network weights from the encoder weights of the pretraining autoencoder."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)


class ODDSDataset(Dataset):
    """
    ODDSDataset class for datasets_cc from Outlier Detection DataSets (ODDS): http://odds.cs.stonybrook.edu/

    Dataset class with additional targets for the semi-supervised setting and modification of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    def __init__(self, data, labels, train=True):
        super(Dataset, self).__init__()
        self.train = train

        if self.train:
            self.data = torch.tensor(data, dtype=torch.float32)
            self.targets = torch.tensor(labels, dtype=torch.int64)
        else:
            self.data = torch.tensor(data, dtype=torch.float32)
            self.targets = torch.tensor(labels, dtype=torch.int64)

        # self.semi_targets = torch.zeros_like(self.targets)
        self.semi_targets = self.targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """
        sample, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])

        return sample, target, semi_target, index

    def __len__(self):
        return len(self.data)


class ODDSADDataset(BaseADDataset):

    def __init__(self, data, labels, train):
        super().__init__(self)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = (0,)
        self.outlier_classes = (1,)

        # training or testing dataset
        self.train = train

        if self.train:
            # Get training set
            self.train_set = ODDSDataset(data=data, labels=labels, train=True)
        else:
            # Get testing set
            self.test_set = ODDSDataset(data=data, labels=labels, train=False)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> tuple[DataLoader, DataLoader]:

        if self.train:
            if len(self.train_set) < 2:
                raise ValueError("DeepSAD needs at least two training samples")
            batch_size = min(batch_size, len(self.train_set))
            train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                      num_workers=num_workers, drop_last=len(self.train_set) % batch_size == 1)
            return train_loader
        else:
            test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                     num_workers=num_workers, drop_last=False)
            return test_loader


def load_dataset(data, labels, train=True):
    """Loads the dataset."""

    # for tabular data
    dataset = ODDSADDataset(data=data, labels=labels, train=train)

    return dataset


def deepsad_train(args, X_train, y_train, X_test, y_test, train_ids, test_ids):
    # create labels: clean inlier set and a few labeled anomalies (inliers 0, outliers 1)
    
    outlier_indices = np.where(y_train == 1)[0]
    normal_indices = np.where(y_train == 0)[0]
    
    n_outliers_to_keep = int(len(outlier_indices) * args.perc_outliers_train)
    np.random.seed(args.seed)
    keep_outlier_indices = np.random.choice(
        outlier_indices, 
        size=n_outliers_to_keep, 
        replace=False
    )

    train_indices = np.concatenate([normal_indices, keep_outlier_indices])

    print(f"number of training outliers included: {len(keep_outlier_indices)} out of {len(outlier_indices)} total outliers in the training set")

    # Create new training set
    X_train_semi = X_train[train_indices]
    y_train_semi = y_train[train_indices]
    train_ids_semi = train_ids[train_indices]

    train_outlier_indices = np.where(y_train_semi == 1)[0]
    train_inlier_indices = np.where(y_train_semi == 0)[0]

    # modeling 
    eta = 1.0 # eta in the loss function
    optimizer_name = 'adam'
    lr = 0.001
    n_epochs = getattr(args, "epochs", None) or 50
    lr_milestone = [0]
    batch_size = 128
    weight_decay = 1e-6
    pretrain = True # whether to use auto-encoder for pretraining
    ae_optimizer_name = 'adam'
    ae_lr = 0.001
    ae_n_epochs = getattr(args, "epochs", None) or 100
    ae_lr_milestone = [0]
    ae_batch_size = 128
    ae_weight_decay = 1e-6
    num_threads = 0
    n_jobs_dataloader = 0
    net_name = 'dense'
    ae_device = args.device
    device = args.device

    input_size = X_train.shape[1]

    deepSAD = deepsad(eta)
    deepSAD.set_network(net_name, input_size)

    dataset = load_dataset(data=X_train_semi, labels=y_train_semi, train=True)
    
    if pretrain:
        deepSAD.pretrain(dataset,
                         input_size,
                         optimizer_name=ae_optimizer_name,
                         lr=ae_lr,
                         n_epochs=ae_n_epochs,
                         lr_milestones=ae_lr_milestone,
                         batch_size=ae_batch_size,
                         weight_decay=ae_weight_decay,
                         device=ae_device,
                         n_jobs_dataloader=0)
        
    deepSAD.train(dataset,
                        optimizer_name=optimizer_name,
                        lr=lr,
                        n_epochs=n_epochs,
                        lr_milestones=lr_milestone,
                        batch_size=batch_size,
                        weight_decay=weight_decay,
                        device=device,
                        n_jobs_dataloader=n_jobs_dataloader)
    
    dataset_test = load_dataset(data=X_test, labels=y_test, train=False)
    score = deepSAD.test(dataset_test)
    print(score)

    y_test_pred = predict_scores(args, score, y_test)

    f1 = f1_score(y_test, y_test_pred)
    acc = accuracy_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, score)
    bal_acc = balanced_accuracy_score(y_test, y_test_pred)

    print(f"DeepSVDD F1 Score: {f1:.4f}")
    print(f"DeepSVDD Accuracy: {acc:.4f}")
    print(f"DeepSVDD AUC: {auc:.4f}")
    print(f"DeepSVDD Balanced Accuracy: {bal_acc:.4f}")

    log_results(args, n_epochs, n_epochs+1, f1, acc, auc, bal_acc, model=None, gt_labels=y_test, pred_labels=y_test_pred.reshape(-1), pred_scores=score, train_ids=train_ids_semi, test_ids=test_ids)  # epoch_num for epoch



def run_deep_sad(args, train_df, test_df):
    print("Running DeepSAD")

    train_type = "semi_supervised"

    # read in data and convert raw traces to tabular format
    X_train, y_train, X_test, y_test, train_ids, test_ids = prepare_traces_tabular(args, train_type, train_df, test_df)
   
    deepsad_train(args, X_train, y_train, X_test, y_test, train_ids, test_ids)