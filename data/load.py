import copy

import numpy as np
from torch.utils.data import ConcatDataset
from torchvision import transforms

from data.available import AVAILABLE_DATASETS, AVAILABLE_TRANSFORMS, DATASET_CONFIGS
from data.HITS import HITSDataset
from data.manipulate import (
    ReducedDataset,
    SubDataset,
    TransformedDataset,
    permutate_image_pixels,
)

try:
    import medmnist
    from medmnist import INFO as MEDMNIST_INFO
except ImportError:
    medmnist = None
    MEDMNIST_INFO = None

def get_dataset(name, type='train', download=True, capacity=None, permutation=None, dir='./store/datasets',
                verbose=False, augment=False, normalize=False, target_transform=None, valid_prop=0.):
    '''Create [train|valid|test]-dataset.'''

    data_name = 'mnist' if name in ('mnist28') else name
    dataset_class = AVAILABLE_DATASETS[data_name]

    # specify image-transformations to be applied
    transforms_list = [*AVAILABLE_TRANSFORMS['augment']] if augment else []
    transforms_list += [*AVAILABLE_TRANSFORMS[name]]
    if normalize:
        transforms_list += [*AVAILABLE_TRANSFORMS[name+"_norm"]]
    if permutation is not None:
        transforms_list.append(transforms.Lambda(lambda x, p=permutation: permutate_image_pixels(x, p)))
    dataset_transform = transforms.Compose(transforms_list)

    # load data-set
    dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=False if type=='test' else True,
                            download=download, transform=dataset_transform, target_transform=target_transform)

    # if relevant, select "train" or "validation"-set from training-part of data
    # NOTE: this split assumes order of items in training-dataset is random!
    # (e.g., not first all samples from clas 1, then all samples from class 2, etc.)
    if (type=='train' or type=='valid') and valid_prop>0:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(valid_prop * dataset_size))
        if type=='train':
            indices_to_use = indices[split:]
        elif type=='valid':
            indices_to_use = indices[:split]
        dataset = ReducedDataset(dataset, indices_to_use)

    # print information about dataset on the screen
    if verbose:
        print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        dataset_copy = copy.deepcopy(dataset)
        dataset = ConcatDataset([dataset_copy for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset


##-------------------------------------------------------------------------------------------------------------------##


def get_singletask_experiment(name, data_dir="./store/datasets", normalize=False, augment=False, verbose=False):
    '''Load, organize and return train- and test-dataset for requested single-task experiment.'''

    # Define data-type
    if name == "MNIST":
        data_type = 'mnist'
    elif name == "MNIST28":
        data_type = 'mnist28'
    elif name == "CIFAR10":
        data_type = 'cifar10'
    elif name == "CIFAR100":
        data_type = 'cifar100'
    else:
        raise ValueError('Given undefined experiment: {}'.format(name))

    # Get config-dict and data-sets
    config = DATASET_CONFIGS[data_type]
    config['normalize'] = normalize
    if normalize:
        config['denormalize'] = AVAILABLE_TRANSFORMS[data_type+"_denorm"]
    trainset = get_dataset(data_type, type='train', dir=data_dir, verbose=verbose, normalize=normalize, augment=augment)
    testset = get_dataset(data_type, type='test', dir=data_dir, verbose=verbose, normalize=normalize)

    # Return tuple of data-sets and config-dictionary
    return (trainset, testset), config


def get_multitask_experiment(name, scenario, tasks, data_dir="./store/datasets", normalize=False, augment=False,
                             only_config=False, verbose=False, exception=False, only_test=False):
    '''Load, organize and return train- and test-dataset for requested multi-task experiment.
    
    Args
    -------
    name: str
        Name of the experiment.
    scenario: str
        Scenario for the experiment.
    tasks: int
        Number of tasks.
    data_dir: str
        Directory where the data is stored.
    normalize: bool
        Whether to normalize the data.
    augment: bool
        Whether to augment the data.
    only_config: bool
        Whether to return only the configuration.
    verbose: bool
        Whether to print verbose output.
    exception: bool
        Whether to use exception handling.
    only_test: bool
        Whether to return only the test dataset.

    Returns
    -------
    tuple
        Tuple of train and test datasets, and the configuration dictionary.
    '''

    ## NOTE: option 'normalize' and 'augment' only implemented for CIFAR-based experiments.

    # depending on experiment, get and organize the datasets
    if name == 'permMNIST':
        # configurations
        config = DATASET_CONFIGS['mnist']
        classes_per_task = 10
        if not only_config:
            # prepare dataset
            if not only_test:
                train_dataset = get_dataset('mnist', type="train", permutation=None, dir=data_dir,
                                            target_transform=None, verbose=verbose)
            test_dataset = get_dataset('mnist', type="test", permutation=None, dir=data_dir,
                                       target_transform=None, verbose=verbose)
            # generate permutations
            if exception:
                permutations = [None] + [np.random.permutation(config['size']**2) for _ in range(tasks-1)]
            else:
                permutations = [np.random.permutation(config['size']**2) for _ in range(tasks)]
            # specify transformed datasets per task
            train_datasets = []
            test_datasets = []
            for task_id, perm in enumerate(permutations):
                target_transform = transforms.Lambda(
                    lambda y, x=task_id: y + x*classes_per_task
                ) if scenario in ('task', 'class', 'all') else None
                if not only_test:
                    train_datasets.append(TransformedDataset(
                        train_dataset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                        target_transform=target_transform
                    ))
                test_datasets.append(TransformedDataset(
                    test_dataset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                    target_transform=target_transform
                ))
    elif name == 'splitMNIST':
        # check for number of tasks
        if tasks>10:
            raise ValueError("Experiment '{}' cannot have more than 10 tasks!".format(name))
        # configurations
        config = DATASET_CONFIGS['mnist28']
        classes_per_task = int(np.floor(10 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.array(list(range(10))) if exception else np.random.permutation(list(range(10)))
            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
            # prepare train and test datasets with all classes
            if not only_test:
                mnist_train = get_dataset('mnist28', type="train", dir=data_dir, target_transform=target_transform,
                                          verbose=verbose)
            mnist_test = get_dataset('mnist28', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario=='domain' else None
                if not only_test:
                    train_datasets.append(SubDataset(mnist_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(mnist_test, labels, target_transform=target_transform))
    elif name == 'CIFAR100':
        # check for number of tasks
        if tasks>100:
            raise ValueError("Experiment 'CIFAR100' cannot have more than 100 tasks!")
        # configurations
        config = DATASET_CONFIGS['cifar100']
        classes_per_task = int(np.floor(100 / tasks))
        if not only_config:
            # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
            permutation = np.random.permutation(list(range(100)))
            target_transform = transforms.Lambda(lambda y, x=permutation: int(permutation[y]))
            # prepare train and test datasets with all classes
            if not only_test:
                cifar100_train = get_dataset('cifar100', type="train", dir=data_dir, normalize=normalize,
                                             augment=augment, target_transform=target_transform, verbose=verbose)
            cifar100_test = get_dataset('cifar100', type="test", dir=data_dir, normalize=normalize,
                                        target_transform=target_transform, verbose=verbose)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split them up into sub-tasks
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(lambda y, x=labels[0]: y-x) if scenario=='domain' else None
                if not only_test:
                    train_datasets.append(SubDataset(cifar100_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(cifar100_test, labels, target_transform=target_transform))
    elif name == 'OrganMNIST':
        # OrganMNIST is defined as a domain-incremental benchmark with 3 tasks.
        if scenario != 'domain':
            raise ValueError("Experiment 'OrganMNIST' only supports scenario='domain'.")
        if tasks != 3:
            raise ValueError("Experiment 'OrganMNIST' requires exactly 3 tasks: OrganAMNIST, OrganCMNIST and OrganSMNIST.")
        # configurations
        config = copy.deepcopy(DATASET_CONFIGS[name.lower()])
        classes_per_task = len(MEDMNIST_INFO['organamnist']['label']) if MEDMNIST_INFO is not None else 11
        if not only_config:
            if medmnist is None or MEDMNIST_INFO is None:
                raise ImportError(
                    "Using OrganMNIST requires 'medmnist' to be installed. "
                    "Please install it (e.g., 'pip install medmnist')."
                )

            data_flags = ['organamnist', 'organcmnist', 'organsmnist']
            dataset_transform = transforms.Compose([*AVAILABLE_TRANSFORMS[name.lower()]])

            train_datasets = []
            test_datasets = []
            for data_flag in data_flags:
                info = MEDMNIST_INFO[data_flag]
                dataset_class = getattr(medmnist, info['python_class'])
                if not only_test:
                    train_datasets.append(dataset_class(
                        split='train', transform=dataset_transform, download=True, root=data_dir
                    ))
                test_datasets.append(dataset_class(
                    split='test', transform=dataset_transform, download=True, root=data_dir
                ))

                if verbose:
                    if not only_test:
                        print(" --> {}: 'train'-dataset consisting of {} samples".format(data_flag, len(train_datasets[-1])))
                    print(" --> {}: 'test'-dataset consisting of {} samples".format(data_flag, len(test_datasets[-1])))
    elif 'HITSsrc' in name:
            # HITS is defined as a domain-incremental benchmark with tasks based on sources or subjects
            if scenario != 'domain':
                raise ValueError("Experiment 'HITSsrc' only supports scenario='domain'.")
            
            # Parse name to get HITS base dataset:
            hits_base = name.split('_')[0]
            assert hits_base in HITSDataset.BASE_HITS_TASK_SRC.keys(), f"Unknown HITS base dataset: {hits_base}, should be one of {list(HITSDataset.BASE_HITS_TASK_SRC.keys())}"
            task_srcs = HITSDataset.BASE_HITS_TASK_SRC[hits_base].get('task_list', [])
            tasks = len(task_srcs)
            
            # Parse name to get task order if specified, and reorder tasks
            if "_task" in name:
                task_order = tuple(map(int, name.split('_task')[1].split('-')))
                assert len(task_order) == len(task_srcs), f"Task order length ({len(task_order)}) does not match task list length of base {hits_base}"
                assert set(task_order) == set(range(len(task_srcs))), f"Task order is not a permutation of the task list indices"
                task_srcs = [task_srcs[i] for i in task_order]

     
            # configurations
            config = copy.deepcopy(DATASET_CONFIGS['hitssrc'])
            classes_per_task = 3
            if not only_config:
                dataset_transform = transforms.Compose([*AVAILABLE_TRANSFORMS['hitssrc']])
                train_datasets = []
                test_datasets = []
                train_all = HITSDataset(
                    hdf5_path=data_dir + '/data_5k-srcCL-subsplit.hdf5',
                    split='train',
                    transform=dataset_transform
                    )
                test_all = HITSDataset(
                    hdf5_path=data_dir + '/data_5k-srcCL-subsplit.hdf5',
                    split='test',
                    transform=dataset_transform
                    )

                for sources in task_srcs:
                    if not only_test:
                        task_indices_train = []
                        for src in sources: # Loop on sources in a task
                            task_indices_train.extend(train_all.get_indices_by_src(src))
                        train_datasets.append(ReducedDataset(train_all, task_indices_train))
                    test_indices_test = []
                    for src in sources: # Loop on sources in a task
                        test_indices_test.extend(test_all.get_indices_by_src(src))
                    test_datasets.append(ReducedDataset(test_all, test_indices_test))
    
                    if verbose:
                        if not only_test:
                            print(" --> {}: 'train'-dataset consisting of {} samples".format(sources, len(train_datasets[-1])))
                        print(" --> {}: 'test'-dataset consisting of {} samples".format(sources, len(test_datasets[-1])))
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    # If needed, update number of (total) classes in the config-dictionary
    config['classes'] = classes_per_task if scenario=='domain' else classes_per_task*tasks
    config['normalize'] = normalize if name=='CIFAR100' else False
    if config['normalize']:
        config['denormalize'] = AVAILABLE_TRANSFORMS["cifar100_denorm"]

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    return config if only_config else ((train_datasets, test_datasets), config, classes_per_task)