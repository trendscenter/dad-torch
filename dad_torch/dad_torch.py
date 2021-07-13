import json as _json
import os as _os
import pprint as _pp
import random as _random
import typing
from argparse import ArgumentParser as _AP

import numpy as _np
import torch as _torch
import torch.distributed as _dist
import torch.multiprocessing as _mp

import dad_torch.config as _conf
import dad_torch.utils as _utils
from dad_torch.config.state import *
from dad_torch.data import ETDataset, ETDataHandle
from dad_torch.trainer import NNTrainer
from dad_torch.utils.logger import *

_sep = _os.sep


def _dad_worker(rank, self, trainer_cls, dataset_cls, data_handle_cls):
    self.args['gpu'] = self.args['gpus'][rank]
    self.args['verbose'] = rank == MASTER_RANK
    world_size = self.args['world_size']
    if not world_size:
        world_size = self.args['num_gpus'] * self.args['num_nodes']
    world_rank = self.args['node_rank'] * self.args['num_gpus'] + rank

    self.args['is_master'] = world_rank == MASTER_RANK
    _dist.init_process_group(backend=self.args['dist_backend'],
                             init_method=self.args['dist_url'],
                             world_size=world_size, rank=world_rank)
    self._run(trainer_cls, dataset_cls, data_handle_cls)


class DADTorch:
    _MODES_ = [Phase.TRAIN, Phase.TEST]
    _MODE_ERR_ = \
        "####  [ERROR]  ### argument 'phase' is required and must be passed to either" \
        '\n\t1). DADTorch(..,phase=<value>,..)' \
        '\n\t2). runtime arguments 2). python main.py -ph <value> ...' \
        f'\nPossible values are:{_MODES_}'

    def __init__(self, dataspecs: typing.List[dict] = None,
                 args: typing.Union[dict, _AP] = _conf.default_args,
                 phase: str = _conf.default_args['phase'],
                 batch_size: int = _conf.default_args['batch_size'],
                 grad_accum_iters: int = _conf.default_args['grad_accum_iters'],
                 epochs: int = _conf.default_args['epochs'],
                 learning_rate: float = _conf.default_args['learning_rate'],
                 gpus: typing.List[int] = _conf.default_args['gpus'],
                 pin_memory: bool = _conf.default_args['pin_memory'],
                 num_workers: int = _conf.default_args['num_workers'],
                 dataset_dir: str = _conf.default_args['dataset_dir'],
                 load_limit: int = _conf.default_args['load_limit'],
                 log_dir: str = _conf.default_args['log_dir'],
                 pretrained_path: str = _conf.default_args['pretrained_path'],
                 verbose: bool = _conf.default_args['verbose'],
                 seed_all: int = _conf.default_args['seed_all'],
                 seed: int = _conf.default_args['seed'],
                 force: bool = _conf.default_args['force'],
                 patience: int = _conf.default_args['patience'],
                 load_sparse: bool = _conf.default_args['load_sparse'],
                 num_folds=_conf.default_args['num_folds'],
                 split_ratio=_conf.default_args['split_ratio'],
                 dataloader_args: dict = None,
                 **kw):
        """
        Order of precedence of arguments is(Higher will override the lower):
            1. Default args as in dad_torch.conf.default_args
            2. Listed args in __init__ method
            3. kwargs in **kw
        @param dataspecs: List of dict with which dataset details like data_files path, ground truth path...
                Example: [{'data_dir':'images', 'labels_dir':'manuals', 'splits_dir':'splits'}]
                Each key with _dir in it will be appended before the value provided in 'dataset_dir' argument.
        @param args: An argument parser, or, dict. (Defaults are loaded from dad_torch.conf.default_args.)
                    Note: values in args will be overridden by the listed args below if provided.
        @param phase: phase of operation; train/test. (Default: None)
                    train phase will run all train, validation, and test step.
        @param batch_size: Default is 32
        @param grad_accum_iters: Number of iterations to accumulate gradients. (Default 1)
        @param epochs: Default is 21
        @param learning_rate: Default is 0.001
        @param gpus: Default [0]. But set to [](or cpu) if no gpus found.
        @param pin_memory: Default is True if cuda found.
        @param num_workers: Default is 4.
        @param dataset_dir: Default is ''. Path to some dataset folder.
        @param load_limit: Load limit for data items for debugging pipeline with few data sample. Default is 1e11
        @param log_dir: Directory path to place all saved models, plots.... Default is net_logs/
        @param pretrained_path: Path to load pretrained model. Default is None
        @param verbose: Show logs? Default is True
        @param seed_all: If seeds to use for reproducibility. Default is False.
        @param force: Force to clear previous logs in log_dir(if any).
        @param patience: Set patience epochs to stop training. Uses validation scores. Default is 11.
        @param load_sparse: Loads test dataset in single data loader to recreate data(eg images) from prediction. Default is False.
        @param num_folds: Number of k-folds to split the data(eg folder with images) into. Default is None.
                        However, if a custom json split(s) are provided with keys train, validation,
                        test is provided in split_dir folder as specified in dataspecs, it will be loaded.
        @param split_ratio: Ratio to split files as specified in data_dir in dataspecs into. Default is 0.6, 0.2. 0.2.
                        However, if a custom json split(s) are provided with keys train, validation,
                        test is provided in split_dir folder as specified in dataspecs, it will be loaded.
        @param dataloader_args: dict with keys train, test, and validation that will ovveride corresponding dataloader args.
                For example, different batch size for validation loader.
        @param kw: Extra kwargs.
        """
        self._init_args(args)

        self.args.update(phase=phase)
        self.args.update(batch_size=batch_size)
        self.args.update(grad_accum_iters=grad_accum_iters)
        self.args.update(epochs=epochs)
        self.args.update(learning_rate=learning_rate)
        self.args.update(gpus=gpus)
        self.args.update(pin_memory=pin_memory)
        self.args.update(num_workers=num_workers)
        self.args.update(dataset_dir=dataset_dir)
        self.args.update(load_limit=load_limit)
        self.args.update(log_dir=log_dir)
        self.args.update(pretrained_path=pretrained_path)
        self.args.update(verbose=verbose)
        self.args.update(seed_all=seed_all)
        self.args.update(seed=seed)
        self.args.update(force=force)
        self.args.update(patience=patience)
        self.args.update(load_sparse=load_sparse)
        self.args.update(num_folds=num_folds)
        self.args.update(split_ratio=split_ratio)
        self.args.update(**kw)

        self.dataloader_args = dataloader_args if dataloader_args else {}
        assert (self.args.get('phase') in self._MODES_), self._MODE_ERR_

        self._init_dataspecs(dataspecs)

        self._device_check()
        self._ddp_setup()
        self._make_reproducible()
        self.args.update(is_master=self.args.get('is_master', True))

    def _device_check(self):
        self.args['gpus'] = self.args['gpus'] if self.args.get('gpus') else []
        if len(self.args['gpus']) > NUM_GPUS:
            self.args['gpus'] = list(range(NUM_GPUS))
            warn(f"{len(self.args['gpus'])} GPU(s) requested "
                 f"but {NUM_GPUS if CUDA_AVAILABLE else 'GPU(s) not'} detected. "
                 f"Using {str(NUM_GPUS) + ' GPU(s)' if CUDA_AVAILABLE else 'CPU(Much slower)'}.", self.args['verbose'])

        if self.args.get('world_size') is not None:
            self.args['gpus'] = [None] * self.args.get('world_size')

    def _ddp_setup(self):
        if all([self.args['use_ddp'], len(self.args['gpus']) >= 1]):
            self.args['num_gpus'] = len(self.args['gpus'])
            _os.environ['MASTER_ADDR'] = self.args.get('master_addr', '127.0.0.1')  #
            _os.environ['MASTER_PORT'] = self.args.get('master_port', '12355')
        else:
            self.args['use_ddp'] = False
            self.args['dad_reduction'] = "None"

    def _show_args(self):
        info('Starting with the following parameters:', self.args['verbose'])
        if self.args['verbose']:
            _pp.pprint(self.args)

    def _init_args(self, args):
        if isinstance(args, _AP):
            self.args = vars(args.parse_args())
        elif isinstance(args, dict):
            self.args = {**args}
        else:
            raise ValueError('2nd Argument of DADTorch could be only one of :ArgumentParser, dict')

    def _make_reproducible(self):
        if self.args['use_ddp'] and self.args['seed'] is None:
            raise ValueError('Seed must be explicitly given as seed=<seed> (Eg.1, 2, 101, 102) in DAD.')

        if self.args['seed'] is None:
            self.args['seed'] = CURRENT_SEED

        if self.args.get('seed_all'):
            _torch.manual_seed(self.args['seed'])
            _torch.cuda.manual_seed_all(self.args['seed'])
            _torch.cuda.manual_seed(self.args['seed'])
            _np.random.seed(self.args['seed'])
            _random.seed(self.args['seed'])
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False

    def _init_dataspecs(self, dataspecs):
        """
        Need to add -data(base folder for dataset) to all the directories in dataspecs.
        THis makes it flexible to access dataset from arbitrary location.
        """
        if dataspecs is None or len(dataspecs) == 0:
            dataspecs = [{'name': 'experiment'}]

        self.dataspecs = [{**dspec} for dspec in dataspecs]
        for dspec in self.dataspecs:
            if dspec.get('name') is None:
                raise ValueError('Each dataspecs must have a unique name.')

    def check_previous_logs(self, cache):
        r"""
        Checks if there already is a previous run and prompt[Y/N] so that
        we avoid accidentally overriding previous runs and lose temper.
        User can supply -f True flag to override by force.
        """
        if self.args['force']:
            warn('Forced overriding previous logs.', self.args['verbose'])
            return
        i = 'y'
        if self.args['phase'] == 'train':
            train_log = f"{cache['log_dir']}{_sep}{cache['experiment_id']}_log.json"
            if _os.path.exists(train_log):
                i = input(f"\n### Previous training log '{train_log}' exists. ### Override [y/n]:")

        if self.args['phase'] == 'test':
            test_log = f"{cache['log_dir']}{_sep}{cache['experiment_id']}_{LogKey.TEST_METRICS}.csv"
            if _os.path.exists(test_log):
                if _os.path.exists(test_log):
                    i = input(f"\n### Previous test log '{test_log}' exists. ### Override [y/n]:")

        if i.lower() == 'n':
            raise FileExistsError(f"Previous experiment logs path: '{self.args['log_dir']} is not empty."
                                  f"\n  Hint. Delete/Rename manually or Override(provide 'y' when prompted).")

    @staticmethod
    def _init_fold_cache(split_file, cache):
        """ Experiment id is split file name. For the example of k-fold. """
        """ Clear cache to save scores for each fold """

        cache[LogKey.TRAIN_LOG] = []
        cache[LogKey.VALIDATION_LOG] = []
        cache[LogKey.TEST_METRICS] = []

        cache['experiment_id'] = split_file.split('.')[0]
        cache['best_checkpoint'] = f"best_{cache['experiment_id']}_chk{CHK_EXT}"
        cache['latest_checkpoint'] = f"latest_{cache['experiment_id']}_chk{CHK_EXT}"
        cache.update(best_val_epoch=0, best_val_score=0.0)
        if cache['metric_direction'] == 'minimize':
            cache['best_val_score'] = MAX_SIZE

    def _global_experiment_end(self, trainer, scores: dict):
        "Save reduced scores"
        if scores is not None:
            """ Finally, save the global score to a file  """
            trainer.cache[LogKey.GLOBAL_TEST_METRICS].append(
                ['Global', *scores['averages'].get(), *scores['metrics'].get()])
            _utils.save_scores(trainer.cache, file_keys=[LogKey.GLOBAL_TEST_METRICS])

            with open(trainer.cache['log_dir'] + _sep + LogKey.SERIALIZABLE_GLOBAL_TEST + '.json', 'w') as f:
                log = {'averages': vars(scores['averages']),
                       'metrics': vars(scores['metrics'])}
                f.write(_json.dumps(log))

    def _train(self, trainer, train_dataset, validation_dataset, dspec):
        trainer.train(train_dataset, validation_dataset)
        trainer.save_checkpoint(trainer.cache['log_dir'] + _sep + trainer.cache['latest_checkpoint'])
        _utils.save_cache({**self.args, **trainer.cache, **dspec},
                          experiment_id=trainer.cache['experiment_id'])

    def _test(self, split_file, trainer, test_dataset) -> dict:
        best_exists = _os.path.exists(trainer.cache['log_dir'] + _sep + trainer.cache['best_checkpoint'])
        if best_exists and (self.args['phase'] == Phase.TRAIN or self.args['pretrained_path'] is None):
            """ Best model will be split_name.pt in training phase, and if no pretrained path is supplied. """
            trainer.load_checkpoint(trainer.cache['log_dir'] + _sep + trainer.cache['best_checkpoint'])

        """ Run and save experiment test scores """
        if test_dataset is not None:
            test_out = trainer.evaluation(mode='test', save_pred=True, distributed=False, dataset=test_dataset)
            test_scores = trainer.reduce_scores([test_out], distributed=False)
            trainer.cache[LogKey.TEST_METRICS] = [[split_file,
                                                   *test_scores['averages'].get(),
                                                   *test_scores['metrics'].get()]]
            _utils.save_scores(trainer.cache, experiment_id=trainer.cache['experiment_id'],
                               file_keys=[LogKey.TEST_METRICS])
            return test_out

    def run(self, trainer_cls: typing.Type[NNTrainer],
            dataset_cls: typing.Type[ETDataset] = None,
            data_handle_cls: typing.Type[ETDataHandle] = ETDataHandle):
        if self.args.get('use_ddp'):
            _mp.spawn(_dad_worker, nprocs=self.args['num_gpus'],
                      args=(self, trainer_cls, dataset_cls, data_handle_cls))
        else:
            self._run(trainer_cls, dataset_cls, data_handle_cls)

    def _run(self, trainer_cls, dataset_cls, data_handle_cls):
        r"""Run for individual datasets"""
        if self.args['verbose']:
            self._show_args()

        for dspec in self.dataspecs:

            data_handle = data_handle_cls(args=self.args, dataloader_args=self.dataloader_args)
            trainer = trainer_cls(args=self.args, data_handle=data_handle)
            trainer.init_nn(init_models=False, init_weights=False, init_optimizer=False)

            trainer.cache['log_dir'] = self.args['log_dir'] + _sep + dspec['name']
            trainer.data_handle.init_dataspec_(dspec)
            trainer.data_handle.create_splits(dspec, out_dir=trainer.cache.get('log_dir'))

            trainer.cache[LogKey.GLOBAL_TEST_METRICS] = []
            trainer.cache['log_header'] = 'Loss|Accuracy'
            trainer.cache.update(monitor_metric='time', metric_direction='maximize')

            """ Init and Run for each splits. """
            test_accum = []
            trainer.init_experiment_cache()
            _os.makedirs(trainer.cache['log_dir'], exist_ok=True)

            for split_file in sorted(_os.listdir(dspec['split_dir'])):
                self._init_fold_cache(split_file, trainer.cache)
                if self.args['is_master']:
                    self.check_previous_logs(trainer.cache)

                trainer.init_nn()
                if self.args['phase'] == Phase.TRAIN:
                    train_dataset = trainer.data_handle.get_train_dataset(split_file, dspec, dataset_cls=dataset_cls)
                    validation_dataset = trainer.data_handle.get_validation_dataset(
                        split_file, dspec, dataset_cls=dataset_cls)
                    self._train(trainer, train_dataset, validation_dataset, dspec)

                if self.args['is_master']:
                    test_dataset = trainer.data_handle.get_test_dataset(
                        split_file, dspec, dataset_cls=dataset_cls
                    )
                    test_accum.append(self._test(split_file, trainer, test_dataset))

                if trainer.args.get('use_ddp'):
                    _dist.barrier()

            if self.args['is_master']:
                global_scores = trainer.reduce_scores(test_accum, distributed=False)
                self._global_experiment_end(trainer, global_scores)

            if trainer.args.get('use_ddp'):
                _dist.barrier()