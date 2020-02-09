import os
import time
import torch
import shutil
import ignite

from ..module import Module
from pathlib import Path
from datetime import datetime
from ignite.engine import Events
from collections import defaultdict


class BestSaver(ignite.handlers.ModelCheckpoint, Module):
    '''
        A handler can be used to save the best objects according to a metric to disk.
        Parameters:
            dirname (str): directory path. A subfolder will be created in the directory. Your objects will be saved in this subfolder.
            score_name (str): name of a metric attached to the engine what defines how good objects are.
            mode (str): one of min, max. In min mode, the least score objects will be saved. In max mode, the best score objects will be saved.
        See Ignite ModelCheckpoint for more details about other parameters.
    '''
    def __init__(self, dirname, score_name, evaluator_name, mode='max', n_saved=1, atomic=True, require_empty=True, create_dir=True, save_as_state_dict=True):
        if mode not in ['min', 'max']:
            raise ValueError(f'mode must be min or max. mode value found is {mode}')
        
        dirname = os.path.join(dirname, datetime.now().strftime('%y%m%d%H%M'))
        
        super(BestSaver, self).__init__(dirname, 'best', score_function=lambda e: e.state.metrics[score_name] if mode == 'max' else - e.state.metrics[score_name], score_name=score_name, n_saved=n_saved, atomic=atomic, require_empty=require_empty, create_dir=create_dir, save_as_state_dict=save_as_state_dict)
        self.evaluator_name = evaluator_name

    def init(self):
        assert self.evaluator_name in self.frame, f'The frame does not have {self.evaluator_name}.'
        assert 'model' in self.frame, 'The frame does not have model.'
        self.frame[self.evaluator_name].engine.add_event_handler(Events.EPOCH_COMPLETED, self, {'model': self.frame['model']})

    def state_dict(self):
        return {
                'n_saved': self._n_saved,
                'saved': self._saved,
                'iteration': self._iteration
            }

    def load_state_dict(self, state_dict):
        self._n_saved = state_dict['n_saved']
        self._saved = state_dict['saved']
        self._iteration = state_dict['iteration']


class BackupSaver(ignite.handlers.ModelCheckpoint, Module):
    '''
        A handler can be used to periodically save objects to disk.
        Parameters:
            dirname (str): directory path. A subfolder will be created in the directory. Your objects will be saved in this subfolder.
        See Ignite ModelCheckpoint for more details about other parameters.
    '''
    class Checkpoint(object):
        def __init__(self, modules, frame):
            super(BackupSaver.Checkpoint, self).__init__()
            assert all(map(lambda x: x in frame, modules)), f'The frame does not have all {modules}'
            
            if 'last_epoch' in modules:
                raise ValueError('modules should have key last_epoch.')

            self.frame = frame
            self.modules = modules

        def state_dict(self):
            checkpoint = {module: self.frame[module].state_dict() for module in self.modules}
            checkpoint['last_epoch'] = self.frame['engine'].engine.state.epoch
            return checkpoint
    
    def __init__(self, modules, dirname, save_interval, n_saved=1, atomic=True, require_empty=True, create_dir=True, save_as_state_dict=True):
        dirname = os.path.join(dirname, datetime.now().strftime('%y%m%d%H%M'))
        super(BackupSaver, self).__init__(dirname, 'backup', save_interval=save_interval, n_saved=n_saved, atomic=atomic, require_empty=require_empty, create_dir=create_dir, save_as_state_dict=save_as_state_dict)
        self.modules = modules

    def init(self):
        assert 'engine' in self.frame, 'The frame does not have engine.'
        shutil.copy(self.frame.config_path, self._dirname)
        checkpoint = self.Checkpoint(self.modules, self.frame)
        self.frame['engine'].engine.add_event_handler(Events.EPOCH_COMPLETED, self, {'checkpoint': checkpoint})
        self.frame['engine'].engine.add_event_handler(Events.EPOCH_COMPLETED, self._correct_checpoint)

    def state_dict(self):
        return {
                'n_saved': self._n_saved,
                'saved': self._saved,
                'iteration': self._iteration
            }

    def load_state_dict(self, state_dict):
        self._n_saved = state_dict['n_saved']
        self._saved = state_dict['saved']
        self._iteration = state_dict['iteration']

    def _correct_checpoint(self, engine):
        checkpoint_path = self.frame['backup_saver']._saved[-1][1][0]
        checkpoint = torch.load(checkpoint_path)
        checkpoint['backup_saver']['saved'] = self.frame['backup_saver']._saved
        torch.save(checkpoint, checkpoint_path)


class CheckpointLoader(Module):
    def __init__(self, checkpoint_path, mode):
        super(CheckpointLoader, self).__init__()
        self.checkpoint_path = checkpoint_path
        if mode in ['train', 'resume', 'retrain', 'test']:
            self.mode = mode
        else:
            raise ValueError('mode must be resume, retrain or test.')


    def init(self):
        assert 'engine' in self.frame, 'The frame does not have engine.'
        self.frame['engine'].engine.add_event_handler(Events.STARTED, self._load_checkpoint, self.mode)

    def _load_checkpoint(self, engine, mode):
        if mode in ['resume']:
            checkpoint = torch.load(self.checkpoint_path)
            
            assert 'engine' in self.frame, 'The frame does not have engine.'
            self.frame['engine'].engine.state.epoch = checkpoint['last_epoch']
            del checkpoint['last_epoch']
            
            for module, state_dict in checkpoint.items():
                self.frame[module].load_state_dict(state_dict)

                if isinstance(self.frame[module], ignite.handlers.ModelCheckpoint):
                    self._symlink_checkpoint(self.frame[module])
        elif mode in ['retrain', 'test']:
            checkpoint = torch.load(self.checkpoint_path)
            self.frame['model'].load_state_dict(checkpoint)

    def _symlink_checkpoint(self, saver):
        for priority, saved_objects in saver._saved:
            for i, path in enumerate(saved_objects):
                path = Path(path)
                new_path = Path(saver._dirname).joinpath(path.name)
                new_path.symlink_to(path.resolve(), target_is_directory=True)
                saved_objects[i] = str(new_path)


class History(Module):
    def __init__(self):
        super(History, self).__init__()
        self.lr = []
        self.time = []
        self.epoch = []
        self.metric_values = defaultdict(lambda: defaultdict(list))

    def init(self):
        assert 'model' in self.frame, 'The frame does not have model.'
        assert 'optim' in self.frame, 'The frame does not have optim.'
        assert 'engine' in self.frame, 'The frame does not have engine.'
        assert 'metrics' in self.frame, 'The frame does not have metrics.'
        self.model = self.frame['model']
        self.optim = self.frame['optim']
        self.engine = self.frame['engine']
        self.metrics = self.frame['metrics']
        self.engine.engine.add_event_handler(Events.EPOCH_COMPLETED, self._update)

    def _update(self, engine):
        self.lr.append(self.optim.state_dict()['param_groups'][0]['lr'])
        self.time.append(time.time())
        self.epoch.append(engine.state.epoch)
        for eval_name, metrics in self.metrics.metric_values.items():
            for metric_name, metric_value in metrics.items():
                self.metric_values[eval_name][metric_name].append(metric_value)

    def state_dict(self):
        return {
            'lr': self.lr,
            'time': self.time,
            'epoch': self.epoch,
            'metric_values': dict(self.metric_values),
        }

    def load_state_dict(self, state_dict):
        self.lr = state_dict['lr']
        self.time = state_dict['time']
        self.epoch = state_dict['epoch']
        self.metric_values = defaultdict(lambda: defaultdict(list), state_dict['metric_values'])