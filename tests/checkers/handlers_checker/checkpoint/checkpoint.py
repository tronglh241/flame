from pathlib import Path

import pytest

from ...checker import Checker


class BestCheckpointChecker(Checker):
    @property
    def command(self) -> str:
        config_file = Path(__file__).parent.joinpath('configs', 'best_checkpoint.yml')
        cmd = f'flame run {config_file}'
        return cmd

    def check(self) -> None:
        context = self.execute_command()

        assert context is not None
        best_checkpoint_accuracy = context['best_checkpoint_accuracy']
        best_checkpoint_precision = context['best_checkpoint_precision']

        assert (best_checkpoint_accuracy.checkpointer.save_handler
                .dirname.joinpath('best_model_1_accuracy=0.9000.pt').exists())
        assert (best_checkpoint_precision.checkpointer.save_handler
                .dirname.joinpath('best_model_1_precision=1.0000.pt').exists())


class BestCheckpointWrongModeChecker(Checker):
    @property
    def command(self) -> str:
        config_file = Path(__file__).parent.joinpath('configs', 'best_checkpoint_wrong_mode.yml')
        cmd = f'flame run {config_file}'
        return cmd

    def check(self) -> None:
        with pytest.raises(ValueError):
            self.execute_command()


class BackupCheckpointChecker(Checker):
    @property
    def command(self) -> str:
        config_file = Path(__file__).parent.joinpath('configs', 'backup_checkpoint.yml')
        cmd = f'flame run {config_file}'
        return cmd

    def check(self) -> None:
        context = self.execute_command()

        assert context is not None
        backup_checkpoint = context['backup_checkpoint']

        assert (backup_checkpoint.checkpointer.save_handler
                .dirname.joinpath('backup_checkpoint_1.pt').exists())


class CheckpointLoaderModelChecker(Checker):
    @property
    def command(self) -> str:
        config_file = Path(__file__).parent.joinpath('configs', 'checkpoint_loader_model.yml')
        cmd = f'flame run {config_file}'
        return cmd

    def check(self) -> None:
        import tempfile

        import torch
        from torchvision.models import MobileNetV2

        model = MobileNetV2(num_classes=10)
        torch.save(model.state_dict(), tempfile.gettempdir() + '/model.pt')
        context = self.execute_command()

        assert context is not None
        loaded_model = context['model']

        assert all(torch.all(p == lp) for p, lp in zip(model.parameters(), loaded_model.parameters()))
