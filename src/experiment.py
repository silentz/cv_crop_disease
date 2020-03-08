from collections import OrderedDict
from catalyst.dl.experiment import ConfigExperiment
import albumentations as A
from dataset import ZindiDataset
from aug import RandomTileAug


class Experiment(ConfigExperiment):
    @staticmethod
    def prepare_train_transforms(augs, tile):
        transforms = [
            RandomTileAug(augs, **tile),
            A.Normalize()
        ]
        return A.Compose(transforms)

    @staticmethod
    def prepare_test_transforms(augs, tile):
        transforms = [
            RandomTileAug(augs, **tile),
            A.Normalize(),
        ]
        return A.Compose(transforms)

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()

        params = self.stages_config[stage]['data_params']
        train_augs = self.get_transforms(stage, 'train')
        valid_augs = self.get_transforms(stage, 'valid')
        common_args = {
            'img_folder': params['img_folder'],
            'train_csv': params['train_csv'],
            'test_fold_number': params['test_fold_number'],
        }

        train = ZindiDataset(
            **common_args,
            is_test=False,
            transform=Experiment.prepare_train_transforms(train_augs, params.get('tile')),
        )

        valid = ZindiDataset(
            **common_args,
            is_test=True,
            transform=Experiment.prepare_test_transforms(valid_augs, params.get('tile'))
        )

        datasets['train'] = train
        datasets['valid'] = valid
        return datasets
