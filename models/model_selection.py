from argparse import ArgumentParser
from .AdaptedMobileNetV3 import AdaptedMobileNetV3
from .Resnet import Res8
from .EfficientNet import EfficientNet
from .SqueezeNet import SqueezeNet
from .MixNet import MixNet
from .ShuffleNet import ShuffleNetV2


class ModelSelection:
    """Select the model according to the name"""

    def __init__(self, model: str) -> None:
        # model is a string
        self.possible_models = {'mobilenetv3': AdaptedMobileNetV3,
                                'res8': Res8,
                                'efficientnet': EfficientNet,
                                'squeezenet': SqueezeNet,
                                'mixnet': MixNet,
                                'shufflenet': ShuffleNetV2,
                                }
        assert model in self.possible_models.keys()

        self.model = self.possible_models[model]

    def parse_params(self, parser: ArgumentParser) -> ArgumentParser:
        return self.model.add_model_specific_args(parser)

    def load(self, path: str, hparams_file: str = None, **kwargs):
        return self.model.load_from_checkpoint(path, hparams_file=hparams_file, **kwargs)

    def __call__(self, **kwargs):
        return self.model(**kwargs)
