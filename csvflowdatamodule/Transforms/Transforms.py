from torchvision import transforms as T
from argparse import ArgumentParser
from .TrAugmentFlow import TrAugmentFlow

class TransformsComposer():
    """
    Compose and setup the transforms depending command line arguments.
    Define a series of transforms, each transform takes a dictionnary
    containing a subset of keys from ['Flow', 'Image', 'GtMask'] and
    has to return the same dictionnary with content elements transformed.
    """
    def __init__(self, flow_augmentation) :
        transfs = []
        transfs.append(TrAugmentFlow(flow_augmentation))

        self.TrCompose = T.Compose(transfs)

    def __call__(self, ret) :
        return self.TrCompose(ret)

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = TrAugmentFlow.add_specific_args(parser)
        return parser
