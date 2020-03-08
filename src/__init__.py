from catalyst.dl import registry, SupervisedRunner as Runner
from experiment import Experiment
from model import ZindiModel
from callbacks import LogLoss


registry.Model(ZindiModel)
registry.Callback(LogLoss)
