import torch.nn.functional as F
from catalyst.dl.core import Callback, CallbackOrder
from sklearn.metrics import log_loss


class LogLoss(Callback):
    def __init__(self, targets = "targets", logits = "logits", prefix = "logloss"):
        self.targets = targets
        self.logits = logits
        self.prefix = prefix
        super().__init__(CallbackOrder.Metric)

    def on_batch_end(self, state):
        targets = state.input[self.targets].detach().cpu().numpy()
        logits = state.output[self.logits]
        logits = F.softmax(logits, 1)
        logits = logits.detach().cpu().numpy()
        final_score = log_loss(targets, logits, labels=[0, 1, 2])
        state.metric_manager.add_batch_value(name=self.prefix, value=final_score)
