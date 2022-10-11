from copy import deepcopy
from typing import Any, Sequence
import torch


def base_distill_wrapper(Method=object):
    class BaseDistillWrapper(Method):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)

            self.output_dim = kwargs["output_dim"]

            self.frozen_encoder = deepcopy(self.encoder)
            self.frozen_projector = deepcopy(self.projector)

        def on_train_start(self):
            super().on_train_start()

            if self.current_task_idx > 0:

                self.frozen_encoder = deepcopy(self.encoder)
                
                # retrieve backupped weights for curriculum learning
                if self.ep_schedule is not None:
                    self._copy_encoder_curriculum()

                self.frozen_projector = deepcopy(self.projector)

                for pg in self.frozen_encoder.parameters():
                    pg.requires_grad = False
                for pg in self.frozen_projector.parameters():
                    pg.requires_grad = False

        @torch.no_grad()
        def frozen_forward(self, X):
            feats = self.frozen_encoder(X)
            return feats, self.frozen_projector(feats)

        def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
            _, (X1, X2), _ = batch[f"task{self.current_task_idx}"]

            out = super().training_step(batch, batch_idx)

            frozen_feats1, frozen_z1 = self.frozen_forward(X1)
            frozen_feats2, frozen_z2 = self.frozen_forward(X2)

            out.update(
                {"frozen_feats": [frozen_feats1, frozen_feats2], "frozen_z": [frozen_z1, frozen_z2]}
            )
            return out

        def _copy_encoder_curriculum(self):
            """
            Makes a full copy of the encoder taking into account also backupped layers.
            """
            self.frozen_encoder.layer3 = deepcopy(self.backup_layer3)
            
            if self.tiny_architecture:
                self.frozen_encoder.layer2 = deepcopy(self.backup_layer2)
            else:
                self.frozen_encoder.layer4 = deepcopy(self.backup_layer4)


    return BaseDistillWrapper
