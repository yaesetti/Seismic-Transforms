from minerva.models.nets.image.deeplabv3 import DeepLabV3
import torch

class SeismicModel(DeepLabV3):
    def configure_optimizers(self):
        self._set_trainable_params()

        optimizer = self.optimizer(
            self.parameters(),
            **self.optimizer_kwargs,
        )

        if self.lr_scheduler is None:
            return optimizer

        scheduler = self.lr_scheduler(
            optimizer,
            **self.lr_scheduler_kwargs
        )

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1,
                },
            }

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
