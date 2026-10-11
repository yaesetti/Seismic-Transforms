from minerva.models.ssl.dinov2 import DinoV2

class DinoV2_Compat(DinoV2):
    def training_step(self, batch, batch_idx):
        data, label = batch
        if len(label.shape) == 4:
            label = label.squeeze(1)
        outputs = self.forward(data)
        loss = self.loss_fn(outputs, label)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        if len(label.shape) == 4:
            label = label.squeeze(1)
        outputs = self.forward(data)
        loss = self.loss_fn(outputs, label)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        data, label = batch
        if len(label.shape) == 4:
            label = label.squeeze(1)
        outputs = self.forward(data)
        loss = self.loss_fn(outputs, label)
        self.log("test_loss", loss)
        return loss