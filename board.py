import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class Board:
    def __init__(self, run_name, enabled=True):
        self.enabled = enabled
        if not self.enabled:
            self.writer = None
        else:
            # Cria uma pasta com a data/hora exata para não misturar treinos
            time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_dir = os.path.join("runs", run_name, time_str)
            
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        if self.writer: self.writer.add_scalar(tag, value, step)

    def log_image(self, tag, image, step):
        if self.writer: self.writer.add_image(tag, image, step)

    def log_layer_gradients(self, model, epoch):
        if self.writer:
            for name, params in model.named_parameters():
                if params.grad is not None:
                    self.writer.add_histogram(f"Gradients/{name}", params.grad, epoch)

    def close(self):
        if self.writer: self.writer.close()