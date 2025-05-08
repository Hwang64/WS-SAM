from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log(self, scalar_dict, step):
        for key, value in scalar_dict.items():
            self.writer.add_scalar(key, value, step)

    def close(self):
        self.writer.close()
