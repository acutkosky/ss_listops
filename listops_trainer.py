
import torch
import listops_loader
import listops_model
import fftconv
from tqdm import tqdm
import torch.nn.functional as F

class TrainerConfig:
    datadir = None # this will cause the loader to use the DATA_DIR environment variable.
    batch_size = 32
    l_max = 2048
    lr = 0.001

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k,v)

class Trainer:

    def __init__(self, config, model_config, listops=None):
        self.config = config
        if listops is None:
            self.listops = listops_loader.listops_dataloader(config.datadir, config.batch_size, config.l_max)
        else:
            self.listops = listops
        vocab = self.listops.vocab

        self.model = listops_model.ListOpsModel(model_config, vocab)




        self.device = 'cpu'# if no GPU then it is cpu
        if torch.cuda.is_available():
            #torch.cuda.empty_cache()
            # print("torch._C._cuda_getDevice()",torch._C._cuda_getDevice())
            # print("torch.cuda.current_device()", torch.cuda.current_device() )
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            #with torch.cuda.device(0):
            self.device = torch.cuda.current_device() # if there is GPU then use GPU  #torch.device("cuda")#

            # print("in training loop self.device", self.device)
            #self.model = torch.nn.DataParallel(self.model)#.to(self.device)
            self.model.to(self.device)
            #self.model.to(1)

        self.model = torch.nn.DataParallel(self.model)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)




    def test_epoch(self):
        test_loader = self.listops.get_dataloader('test')
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        running_loss = 0.0
        running_accuracy = 0.0

        for it, (x, targets, lengths) in enumerate(pbar):

            x.to(self.device)  # [B, L]
            targets.to(self.device) # [B]
            lengths.to(self.device) # [B]

            B, L = x.size()

            logits, probabilities = self.model(x, lengths) # [B, C]

            loss = F.cross_entropy(probabilities, targets)

            accuracy = torch.sum(torch.argmax(probabilities, dim=1) == targets)/B

            running_loss += (loss.item - running_loss)/(it+1.0)
            running_accuracy += (accuracy.item - running_accuracy)/(it+1.0)

            pbar.set_description(f"Testing:, running_loss: {running_loss}, running accuracy: {running_accuracy}")

        print(f"Test loss: {running_loss}, test accuracy: {running_accuracy}")

    def train_epoch(self, epoch):

        train_loader = self.listops.get_dataloader('train')

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))

        device = self.device

        running_loss = 0.0
        running_accuracy = 0.0

        for it, (x, targets, lengths) in pbar:

            x = x.to(device)  # [B, L]
            targets = targets.to(device) # [B]
            lengths = lengths.to(device) # [B]

            B, L = x.size()

            logits = self.model(x, lengths) # [B, C]

            log_softmax = F.log_softmax(logits, dim=1) # [B, C]

            loss = F.nll_loss(log_softmax, targets) # [B]

            accuracy = torch.sum(torch.argmax(logits, dim=1) == targets)/B


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += (loss.item() - running_loss)/(it+1.0)
            running_accuracy += (accuracy.item() - running_accuracy)/(it+1.0)

            pbar.set_description(f"Epoch: {epoch+1}, running_loss: {running_loss}, running accuracy: {running_accuracy}")


    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.train_epoch(epoch)
            self.test_epoch()




