
import torch
import torch.nn.functional as F
import torch
from torch.optim.lr_scheduler import StepLR
from multiprocessing import Pool, Manager, cpu_count
import torch.optim as optim

from image_util import DEVICE, IMAGE_SIZE, MIN_ANGLE_ZERO_OFFSET, EPOCHS, N_NN_OUTPUT_CLASSES, N_TRAIN_EXAMPLES, get_train_test_dataset, save_plot
from model import DeskewCNN

def train(model: torch.nn.Module, train_loader, optimizer, epoch):
  losses = []
  model.train()
  for idx, sample in enumerate(train_loader):
      x, y = sample['data'], sample['target']
      # optimizer.zero_grad()                                             
      hypothesis = model(x)
      loss = F.nll_loss(hypothesis, y)
      loss.backward()
      y_preds = hypothesis.argmax(dim=1)
      loss_val = int(loss * 1000) / 1000
      losses.append(loss_val)
      for y_idx in range(len(y_preds)):
        print(f"epoch {epoch}/{EPOCHS} loss {loss_val} predicted: {y_preds[y_idx].item() - MIN_ANGLE_ZERO_OFFSET} actual: {y[y_idx].item()- MIN_ANGLE_ZERO_OFFSET}")
      optimizer.step()
  return losses 

def test(model: torch.nn.Module, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample in test_loader:
            data, target = sample['data'].to(device), sample['target'].to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= max(len(test_loader.dataset), 1)
    print(f"nTest set: Average loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)")

def main():
  
  torch.manual_seed(42)

  model = DeskewCNN(N_NN_OUTPUT_CLASSES, IMAGE_SIZE)

  # criterion = torch.nn.CrossEntropyLoss(reduction='sum')                
  optimizer = optim.Adam(model.parameters(), lr=0.005)
  train_loader, test_loader = get_train_test_dataset()
  
  losses = []
  scheduler = StepLR(optimizer, step_size=1)
  for epoch in range(1, EPOCHS + 1):
    losses_ = train(model, train_loader, optimizer, epoch)
    losses.extend(losses_)
    test(model, test_loader, DEVICE)
    scheduler.step()
  save_plot(losses, [x*N_TRAIN_EXAMPLES for x in range(0,len(losses))])
  torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
  main()
