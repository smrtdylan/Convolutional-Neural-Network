import torch
from PIL import Image
from torch import nn,save,load
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision import datasets, transforms # Import transforms
from torchvision.transforms import ToTensor, Resize

# Get + load data
transformation = transforms.Compose([
      transforms.ToTensor()

    ])

train = datasets.MNIST(
    root="data",
    download = True,
    transform = transformation
    )
dataset = DataLoader(train, 32)


# Core CLF NN
class ConvNet(nn.Module):
  def __init__ (self):
    super().__init__()

    self.initial_model = nn.Sequential (
        #this is def. overkill

        #layer 1
        nn.Conv2d(1,32,(3,3)),
        nn.ReLU(),

        #layer 2
        nn.Conv2d(32,64,(3,3)),
        nn.ReLU(),

        #layer 3
        nn.Conv2d(64,64,(3,3)),
        nn.ReLU(),

        #layer 4
        nn.Conv2d(64,128,(3,3)),
        nn.ReLU(),


        nn.Flatten(),
        nn.Linear(128*(28-8)*(28-8), 10)

    )
  def forward(self,x):
    return self.initial_model(x)


clf = ConvNet().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training flow
if __name__ == "__main__":
    for epoch in range(11): # train for 10 epochs
        for batch in dataset:
            X,y = batch
            X, y = X.to('cuda'), y.to('cuda')
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            # Apply backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch+1}: loss is {loss.item()}")

    with open('model_state.pt', 'wb') as f:
        save(clf.state_dict(), f)

#test the network
with open('model_state.pt','rb') as f:
      clf.load_state_dict(load(f))

      img = Image.open('img_1.jpeg').convert('L') # Convert image to grayscale

      #transforms to our network liking
      transformation = transforms.Compose([
      transforms.Resize((28,28)),
      transforms.ToTensor()
    ])
      img = transformation(img)
      img_tensor = (img).unsqueeze(0).to('cuda')
      img_tensor_convert = clf(img_tensor)
      print(torch.argmax(img_tensor_convert))
