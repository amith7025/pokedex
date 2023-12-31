import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T 
from PIL import Image
import pandas as pd


df = pd.read_csv('data.csv')

label = {
        0: 'Voltorb',
        1: 'Psyduck',
        2: 'Blastoise',
        3: 'Charizard',
        4: 'Pikachu',
        5: 'Wartortle',
        6: 'Nidoking',
        7: 'Pidgeot',
        8: 'Gengar',
        9: 'Kingler',
        10: 'Diglett',
        11: 'Electrode',
        12: 'Poliwag',
        13: 'Dragonite',
        14: 'Tauros',
        15: 'Magnemite',
        16: 'Bulbasaur',
        17: 'Golem',
        18: 'Jigglypuff',
        19: 'Abra'
}

test_transforms = T.Compose([
    T.Resize(size=(128,128)),
    T.ToTensor()
])

class PokemonModel(nn.Module):
    def __init__(self,input_size=3,output_size=len(label)):
        super().__init__()
        self.conv_blk1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size,out_channels=32,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=238144,out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32,out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16,out_features=output_size)
        )

    def forward(self,x):
        x = self.conv_blk1(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x
    
model = PokemonModel()
model.load_state_dict(torch.load('final.pth',map_location=torch.device('cpu')))

def prediction(text):
    img = Image.open(text)
    transformed = test_transforms(img)
    prediction = model(transformed.unsqueeze(0))
    prob = torch.softmax(prediction,dim=1)
    output = torch.argmax(prob,dim=1)
    return label[output.item()]

if __name__ == '__main__':
    text = input('enter the location of img')
    output = prediction(text)
    print(df.loc[df['Pokemon'] == output])