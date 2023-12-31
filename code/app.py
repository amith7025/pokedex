import gradio as gr 
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T 
from PIL import Image

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


def prediction(img):
    transformed = test_transforms(img)
    prediction = model(transformed.unsqueeze(0))
    prob = torch.softmax(prediction,dim=1)
    output = torch.argmax(prob,dim=1)
    ans =  label[output.item()]
    content = df.loc[df['Pokemon'] == ans]
    pokemon_name, pokemon_type, attack, defense, hp, special_attack, special_defense, speed = content.values.flatten()

    return img,pokemon_name, pokemon_type, attack, defense, hp, special_attack, special_defense, speed


app_description = """
🌟 Welcome to the magical world of Pokémon with Amith E K's Pokedex app! Immerse yourself in this meticulously crafted application, your ultimate companion for all things Pokémon. 🚀

Explore the Pokedex's visually stunning interface, designed with precision and passion by Amith E K. This feature-rich app combines cutting-edge technology with an extensive Pokemon database, bringing your favorite creatures to life! 🎮

⚡️ Discover detailed information on each Pokemon, including type, attack, defense, HP, special attacks, special defense, and speed. The intuitive image input feature allows you to effortlessly identify Pokemon and dive deep into their fascinating details.

🌈 Whether you're a seasoned Pokemon Master or just starting your adventure, Amith E K's Pokedex is designed for you. Stay connected with your Pokemon journey, discover new species, and catch 'em all with style and ease.

🔍 Ready to embark on the ultimate Pokemon adventure? Download Amith E K's Pokedex app now and become the very best, like no one ever was! 🌟📱 #PokemonMaster #PokedexAdventure #GottaCatchEmAll
"""

demo = gr.Interface(
    fn=prediction,
    inputs=gr.Image(type='pil'),
    outputs=[
        gr.Image(type="pil", label="Pokemon"),
        gr.Text(label="Pokemon Name"),
        gr.Text(label="Type"),
        gr.Text(label="Attack"),
        gr.Text(label="Defense"),
        gr.Text(label="HP"),
        gr.Text(label="Special Attack"),
        gr.Text(label="Special Defense"),
        gr.Text(label="Speed")
    ],
    title='Pokedex',
    description=app_description,
    article='Created by Amith E K'
)

if __name__ == '__main__':
    demo.launch()