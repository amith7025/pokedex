# Pokedex - Pokémon Details Gaining App

Welcome to Pokedex, an app that leverages the power of PyTorch and Gradio with a Convolutional Neural Network (CNN) architecture to provide an interactive and user-friendly experience for exploring and learning about Pokémon.


## Getting Started

Follow these steps to set up and run the Pokedex app on your machine:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/[Your Username]/pokedex.git
    cd pokedex
    ```

2. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Pokedex app:**

    ```bash
    python code/main.py
    ```

    Visit [http://localhost:5000](http://localhost:5000) in your web browser to explore the Pokedex app.

## Features

### Convolutional Neural Network (CNN) Architecture

Our Pokedex app leverages a CNN architecture for accurate and efficient image recognition of Pokémon. The model is trained on a diverse dataset of Pokémon images to ensure reliable predictions.

### Data Processing

The `data_processing.py` script handles the processing of raw Pokémon data, preparing it for training the CNN model. This includes data cleaning, normalization, and augmentation.

### Gradio Integration

Gradio is seamlessly integrated into the app to provide a user-friendly interface for interacting with the Pokedex. Users can input images of Pokémon, and the app will predict and display details about the identified Pokémon.

## Screenshots

![Screenshot 1](screenshots/Screenshot_1.png)
*Caption: Pokémon details prediction with Pokedex.*

![Screenshot 2](screenshots/Screenshot_2.png)
*Caption: User-friendly interface for interacting with the app.*

## Contributing

We welcome contributions! If you'd like to enhance the Pokedex app or fix any issues, please submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Happy exploring with Pokedex! 🚀✨


