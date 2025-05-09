
# Multilingual Emotion Detection in Voice

This project focuses on detecting emotions from speech across multiple languages using machine learning and deep learning techniques. It involves preprocessing audio data, extracting features, training a model, and evaluating its performance on emotion classification.

## Features

- Multilingual audio emotion recognition
- Feature extraction using MFCCs
- Classification using deep learning (e.g., CNN, RNN, or LSTM)
- Support for emotion categories like Happy, Sad, Angry, Neutral, etc.
- Visualization of training metrics and confusion matrix

## Contents

- `multilingual_emotion_detection_in_voice.ipynb`: Main Jupyter notebook containing the code and workflow.
- `requirements.txt`: List of required Python packages (optional, if available).
- `README.md`: Project overview and instructions.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.7+
- Jupyter Notebook or JupyterLab
- Pip

Install dependencies (if `requirements.txt` exists):

```bash
pip install -r requirements.txt
```

Or manually install key libraries:

```bash
pip install numpy pandas librosa matplotlib seaborn scikit-learn tensorflow
```

### Usage

1. Open the notebook:

```bash
jupyter notebook multilingual_emotion_detection_in_voice.ipynb
```

2. Run the cells in sequence to:

   - Load and preprocess audio data
   - Extract audio features (e.g., MFCCs)
   - Train a neural network model
   - Evaluate and visualize results

### Dataset

You will need a multilingual speech emotion dataset. Suggested datasets:

- **RAVDESS** (English)
- **EmoDB** (German)
- **CREMA-D**
- Combine or adapt datasets with labeled emotions across languages.

Make sure the dataset structure aligns with how the notebook loads the files.

## Model Architecture

The model uses deep learning (e.g., LSTM/CNN) for sequence modeling from extracted audio features. Details are available in the notebook.

## Results

The notebook includes:

- Accuracy and loss plots
- Confusion matrix
- Emotion-wise classification report

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Librosa for audio processing
- TensorFlow/Keras for model building
- Public datasets like RAVDESS and EmoDB
