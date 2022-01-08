# OCR_MNIST


## The purpose of this repository

Familiarization with Text Recognition using Recurrent Neural Networks. The accuracy of the networks is not state of the art, but the main objective was to create a minimal size network for reasonable results. The number of parameters for the final version has about 85K parameters.

## Dataset

Images from the generated dataset. For a generation, only the vanilla **MNIST** dataset was used.

![Images_dataset](https://github.com/BioWar/OCR_MNIST/blob/main/images_readme/generated_images_dataset.png "Generated images")

Ground truth strings were encoded following the rules established in this work:
  - Numbers now have the prefix "**s_**"
  - Empty space is encoded with the symbol "**-**"
  - Space between numbers is encoded as "**|**"

![Transcription](https://github.com/BioWar/OCR_MNIST/blob/main/images_readme/encoded_strings.png "Transcription CSV file")

## Predictions

Prediction of the network on the test dataset is saved in an HTML file using the Jinja2 template. Here is an example of the saved file from the Results directory:

![Prediction_1](https://github.com/BioWar/OCR_MNIST/blob/main/images_readme/prediction_1.png "Prediction 1")
![Prediction_2](https://github.com/BioWar/OCR_MNIST/blob/main/images_readme/prediction_2.png "Prediction 2")
