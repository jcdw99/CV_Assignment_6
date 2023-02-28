
## Configuring the Image Directory
Please drop the image folder directly inside the resources/ directory. 
It should be inline with the font file.
## The Model
The CNN model is initialized in Q1_model.py, feel free to view the file, though it is instantiated in the Q1_train.py file.

## Training from scratch
Training from scratch is done in the Q1_train.py file. Just run the script.

## Top3 accuracy
For both questions 2 and 3, we gather top-n accuracy in the top3acc.py file. Simply load in the serialized version at the top of the file, and run.

## Transfer learning model
The transfer_features.py file is functionally equivalent to Q1_train.py, though for Question 3.

## Training the transfer learning model
The training of the AlexNet model is done in transfer_example.py. Simply run the file.

