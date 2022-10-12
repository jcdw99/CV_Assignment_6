import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Q1_dataset import imgDataset
import Q1_admin
from sklearn.metrics import classification_report
from PIL import Image, ImageDraw, ImageFont
import numpy as np


    
if __name__ == "__main__":
    model = torch.load("../output/bestmod.pth")
    pics, labels = Q1_admin.get_data('test')
    testData = imgDataset(pics, labels)
    testDataLoader = DataLoader(testData, 1, num_workers=1, pin_memory=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes =  ['Coast', 'Forest', 'Highway', 'Kitchen', 'Mountain', 'Office', 'Store', 'Street', 'Suburb']
    intlabels = [classes.index(i) for i in labels]
    wrongtop7pics = []
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # initialize a list to store our predictions
        preds = []
        notintop3 = []
        # loop over the test set
        intop3 = 0
        
        bigdiffs = []
        bigdiffdexes = []
        # I identified that maxdiff is 12.235
        for (x, y) in testDataLoader:
            # send the input to the device
            x = x.to(device)
            # make the predictions and add them to the list
            pred = model(x)
            correct = int(y.numpy()[0])
            top1 = list(torch.topk(pred, k=1).indices.numpy().astype(int)[0])
            top3 = list(torch.topk(pred, k=3).indices.numpy().astype(int)[0])
            top2 = torch.topk(pred, k=2)
            if correct in top3:
                vals = top2.values.numpy()
                diff = np.max(vals) - np.min(vals)
                if diff > 9:
                    bigdiffs.append(Image.fromarray((255 * x.cpu().numpy()).astype(np.uint8)[0,0,:,:]))
                    bigdiffdexes.append(correct)
            else:
                print('Correct was', correct)
                print('\t\t but guessed')
                print(top3)
                print('\n\n')
                notintop3.append((correct, top1[0]))
                wrongtop7pics.append(Image.fromarray((255 * x.cpu().numpy()).astype(np.uint8)[0,0,:,:]))
            preds.extend(pred.argmax(axis=1).cpu().numpy())
     
        # print(notintop3)
        font_path = "../resources/InputSans-Regular.ttf"
        font = ImageFont.truetype(font_path, 15)
        # for index in range(len(wrongtop7pics)):
        #     i = wrongtop7pics[index].convert('RGB')
        #     draw = ImageDraw.Draw(i)            
        #     draw.text((int(i.size[0]/2), int(i.size[1]/20)), f'Correct: {classes[notintop3[index][0]]}' , fill=(0,255,0), align ="center", anchor="mm", font=font)
        #     draw.text((int(i.size[0]/2), int(i.size[1] - i.size[1]/20)), f'Prediction: {classes[notintop3[index][1]]}', fill="red", align ="center", anchor="mm", font=font)
        #     i.show()
        # print(f'{float(100*(intop3/len(testDataLoader)))}% top3 accuracy')

        for index in range(len(bigdiffs)):
            i = bigdiffs[index].convert('RGB')
            draw = ImageDraw.Draw(i)              
            draw.text((int(i.size[0]/2), int(i.size[1]/20)), f'Correct: {classes[bigdiffdexes[index]]}' , fill=(0,255,0), align ="center", anchor="mm", font=font)
            i.show()