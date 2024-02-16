import cv2
import numpy as np
import random
import sys
import datetime
import os
import time
import pytesseract
pytesseract.pytesseract.tesseract_cmd=r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
from ultralytics import YOLO
import torch
import pandas as pd


cap = cv2.VideoCapture("test_sample/vid.mp4")
model = YOLO("n-15-model.pt")
class_IDS = [0,1]
conf_level = 0.8 #default=0.8
dict_classes = model.model.names
# print(dict_classes)

def generator():

    N = 10
    n=[random.randint(0,sys.maxsize) for _ in range(N)]
    b=str(n[0])
    a=b[:8]
    
    return a

def number_plate_recog(img):
    # img = cv2.imread(img)

    img= cv2.resize(img,(450,80))
    cropnum = cv2.normalize(img.astype('float'), None, 100.0, 150.0, cv2.NORM_MINMAX)
    cropnum = cv2.cvtColor(np.uint8(cropnum), cv2.COLOR_BGR2RGB)
    # Apply adaptive thresholding and denoising
    gray = cv2.cvtColor(cropnum, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 5)
    thresh = cv2.adaptiveThreshold(blur, 180, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    cropnum = cv2.medianBlur(thresh, 5)
    crop_number_plate_file_path = os.path.join(f"./experiment/crop_img_number_plate/num{generator()}object.jpg")
    cv2.imwrite(crop_number_plate_file_path,cropnum, [cv2.IMWRITE_JPEG_QUALITY, 90])
    text_lp = pytesseract.image_to_string(cropnum, lang='eng', config='--psm 11') #---psm 7 ,#---psm 11
    if len(text_lp)>=10:
        # define the string to be cleaned
        string_to_clean = text_lp
        # define the list of characters to be removed
        chars_to_remove = ",:;\"'?]}{[`~!|\\/*_//-=+&^%$#@><.)(“°  —’§”∎□"
        # create a translation table with the characters to be removed
        translation_table = str.maketrans("", "", chars_to_remove)
        # use the translation table to remove the characters from the string
        cleaned_string = string_to_clean.translate(translation_table)
        # print the cleaned string
        # print(cleaned_string)  # output: Hello World Whats up
        # print("Number plate text is:", cleaned_string)

        data_1=f"time stamp of NPR {datetime.datetime.now()} and Number plate ---> {cleaned_string}"            
        with open("./number_plate_results.txt", "a") as text_file:
            text_file.write(data_1 + "\n")

        return cleaned_string , cropnum

# Create output folder if not exists
output_folder = "output_video"
os.makedirs(output_folder, exist_ok=True)

output_frames = "output_frames"
os.makedirs(output_frames, exist_ok=True)


# Define video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video_path = f"{output_folder}/{generator()}.MP4"
video_writer = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    # Read the frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the entire frame using YOLOv8
    # Getting predictions
    y_hat = model.predict(frame, conf = conf_level, classes = class_IDS, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), verbose = False)
    
    # Getting the bounding boxes, confidence and classes of the recognize objects in the current frame.
    boxes   = y_hat[0].boxes.xyxy.cpu().numpy()
    confs    = y_hat[0].boxes.conf.cpu().numpy()
    class_ids = y_hat[0].boxes.cls.cpu().numpy()
    # Storing the above information in a dataframe
    positions_frame = pd.DataFrame(y_hat[0].cpu().numpy().boxes.boxes, columns = ['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])
    print(positions_frame)
    #Translating the numeric class labels to text
    labels = [dict_classes[i] for i in class_ids]
    # Initialize a list to store the IDs of cropped objects
    processed_ids = []
##########################################################
    # For each people, draw the bounding-box and counting each one the pass thought the ROI area
    for ix, row in enumerate(positions_frame.iterrows()):
        # Getting the coordinates of each vehicle (row)
        xmin, ymin, xmax, ymax, confidence, category,  = row[1].astype('int')
        
        # Calculating the center of the bounding-box
        center_x, center_y = int(((xmax+xmin))/2), int((ymax+ ymin)/2)
        c_x = int(xmax+xmin)//2 
        c_y = int(ymax+ ymin)//2
        
        # Check if the object ID has already been processed
        obj_id = str(ix)
        if obj_id in processed_ids:
            continue


        cropped_img = frame[ymin:ymax, xmin:xmax]
        cv2.imshow("crop_object",cv2.resize(cropped_img,(400,100)))
        # Save the object crop to a file
        filename = F'./detect_object/img{str(generator())}.jpg'
        cv2.imwrite(filename, cropped_img)
        # Add the object ID to the list of processed IDs
        processed_ids.append(obj_id)


        # for number plate recognition
        # num,img = number_plate_recog(cropped_img)
        # print("NUMBER_palte is",num)
        # cv2.imshow("number_plate",img)


        # drawing center and bounding-box in the given frame 
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),(0,255,0), 2) # box

        # #Drawing above the bounding-box the name of class recognized.
        cv2.putText(img=frame, text= f'{labels[ix]}-{obj_id} : {str(np.round(confs[ix],2))}',
                    org= (xmin,ymin-10), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(20, 255, 255),thickness=1)
    
    # Save the frame to the output folder
        
    output_frames_path = os.path.join(output_frames, f"{generator()}.jpg")
    cv2.imwrite(output_frames_path, frame)


    # Write the frame to the output video
    video_writer.write(frame)
    
    cv2.imshow("original", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()