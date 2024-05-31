import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


def main():

    data_path = 'C:/Users/CHISOME/.spyder-py3/End/faces/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
    filename = 'C:\\Users\\CHISOME\\.spyder-py3\\End\\unrecognized_face\\unrecognized_face.avi'
    
    filemane = 'C:\\Users\\CHISOME\\.spyder-py3\\End\\recognized_faces\\recognized_faces.avi'
    
    
    codec = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    framerate = 30
    resolution = (640, 480)
    VideoFileOutput1 = cv2.VideoWriter(filename, codec, framerate, resolution)
    VideoFileOutput = cv2.VideoWriter(filemane, codec, framerate, resolution)
    
    Training_Data, Labels = [], []
    
    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images,dtype=np.uint8))
        Labels.append(i)
    
    Labels = np.asarray (Labels, dtype=np.int32)
    
    model = cv2.face.LBPHFaceRecognizer_create()
    
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    
    print("Model Training Complete!!!")
    

    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def face_detector(img,size = 0.5):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        
        if faces is():
    
            return img,[]
        
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
            roi = img[y:y+h,x:x+w]
            roi = cv2.resize(roi, (200,200))
            
        return img,roi
    cap = cv2.VideoCapture(0)
    
    i = 1
    while True:
       
            
        
        ret, frame = cap.read()
        
        image, face = face_detector(frame)
        
        
        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)
            
            if result[1] < 500:
                confidence = int(100*(1-(result[1])/300))
                display_string = str(confidence)+'% Confidence it is user'
                
            cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
            
            
            if confidence > 75:
                
                cv2.putText(image,"Recognized Image",(250, 450), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.imshow('Face Cropper', image)
                VideoFileOutput.write(frame)
                
            else:
                 cv2.putText(image,"Unrecognized Face",(250, 450), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255),2)
                 cv2.imshow('Face Cropper', image)
                 plt.imshow(image)
                
                 VideoFileOutput1.write(frame)
                 
                 
                 
                 
               
        except: 
            cv2.putText(image,"Face Not Found",(250, 450), cv2.FONT_HERSHEY_COMPLEX,1,(102,195,78),2)
            
            cv2.imshow('Face Cropper', image)
            pass
        
        i = i + 1
        if(i > 500 or cv2.waitKey(1)==27):
            break         
            
           
        
        
    cv2.destroyAllWindows()
    cap.release()
if __name__=="__main__":
    main()
        
    
 