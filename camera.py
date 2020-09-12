import cv2
import boto3
import datetime
import requests
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6

count=0

class VideoCamera(object):    
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        #count=0
        global count
        success, image = self.video.read()
        is_success, im_buf_arr = cv2.imencode(".jpg", image)
        image1 = im_buf_arr.tobytes()
        client=boto3.client('rekognition',
                        aws_access_key_id="ASIAQ7GL5D55ABUIEGXX",
                        aws_secret_access_key="z8D5GZIAO0Y/4Pt08GAuPkOpnpOAcOK4df8AA9XT",
                        aws_session_token="FwoGZXIvYXdzEGsaDFnHztFdhRXSPNd8aCLRAa/Bl934blxG4YMgeB9k9EShUjuWIw0n7+O9FC8y/URW7Ip5PV0NFWQJmrScnrSPs38vXLZnpo/jmh/emRKG9U4hJ5Ul/lxGDXMUIaVpUubdO/nM7sgbGmidts5SLdqRhbpQ9BF0iGb9cbgncOnJ8dNE/xH60rs+5wTwPdblQZYPQOYkFCus3Tchj8tFeq2TwCw9uZQP64M/l8HYfBYzntL26JIbpP/RM8WiVTZACd8Fb/AR87fRDAVX+gcq6J9UOudLYZNv8iIlESB3XGGv0Fn8KMvD6foFMi3ZXUFiKYivy91+Vrchv7o5XcSiUl7OZh/1NIasAYWXrKjdscLiM1V5s4jGq4w=",
                        region_name='us-east-1')
        response = client.detect_custom_labels(
        ProjectVersionArn='arn:aws:rekognition:us-east-1:066999623546:project/mask-detect/version/mask-detect.2020-09-09T14.13.59/1599641040321',Image={
            'Bytes':image1})
        print(response['CustomLabels'])
        
        if not len(response['CustomLabels']):
            count=count+1
            date = str(datetime.datetime.now()).split(" ")[0]
            #print(date)
            url = " https://p2p9kjuta8.execute-api.us-east-1.amazonaws.com/Main123/maskcount?date="+date+"&count="+str(count)
            resp = requests.get(url)
            f = open("countfile.txt", "w")
            f.write(str(count))
            f.close()
            #print(count)

        image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face_rects:
        	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        	break
        ret, jpeg = cv2.imencode('.jpg', image)
        #cv2.putText(image, text = str(count), org=(10,40), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(1,0,0))
        cv2.imshow('image',image)
        return jpeg.tobytes()
