# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 19:12:13 2020

@author: Sai Nidhi
"""

import cv2
import boto3

global count
count=0

video = cv2.VideoCapture(0)

while(1):
    success, frame = video.read()
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame',(600,600))
    is_success, im_buf_arr = cv2.imencode(".jpg", frame)
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
        print(count)
    
    cv2.putText(frame, text = str(count), org=(10,40), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(1,0,0))
    cv2.imshow('frame',frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
        
video.release()
cv2.destroyAllWindows()
