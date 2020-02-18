#face detection library
import cv2

#harcascade file
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_cacade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

#starts camera
cap = cv2.VideoCapture(0)

while True:
    ret,img = cap.read()
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey)
    for face in  faces:
        x,y,w,h = face
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img,'face',(x,y),font,1,(255,255,255),2)


        rimg = img[y:y+h,x:x+w]
        rgrey = grey[y:y+h,x:x+w]
        eyes = eye_cacade.detectMultiScale(rgrey)
        for eye in eyes:
            ex,ey,ew,eh = eye
            cv2.rectangle(rimg,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('Face',img)
    cv2.waitKey(30)

cap.release()
cv2.destroyAllWindows()
