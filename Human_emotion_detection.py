import cv2
import numpy as np
import pyttsx3
engine=pyttsx3.init()


net=cv2.dnn.readNet(r'C:\Users\Nanthu s\Downloads\emotion_check\yolov3_custom_last.weights',r'C:\Users\Nanthu s\Downloads\emotion_check\yolov3_custom.cfg')
classes=[]
with open(r'C:\Users\Nanthu s\Downloads\emotion_check\obj.names.txt') as f:
    classes=f.read().splitlines()
print(classes)
img=cv2.imread(r'C:\Users\Nanthu s\Downloads\emotion_check\Untitled design.jpg')
img = cv2.resize(img,(1250,600))


height,width,_ = img.shape

blob=cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0), swapRB=True, crop=False)


net.setInput(blob)

output_layers_names=net.getUnconnectedOutLayersNames()
layerOutputs= net.forward(output_layers_names)
boxes=[]
confidences=[]
class_ids=[]

for output in layerOutputs:
    for detection in output:
        scores=detection[5:]
        class_id=np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            centre_x= int(detection[0]*width)
            centre_y = int(detection[1]*height)
            w= int(detection[2]*width)
            h = int(detection[3]*height)

            x= int(centre_x-w/2)
            y=int(centre_y-h/2)

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# print(len(boxes))
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.4)

font=cv2.FONT_HERSHEY_COMPLEX_SMALL
colors=np.random.uniform(0,255,size=(len(boxes),3))

oo=[]       
for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (0,0,255), 2)
            #print(label)
            oo.append(label)
#print(oo)
for i in oo:
        
        if i=='fear':
            engine.say('fear face detected')
        elif i=='sad':
            engine.say('sad face detected')
        elif i=='smile':
            engine.say('smile face detected')
        elif i=='neutral':
            engine.say('neutral face detected')
        
        engine.runAndWait()
                        
                  

cv2.imshow('Image',img)
cv2.waitKey()
cv2.destroyAllWindows()
