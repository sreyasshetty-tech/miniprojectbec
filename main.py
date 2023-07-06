#import cv2
#import pyttsx3

wCam, hCam = 640, 480

cap=cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

classNames = []
classFile ='coco.names'
with open(classFile,'rt') as f:
    classNames = [line.rstrip() for line in f]

configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weigtsPath='frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weigtsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)
engine = pyttsx3.init()
while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=0.6)
    print(classIds, bbox)
    # if classIds==1:
    #     print("Person is detected")
    # elif classIds==2:
    #     print("Bicycle is detected")
    # elif classIds==3:
    #     print("Car is detected")
    # elif classIds==4:
    #     print("Motorcycle is detected")
    # elif classIds==6:
    #     print("Bus is detected")
    # elif classIds==7:
    #     print("Train is detected")
    # elif classIds==8:
    #     print("Truck is detected")
    # elif classIds==9:
    #     print("Boat is detected")
    # elif classIds==15:
    #     print("Bench is detected")
    # elif classIds==26:
    #     print("Hat is detected")
    # elif classIds==27:
    #     print("Backpack is detected")
    # elif classIds==29:
    #     print("Shoe is detected")
    # elif classIds==30:
    #     print("Eye glasses is detected")
    # elif classIds==31:
    #     print("Bagpack is detected")
    # elif classIds==33:
    #     print("Bagpack is detected")
    # elif classIds==38:
    #     print("Kite is detected")
    # elif classIds==44:
    #     print("Bottle is detected")
    # elif classIds==45:
    #     print("Plate is detected")
    # elif classIds==46:
    #     print("Wine glass is detected")
    # elif classIds==47:
    #     print("Cup is detected")
    # elif classIds==48:
    #     print("Fork is detected")
    # elif classIds==49:
    #     print("Knife is detected")
    # elif classIds==50:
    #     print("Spoon is detected")
    # elif classIds==51:
    #     print("Bowl is detected")
    # elif classIds==62:
    #     print("Chair is detected")
    # elif classIds==64:
    #     print("Potted Plant is detected")
    # elif classIds==66:
    #     print("Mirror is detected")
    # elif classIds==67:
    #     print("Dining Table is detected")
    # elif classIds==68:
    #     print("Window is detected")
    # elif classIds==69:
    #     print("Desk is detected")
    # elif classIds==71:
    #     print("Door is detected")
    # elif classIds==72:
    #     print("TV is detected")
    # elif classIds==73:
    #     print("Laptop is detected")
    # elif classIds==74:
    #     print("Mouse is detected")
    # elif classIds==77:
    #     print("Cell Phone is detected")
    # elif classIds==83:
    #     print("Blender is detected")
    # elif classIds==84:
    #     print("Book is detected")
    # elif classIds==85:
    #     print("Clock is detected")
    # elif classIds==86:
    #     print("Vase is detected")
    # elif classIds==87:
    #     print("Scissors is detected")
    # elif classIds==89:
    #     print("Hair Drier is detected")
    # elif classIds==90:
    #     print("Toothbrush is detected")
    # else:
    #     print("No Object detected")


    if len(classIds) !=0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            str1 = str(className)
            print(str1 + " is detected")
            engine.say(str1 + "detected")
            engine.runAndWait()
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId-1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("output", img)
            cv2.waitKey(1)