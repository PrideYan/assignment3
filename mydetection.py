#from jetson_inference import detectNet
#from jetson_utils import videoSource, videoOutput
import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
#camera = jetson.utils.videoSource("/dev/video0") # '/dev/video0' for V4L2
display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file

while display.IsStreaming(): # main loop will go here
    img = jetson.utils.loadImage("/home/nvidia/Pictures/2.png")
    if img is None: # capture timeout
        break
    detections = net.Detect(img)
    display.Render(img)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
    # print the detections
    print("detected {:d} objects in image".format(len(detections)))
    for detection in detections:
        print(detection)
    input("Press ENTER to continue.")

