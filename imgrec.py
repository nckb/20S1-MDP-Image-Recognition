import darknet
import cv2
import numpy as np
import imutils
import random
import time

#Setup sending of string and receiving of coordinate
import socket
import threading
PORT = 5051
FORMAT = 'utf-8'
SERVER = '192.168.32.32'
ADDR = (SERVER, PORT)

#robot_coord = 'empty'

ir_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ir_socket.connect(ADDR)


WEIGHT_FILE_PATH = 'yolov4tiny.weights'
CONFIG_FILE_PATH = './cfg/custom-yolov4-tiny-detector.cfg'
DATA_FILE_PATH = './cfg/coco.data'
RPI_IP = '192.168.32.32'
MJPEG_STREAM_URL = 'http://' + RPI_IP + '/html/cam_pic_new.php'
YOLO_BATCH_SIZE = 4
THRESH = 0.85 #may want to lower and do filtering for specific images later

def retrieve_img():
    #captures a frame from mjpeg stream
    #returns opencv image
    cap = cv2.VideoCapture(MJPEG_STREAM_URL)
    ret, frame = cap.read()
    return frame

def image_detection(image, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    #Modified from darknet_images.py
    #Takes in direct image instead of path
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

def show_all_images(frame_list):
    for index, frame in enumerate(frame_list):
        frame = imutils.resize(frame, width=400)
        cv2.imshow('Image' + str(index), frame)

    if cv2.waitKey() & 0xFF == ord('q'):
        cv2.destroyAllWindows()

def leading_zero(int_string):
    if int(int_string) < 10:
        return '0' + int_string
    else:
        return int_string

def test_detect():
    frame = cv2.imread('C:\\Users\\CZ3004\\Downloads\\images\\multi_142.jpeg')
    #frame = retrieve_img()
    network, class_names, class_colors = darknet.load_network(
        CONFIG_FILE_PATH,
        DATA_FILE_PATH,
        WEIGHT_FILE_PATH,
        YOLO_BATCH_SIZE
    )

    image, detections = image_detection(frame, network, class_names, class_colors, THRESH)
    print(detections)
    cv2.imshow('Inference', image)
    if cv2.waitKey() & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    cv2.imwrite('./result.jpeg', image)

def continuous_detect():
    #use dictionary to store results
    #structure: dictionary, tuple of (id, confidence,(bbox))
    #bbox: x,y,w,h
    #global robot_coord
    #local_robot_coord = 'empty'
    #if robot_coord != 'empty':
    #    local_robot_coord = robot_coord
    #    robot_coord = 'empty'
    
    #local_robot_coord = '(1,1)|N'

    results = {}
    images = {}
    network, class_names, class_colors = darknet.load_network(
        CONFIG_FILE_PATH,
        DATA_FILE_PATH,
        WEIGHT_FILE_PATH,
        YOLO_BATCH_SIZE
    )
    try:
        print('Image recognition started!')
        while True:
            #print('Robot coordinates: ' + local_robot_coord)
            cv2.waitKey(50)
            frame = retrieve_img()
            image, detections = image_detection(frame, network, class_names, class_colors, THRESH)
            
            #structure: in a list, (id, confidence, (bbox))
            #[('9', '99.72', (377.555419921875, 147.49517822265625, 87.70740509033203, 173.86444091796875)), ('7', '99.95', (43.562461853027344, 134.47283935546875, 91.14225006103516, 181.6890411376953)), ('8', '99.96', (214.2314453125, 143.147216796875, 85.68460845947266, 166.68231201171875))]
            #index: 0-id 1-confidence 2-bbox
            #bbox: x,y,w,h
            for i in detections:
                id = i[0] #string
                confidence = i[1] #string
                bbox = i[2] #tuple
                print('ID detected: ' + id, ', confidence: ' + confidence)
                if id in results:
                    print('ID has been detected before')
                    if float(confidence) > float(results[id][1]):
                        print('Confidence higher. Replacing existing image.')
                        del results[id] #remove existing result from dict
                        del images[id] #remove existing img from dict
                        results[id] = i #add new result to dict. DUPLICATE ID IN VALUE PAIR!
                        images[id] = image #add new result to dict
                    else:
                        print('Confidence lower. Keeping existing image.')
                        pass
                else:
                    print('New ID. Saving to results and image dict.')
                    results[id] = i
                    images[id] = image
    except KeyboardInterrupt:
        print('End of image recognition.')
    
    #generate string
    img_rec_result_string = '{'
    print("Detection results:")
    
    for i in results:
        #here you would pull actual coordinates and compute
        #coordinates should already been loaded and accessible through a variable
        x_coordinate = random.randint(0,14)
        y_coordinate = random.randint(0,19)
        id_coordinate_str = '(' + i + ',' + str(x_coordinate) + ',' + str(y_coordinate) + '),'
        img_rec_result_string += id_coordinate_str

        # Android: NumberIDABXXYY
        # ANDROID STRING
        android_string ='NumberID'
        android_id = leading_zero(i)
        android_x = leading_zero(str(x_coordinate))
        android_y = leading_zero(str(y_coordinate))
        android_string += android_id + android_x + android_y
        
        # send string to android
        message = android_string.encode(FORMAT)
        ir_socket.send(message)
        print('Sent ' + android_string + ' to Android.')
        time.sleep(0.1)
        #finish send string to android

        print('ID: ' + i + ', Coordinates: (' + str(x_coordinate) +',' + str(y_coordinate) + ')' + ', Confidence: ' + results[i][1])

    if img_rec_result_string[-1] == ',':
        img_rec_result_string = img_rec_result_string[:-1]
    img_rec_result_string += '}'
    print(img_rec_result_string)
    android_string_all = 'ImageID' + img_rec_result_string
    message = android_string_all.encode(FORMAT)
    ir_socket.send(message)
    print('Sent ' + android_string_all + ' to Android.')

    #generate image mosaic
    result_frame_list = list(images.values())
    show_all_images(result_frame_list)

def readRPI():
    while True:
        msg = ir_socket.recv(1024)
        if msg:
            print('Received coordinates')
            robot_coord = msg

            

if __name__ == "__main__":
    #test_detect()
    #read_rpi_thread = threading.Thread(target = readRPI, name = "read_rpi_thread")
    #read_rpi_thread.daemon = True
    #print('Starting RPi comm thread...')
    #read_rpi_thread.start()
    #print('RPi comm thread started.')
    continuous_detect()