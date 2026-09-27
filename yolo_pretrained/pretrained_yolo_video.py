import cv2
import numpy as np
import paths

#import video
file_video_stream = cv2.VideoCapture(paths.video_path)

while file_video_stream.isOpened():
    ret, current_frame = file_video_stream.read()
    #getting the frame
    img_to_detect = current_frame
    image_height = img_to_detect.shape[0]
    image_width = img_to_detect.shape[1]
    #converting the image to blob
    img_blob = cv2.dnn.blobFromImage(img_to_detect, 1/255, (416, 416), swapRB = True, crop = False)
    #class labels
    class_labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
                    "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
                    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
                    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
                    "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
                    "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
                    "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
                    "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
                    "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
                    "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]
    #setting the colors
    class_colors = ["0,0,255", "0,255,0", "255,0,0", "255, 255,0", "0, 255, 255"]
    class_colors = [np.array(color.split(",")).astype("int") for color in class_colors]
    class_colors = np.array(class_colors)
    class_colors = np.tile(class_colors, (16,1))
    #loading the model
    yolo_model = cv2.dnn.readNetFromDarknet(str(paths.config_path), str(paths.weights_path))
    #get the layers
    yolo_layers = yolo_model.getLayerNames()
    yolo_output_layer = [yolo_layers[int(yolo_layer) - 1] for yolo_layer in yolo_model.getUnconnectedOutLayers()]
    #set input
    yolo_model.setInput(img_blob)
    #go though each layer to get the output using forward method
    obj_detection_layers = yolo_model.forward(yolo_output_layer)

    #declare three lists
    class_ids_list = []
    boxes_list = []
    confidence_list = []

    #make two for loops 
    #we have multiple layers, and with each layer we have multiple detections
    # we will have multiple detections for each layer, loop through them to get our scores and the predicted class and confidence of prediction
    for object_detection_layer in obj_detection_layers:
        for object_detection in object_detection_layer:
            all_scores = object_detection[5:] #scores for all objects will be given to us from 5th index onwards
            #index 1 to 4 gives us the coordinates of the boudning box
            predicted_class_id = np.argmax(all_scores) #class id with the max score obtained
            prediction_confidence = all_scores[predicted_class_id] #prediction value of that class id - like 5% confident its x, 80% confident it's y

            if prediction_confidence > 0.50: #we will only take predictions with confidence more than 20%
                predicted_class_label = class_labels[predicted_class_id]
                #now we get our bounding box coordinates
                bounding_boxes = object_detection[0:4] * np.array([image_width, image_height, image_width, image_height])
                (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_boxes.astype("int")
                start_x_pt = int(box_center_x_pt - (box_width/2))
                start_y_pt = int(box_center_y_pt - (box_height / 2))
                #nms addition
                class_ids_list.append(predicted_class_id)
                confidence_list.append(float(prediction_confidence))
                boxes_list.append([start_x_pt, start_y_pt, int(box_width), int(box_height)])

        #applying nms using OpenCV method
        max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidence_list, 0,5, 0.4)
        #arguments are boxes list, confidence list, nms confidence, max supression threshold
        #return the values in a way that the 0th index value has the highest confidence value
        #returns the decreasing value
        
    for max_valueid in max_value_ids:
        max_class_id = max_valueid
        box = boxes_list[max_class_id]
        start_x_pt = box[0]
        start_y_pt = box[1]
        box_width = box[2]
        box_height = box[3]

        predicted_class_id = class_ids_list[max_class_id]
        predicted_class_labels = class_labels[predicted_class_id]
        prediction_confidence = confidence_list[max_class_id]

        end_x_pt = start_x_pt + box_width
        end_y_pt = start_y_pt + box_height

        #get a random mask color from the numpy array of colors
        box_color = class_colors[predicted_class_id]
        #convert the color numpy array as a list and apply to text and box
        box_color = [int(c) for c in box_color]
                            
        # print the prediction in console
        predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
        print("predicted object {}".format(predicted_class_label))
                    
        # draw rectangle and text in the image
        cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
        cv2.putText(img_to_detect, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
                
    cv2.imshow("Detection Output", img_to_detect)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#releasing the stream and the camera
#close all opencv windows
file_video_stream.release()
cv2.destroyAllWindows()




