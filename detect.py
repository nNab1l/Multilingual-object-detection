import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
from googletrans import Translator
from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator



class ObjectDetection:

    def __init__(self, capture_index):
       
        self.capture_index = capture_index
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model()
        
        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = BoxAnnotator(color=ColorPalette(), thickness=3, text_thickness=3, text_scale=1.5)
        self.translator = Translator()
        self.translated_class_names = self.translate_class_names()
    

    def load_model(self):
       
        model = YOLO("weights/yolov8n.pt") 
        model.fuse()
    
        return model


    def predict(self, frame):
       
        results = self.model(frame)
        
        return results
    
    def translate_class_names(self):
        class_names = list(self.CLASS_NAMES_DICT.values())
        translated_class_names = [self.translate(name) for name in class_names]
        return translated_class_names

    def translate(self, text):
        return self.translator.translate(text, dest='de').text

    def preprocess_translations(self, detections):
        translated_labels = [
            f"{self.translated_class_names[class_id]} {confidence:.2f}"
            for _, confidence, class_id, _ in detections
        ]
        return translated_labels
    

    def plot_bboxes(self, results, frame):
        
        xyxys = []
        confidences = []
        class_ids = []
        
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)
            
            if class_id == 0:
                
                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(result.boxes.conf.cpu().numpy())
                class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
            
        
        detections = Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                    )


        
    
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections]

        translated_labels = self.preprocess_translations(detections)
        frame = self.box_annotator.annotate(frame=frame, detections=detections, labels=translated_labels)
        
        return frame
    
    
    
    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        screen_width, screen_height = 1366, 768
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
      
        while True:
          
            start_time = time()
            
            ret, frame = cap.read()
            assert ret



            frame = cv2.resize(frame, (screen_width, screen_height))
            
            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
             
            
            cv2.imshow('YOLOv8 Detection', frame)
 
            if cv2.waitKey(5) & 0xFF == 27:
                
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        
    
detector = ObjectDetection(capture_index=0)
detector()