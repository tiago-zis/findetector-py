import numpy as np
import time
import json
import os
import uuid
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageDraw
from lib.objectdetection import ObjectDetection

class DataManagerTest:        
    
    RESULT_PATH = join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'process_result/')
    IMAGES_PATH = join(RESULT_PATH, 'images/')
    
    OBJECT_DETECTION_PATH = 'models/research/object_detection'
    OBJECT_DETECTION_MODEL_PATH = join(os.path.dirname(os.path.dirname(__file__)), 'models/object_detection/')

    OBJECT_DETECTION_SAVE_PATH = join(RESULT_PATH,'object_detection/')
    OBJECT_DETECTION_IMAGE_PATH = join(OBJECT_DETECTION_SAVE_PATH, 'images/')
    OBJECT_DETECTION_CROP_PATH = join(OBJECT_DETECTION_SAVE_PATH, 'crop/')
    
    JSON_SAVE_DATA_PATH = join(RESULT_PATH, 'json/')
    LIB_PATH = join(os.getcwd(), 'lib/')
    INPUT_SIZE = 2048
    
    def __init__(self, accuracy = .5, show_mask = False, threshold=-1, 
                 obj_dection_model_path = '', tf_path = ''):
        
        self.tf_path = tf_path
        self.accuracy = accuracy
        self.show_mask = show_mask
        self.threshold = threshold
        
        self.check_paths(obj_dection_model_path)
            
        
    def check_paths(self, obj_dection_model_path):
        self.OBJECT_DETECTION_PATH = join(self.tf_path, self.OBJECT_DETECTION_PATH)

        if os.path.isdir(self.OBJECT_DETECTION_PATH) == False:
            print('TensorFlow object detection path not exist!!!')
            exit()
        
        if obj_dection_model_path != '':
            self.OBJECT_DETECTION_MODEL_PATH = obj_dection_model_path
        
        os.makedirs(self.RESULT_PATH, exist_ok=True)        
        os.makedirs(self.IMAGES_PATH, exist_ok=True)
        os.makedirs(self.OBJECT_DETECTION_SAVE_PATH, exist_ok=True)
        os.makedirs(self.OBJECT_DETECTION_IMAGE_PATH, exist_ok=True)
        os.makedirs(self.OBJECT_DETECTION_CROP_PATH, exist_ok=True)
        os.makedirs(self.JSON_SAVE_DATA_PATH, exist_ok=True)
        

    def runInternal(self, f, box_magin = .10):
        self.box_magin = box_magin
        self.detection = ObjectDetection(self.OBJECT_DETECTION_PATH, self.OBJECT_DETECTION_MODEL_PATH)

        if os.path.isfile(f) and (f.rfind('jpg')>-1 or f.rfind('jpeg')>-1 or f.rfind('png')>-1 or f.rfind('JPG')>-1 or f.rfind('JPEG')>-1 or f.rfind('PNG')>-1):
            self.filespath = f[:f.rfind('/')+1]
            return self.__run_for_single_image(f, True)
        
        return

    def run(self, f, box_magin = .10):
        self.box_magin = box_magin
        self.detection = ObjectDetection(self.OBJECT_DETECTION_PATH, self.OBJECT_DETECTION_MODEL_PATH)
        self.current_files_tag = str(int(round(time.time() * 1000)))        
        self.object_detection_json = 'object_detection_result_' + self.current_files_tag + '.json'
        
        if os.path.isfile(f) and (f.rfind('jpg')>-1 or f.rfind('jpeg')>-1 or f.rfind('png')>-1 or f.rfind('JPG')>-1 or f.rfind('JPEG')>-1 or f.rfind('PNG')>-1):
            self.filespath = f[:f.rfind('/')+1]
            self.__run_for_single_image(f)            
        else:
            self.filespath = f
            self.__run_for_multiple_images(f) 
        

    def __run_for_single_image(self, file, isDb = False):
        image = Image.open(file)
        width, height = image.size
        name = file[file.rfind('/')+1:]        
        
        #zis: removido temporáriamente, pois estava interferindo no resultado das detecções
        #image, is_resized = self.check_image_size(image)
        
        image = image.convert('RGB')
        boxes, scores = self.__detect_object(image)

        if (isDb):
            return self.__create_data_from_db(boxes, scores, width, height)
        else:
            return self.__save_data(image, name, boxes, scores, width, height)


    def __run_for_multiple_images(self, path):
        files = [f for f in listdir(path) if isfile(join(path, f)) and (f.rfind("jpg")>-1 or f.rfind("jpeg")>-1 or f.rfind("png")>-1 or f.rfind("JPG")>-1 or f.rfind("JPEG")>-1 or f.rfind("PNG")>-1)]
        
        for file in files:
            print("Processing file: " + file)
            
            image = Image.open(join(path, file))
            name = file[file.rfind('/')+1:]        
            
            #zis: removido temporáriamente, pois estava interferindo no resultado das detecções
            #image, is_resized = self.check_image_size(image)
            
            image = image.convert('RGB')

            boxes, scores = self.__detect_object(image)
            json_file = self.__save_data(image, name, boxes, scores)
        
        return json_file
    
    
    def __detect_object(self, image):
        boxes, scores, classes, num_detections = self.detection.detect(image)
        width, height = image.size

        boxesList = []
        scoresList = []

        for i in range(num_detections):
            if scores[i] >= self.accuracy and classes[i] == 1:            
                box = self.__convert_box(boxes[i], width, height)
                boxesList.append(box)
                scoresList.append(scores[i])

        return boxesList, scoresList

    def __convert_box(self, box, width, height):
        left = box[1] * width
        right = box[3] * width
        top = box[0] * height
        botton = box[2] * height
        
        bw = right - left
        bh = botton - top

        box[1] = round(left - self.box_magin * bw) #left
        box[3] = round(right + self.box_magin * bw) #right
        box[0] = round(top - self.box_magin * bh) #top
        box[2] = round(botton + self.box_magin * bh) #botton
        box = box.astype(int)
        
        if box[1] < 0:
            box[1] = 1
        
        if box[3] > width:
            box[3] = width - 1
        
        if box[0] < 0:
            box[0] = 1
        
        if box[2] > height:
            box[2] = height - 1
        
        return box
    
    def __save_data(self, image, fileName, boxes, scores, old_width, old_height):

        file = join(self.JSON_SAVE_DATA_PATH, self.object_detection_json)
        name = fileName[:fileName.rfind('.')]
        type = fileName[fileName.rfind('.'):]
        draw_detection_filename = name + '_' + self.current_files_tag + type
        
        boxes_list = []
        cropfiles = []
        
        imagecopy = image.copy()
        width, height = image.size
        draw = ImageDraw.Draw(imagecopy)
        image.save(self.IMAGES_PATH+draw_detection_filename)
        
        print("Save data from file: " + fileName)
        
        for i, box in enumerate(boxes):
            p = 0
            if box[0] > 15:
                p = 15

            xmin = box[1]
            ymin = box[0]
            xmax = box[3]
            ymax = box[2]
            bb = [xmin, ymin, xmax, ymax]
            boxes_list.append(bb)
            score = format(scores[i] * 100, '.1f')
            
            crop = image.crop((xmin, ymin, xmax, ymax))
            crop_name = name + '_crop_' + str(i) + '_' + self.current_files_tag + type
            crop.save(join(self.OBJECT_DETECTION_CROP_PATH, crop_name))
            cropfiles.append(crop_name)
            
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline = '#ffff00')
            draw.rectangle(((xmin, ymin - p), (xmin + 45, ymin - p + 15)), fill = '#ffff00')
            draw.text((xmin+3, ymin+3 - p), str(score) + '%', fill='black')
            
            
        imagecopy.save(join(self.OBJECT_DETECTION_IMAGE_PATH, draw_detection_filename))

        if os.path.isfile(file):
            with open(file) as json_file:
                data = json.load(json_file)
        else:
            data = {}
            data['files'] = []
        
        data['files'].append({
            'filename': draw_detection_filename,
            'boxes': np.array(boxes_list).tolist(),
            'scores': np.array(scores).tolist(),
            'cropfiles': cropfiles,
            'oldwidth': old_width,
            'oldheight': old_height,
            'width': width,
            'height': height            
        })

        with open(file,"w+") as json_file:
            json.dump(data, json_file, indent=4)
            json_file.close()
        
        return file


    def __create_data_from_db(self, boxes, scores, width, height):

        boxes_list = []       
        
        for i, box in enumerate(boxes):
            p = 0
            if box[0] > 15:
                p = 15

            xmin = box[1]
            ymin = box[0]
            xmax = box[3]
            ymax = box[2]
            uid = uuid.uuid4()
            
            bb = {
                "uid": str(uid),
                "p1": {
                    "x": xmin,
                    "y": ymin
                },
                "p2": {
                    "x": xmax,
                    "y": ymax
                },
                "valid": 0          
            }

            #bb = [xmin, ymin, xmax, ymax]
            boxes_list.append(bb)
            
        
        return {
            'boxes': np.array(boxes_list).tolist(),
            'scores': np.array(scores).tolist(),
            "width": width,
            "height": height  
        }

        
    
    def check_image_size(self, image):
        width, height = image.size
        resized = False
        
        if width > self.INPUT_SIZE or height > self.INPUT_SIZE:        
            resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
            target_size = (int(resize_ratio * width), int(resize_ratio * height))
            image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
            resized = True
        else:
            image = image.convert('RGB')
            
        return image, resized
    