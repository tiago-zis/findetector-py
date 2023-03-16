import numpy as np
import cv2
import time
import json
import os
import lib.matting.knn
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from lib.objectdetection import ObjectDetection
from lib.deeplabmodel import DeepLabModel
from lib.libutils import LibUtils as ut
from lib.matting.learning_based import LearningBased
from lib.matting.bayesian import Bayesian
from lib.matting.closed_form import ClosedForm
import xml.etree.ElementTree as ET
from numpy.core.defchararray import replace
import shutil


#from matting import alpha_matting, load_image, save_image, stack_images,\
#    estimate_foreground_background, METHODS, PRECONDITIONERS

from scipy.constants.constants import alpha

class DataManager2:        
    
    RESULT_PATH = join(os.getcwd(),'process_result/')
    IMAGES_PATH = join(RESULT_PATH, 'images/')
    
    OBJECT_DETECTION_PATH = 'models/research/object_detection'
    OBJECT_DETECTION_MODEL_PATH = join(os.getcwd(), 'models/object_detection/')
            
    OBJECT_DETECTION_SAVE_PATH = join(RESULT_PATH,'object_detection/')
    OBJECT_DETECTION_IMAGE_PATH = join(OBJECT_DETECTION_SAVE_PATH, 'images/')
    OBJECT_DETECTION_CROP_PATH = join(OBJECT_DETECTION_SAVE_PATH, 'crop/')
    
    DEEPLAB_MODEL_PATH = join(os.getcwd(), 'models/deeplab/')
    
    SEGMENTATION_PATH = join(RESULT_PATH, 'segmentation/')
    SEGMENTATION_IMAGE_PATH = join(SEGMENTATION_PATH, 'images/')
    SEGMENTATION_COLOR_IMAGE_PATH = join(SEGMENTATION_PATH, 'color_images/')
    SEGMENTATION_CROP_PATH = join(SEGMENTATION_PATH, 'crop/')
    SEGMENTATION_TRIMAP_PATH = join(SEGMENTATION_PATH, 'trimap/')
    
    JSON_SAVE_DATA_PATH = join(RESULT_PATH, 'json/')
    MATTING_PATH = join(RESULT_PATH, 'matting/')
    TMP_PATH = join(RESULT_PATH, 'tmp/')
    LIB_PATH = join(os.getcwd(), 'lib/')
    INPUT_SIZE = 2048
    
    COLOR_MAP =[
        [255,0,0],
        [0,255,0],
        [0,0,255]
    ]
    
    def __init__(self, accuracy = .5, show_mask = False, threshold=-1, 
                 obj_dection_model_path = '', deeplab_model_path = '',
                 tf_path = ''):
        
        self.tf_path = tf_path
        self.accuracy = accuracy
        self.show_mask = show_mask
        self.threshold = threshold
        
        self.check_paths(obj_dection_model_path, deeplab_model_path)
            
        
    def check_paths(self, obj_dection_model_path, deeplab_model_path):
        self.OBJECT_DETECTION_PATH = join(self.tf_path, self.OBJECT_DETECTION_PATH)
        if os.path.isdir(self.OBJECT_DETECTION_PATH) == False:
            print('TensorFlow object detection path not exist!!!')
            exit()
        
        if obj_dection_model_path != '':
            self.OBJECT_DETECTION_MODEL_PATH = obj_dection_model_path
        
        if deeplab_model_path != '':
            self.DEEPLAB_MODEL_PATH = deeplab_model_path
                
        os.makedirs(self.RESULT_PATH, exist_ok=True)        
        os.makedirs(self.IMAGES_PATH, exist_ok=True)
        os.makedirs(self.OBJECT_DETECTION_SAVE_PATH, exist_ok=True)
        os.makedirs(self.OBJECT_DETECTION_IMAGE_PATH, exist_ok=True)
        os.makedirs(self.OBJECT_DETECTION_CROP_PATH, exist_ok=True)
        os.makedirs(self.JSON_SAVE_DATA_PATH, exist_ok=True)
        os.makedirs(self.SEGMENTATION_PATH, exist_ok=True)
        os.makedirs(self.SEGMENTATION_IMAGE_PATH, exist_ok=True)
        os.makedirs(self.SEGMENTATION_COLOR_IMAGE_PATH, exist_ok=True)
        os.makedirs(self.SEGMENTATION_CROP_PATH, exist_ok=True)
        os.makedirs(self.SEGMENTATION_TRIMAP_PATH, exist_ok=True)
        os.makedirs(self.MATTING_PATH, exist_ok=True)
        os.makedirs(self.TMP_PATH, exist_ok=True)
        
        matting_folders = ['cf_None',
            'knn_None',
            'lkm_None',
            'ifm_None',
            'bayesian',
            'closed_form',
            'knn',
            'learding_based']
        
        for fl in matting_folders:
            os.makedirs(join(self.MATTING_PATH, fl), exist_ok=True)
        

    def run(self, f, box_magin = .10):
        self.box_magin = box_magin
        self.detection = ObjectDetection(self.OBJECT_DETECTION_PATH, self.OBJECT_DETECTION_MODEL_PATH)
        self.segmentation = DeepLabModel(self.DEEPLAB_MODEL_PATH)
        self.current_files_tag = str(int(round(time.time() * 1000)))        
        self.object_detection_json = 'object_detection_result_' + self.current_files_tag + '.json'
        
        if os.path.isfile(f) and (f.rfind('jpg')>-1 or f.rfind('jpeg')>-1 or f.rfind('png')>-1 or f.rfind('JPG')>-1 or f.rfind('JPEG')>-1 or f.rfind('PNG')>-1):
            self.filespath = f[:f.rfind('/')+1]
            self.__run_for_single_image(f)            
        else:
            self.filespath = f
            self.__run_for_multiple_images(f) 
        

    def __run_for_single_image(self, file):   
        image = Image.open(file)
        width, height = image.size
        name = file[file.rfind('/')+1:]        
        image, is_resized = self.check_image_size(image)        
        
        resized_im, seg_map = self.segmentation.run(image)      
        boxes, scores = self.__detect_object(image)
        json_file = self.__save_data(image, name, boxes, scores, seg_map, is_resized, width, height)
                
        return json_file


    def __run_for_multiple_images(self, path):
        files = [f for f in listdir(path) if isfile(join(path, f)) and (f.rfind("jpg")>-1 or f.rfind("jpeg")>-1 or f.rfind("png")>-1 or f.rfind("JPG")>-1 or f.rfind("JPEG")>-1 or f.rfind("PNG")>-1)]
        
        for file in files:
            print("Processing file: " + file)
            
            image = Image.open(join(path, file))
            name = file[file.rfind('/')+1:]        
            image, is_resized = self.check_image_size(image)
            
            resized_im, seg_map = self.segmentation.run(image)      
            boxes, scores = self.__detect_object(image)
            json_file = self.__save_data(image, name, boxes, scores, seg_map, is_resized)
        
        return json_file
    
    
    def run_matting(self, cropfiles, matting = 'all'):
        list = []
        
        for file in cropfiles:
            name = file[:file.rfind('.')]
            type = file[file.rfind('.'):]
            
            trimap_img_name = name
            trimap_img_name = trimap_img_name.replace('_crop_', '_trimap_')
            trimap_img_name = join(self.SEGMENTATION_TRIMAP_PATH, trimap_img_name + '.png')
            
            image_name = join(self.OBJECT_DETECTION_CROP_PATH, file)
            
            np_image = cv2.imread(image_name)
            image = Image.open(image_name)
            
            if len(np_image) < 0:
                print('file error ' + name)
            
            trimap = cv2.imread(trimap_img_name, 0)
            trimap = trimap.astype(float)
            new_name = name
            
            data = {}
            
            if matting == 'all' or matting == 'learding_based':
                alpha, binary_image, canny_binary = self.__learding_based(np_image, trimap)
                result = self.__save_matting_files('learding_based', new_name, trimap, alpha, binary_image, canny_binary, np_image)
                data['learding_based'] = result
            
            if matting == 'all' or matting == 'bayesian':
                alpha, binary_image, canny_binary = self.__bayesian(np_image, trimap)
                result = self.__save_matting_files('bayesian', new_name, trimap, alpha, binary_image, canny_binary, np_image)
                data['bayesian'] = result
                
            if matting == 'all' or matting == 'knn':
                alpha, binary_image, canny_binary = self.__knn(image, trimap, type)
                result = self.__save_matting_files('knn', new_name, trimap, alpha, binary_image, canny_binary, np_image)
                data['knn'] = result
                
            if matting == 'all' or matting == 'closed_form':
                alpha, binary_image, canny_binary = self.__closed_form(np_image, trimap)
                result = self.__save_matting_files('closed_form', new_name, trimap, alpha, binary_image, canny_binary, np_image)
                data['closed_form'] = result
                
            #if matting == 'all' or matting == 'new_matting':
            #    data = self.__new_matting(image_name, trimap_img_name, new_name, data)
            
            list.append(data)
            
        return list
    
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
    
    def __create_trimap(self, seg_map, size=1, kernel=3):
        seg_map[seg_map>0] = 255
        canny = ut.auto_canny(np.uint8(seg_map))
        dilatation = ut.dilation(canny, size, kernel)

        trimap = seg_map.copy()
        trimap[dilatation == 255] = 128
        trimap = trimap.astype(float)
        return trimap
    
    def __save_matting_files(self, path, name, trimap, alpha, binary_image, canny_binary, image):
        np_image = image.copy()
        np_image[canny_binary == 255] = [0,255,255]
        
        trimap_name = name.replace('crop', 'trimap') + '.png'
        alpha_name = name.replace('crop', 'alpha') + '.png'
        binary_name = name.replace('crop', 'binary') + '.png'
        canny_name = name.replace('crop', 'canny') + '.png'
        final_name = name.replace('crop', 'final') + '.png'
        
        data = {
            'trimap': trimap_name,
            'alpha': alpha_name,
            'binary': binary_name,
            'canny': canny_name,
            'final': final_name
        }
        
        cv2.imwrite(join(self.MATTING_PATH + path + '/', trimap_name), trimap)
        cv2.imwrite(join(self.MATTING_PATH + path + '/', alpha_name), alpha)
        cv2.imwrite(join(self.MATTING_PATH + path + '/', binary_name), binary_image)
        cv2.imwrite(join(self.MATTING_PATH + path + '/', canny_name), canny_binary)
        cv2.imwrite(join(self.MATTING_PATH + path + '/', final_name), np_image)
        return data
    
    def __learding_based(self, image, trimap):
        lb = LearningBased()
        alpha = lb.run(image, trimap)
        return self.__convert_alpha(alpha)

    def __bayesian(self, image, trimap):
        b = Bayesian()
        alpha = b.run(image, trimap)
        return self.__convert_alpha(alpha)

    def __knn(self, image, trimap, img_type):
        img_name = join(self.TMP_PATH, "image" + img_type)
        trimap_name = join(self.TMP_PATH, "trimap.png")
        alpha_name = join(self.TMP_PATH, "alpha.png")

        newtrimap = cv2.merge((trimap,trimap,trimap))
        cv2.imwrite(trimap_name, newtrimap)
        image.save(img_name)

        command = 'python '+self.LIB_PATH+'knn_matting.py -image='+img_name+' -trimap='+trimap_name+' -output='+alpha_name
        print(command)
        os.system(command)
        alpha = cv2.imread(alpha_name, 0)
        binary_image = alpha.copy()
        binary_image[binary_image < 128] = 0
        binary_image[binary_image >= 128] = 255
        canny_binary = ut.auto_canny(np.uint8(binary_image))

        return alpha, binary_image, canny_binary
        
    def __closed_form(self, image, trimap):
        cf = ClosedForm()
        alpha = cf.run(image, trimap)
        return self.__convert_alpha(alpha)
    
    def __new_matting(self, image_path, trimap_path, name, data):
        
        image = load_image(image_path, "RGB", "BILINEAR")
        trimap = load_image(trimap_path, "GRAY", "NEAREST")
        trimap_copy = trimap.copy()
        trimap_copy = trimap_copy * 255

        for method in METHODS:            
            for preconditioner in PRECONDITIONERS[method]:
                if preconditioner != None:
                    continue
                alpha = alpha_matting(
                    image, trimap,
                    method, preconditioner,
                    print_info=True)
                
                p = method+"_"+str(preconditioner)
                alpha, binary_image, canny_binary = self.__convert_alpha(alpha)
                np_image = cv2.imread(image_path)
                np_image[canny_binary == 255] = [0,255,255]
                
                trimap_name = name.replace('crop', 'trimap') + '.png'
                alpha_name = name.replace('crop', 'alpha') + '.png'
                binary_name = name.replace('crop', 'binary') + '.png'
                canny_name = name.replace('crop', 'canny') + '.png'
                final_name = name.replace('crop', 'final') + '.png'
                
                data[p] = {
                    'trimap': trimap_name,
                    'alpha': alpha_name,
                    'binary': binary_name,
                    'canny': canny_name,
                    'final': final_name
                }
                                
                cv2.imwrite(join(self.MATTING_PATH + p + '/', trimap_name), trimap_copy)
                cv2.imwrite(join(self.MATTING_PATH + p + '/', alpha_name), alpha)
                cv2.imwrite(join(self.MATTING_PATH + p + '/', binary_name), binary_image)
                cv2.imwrite(join(self.MATTING_PATH + p + '/', canny_name), canny_binary)
                cv2.imwrite(join(self.MATTING_PATH + p + '/', final_name), np_image)
        
        return data
        
            

    def __convert_alpha(self, alpha):
        binary_image = self.__binarize_alpha(alpha)
        alpha = alpha*255
        canny_binary = ut.auto_canny(np.uint8(binary_image))
        return alpha, binary_image, canny_binary

    def __binarize_alpha(self, alpha):
        binary_image = alpha.copy()
        binary_image[np.isnan(binary_image)] = .0
        th = self.threshold
        
        if self.threshold == -1:
            th = np.average(binary_image)
        
        binary_image[binary_image < th] = 0
        binary_image[binary_image >= th] = 255
        return binary_image
    

    def __save_data(self, image, fileName, boxes, scores, seg_map, is_resized, old_width, old_height):

        file = join(self.JSON_SAVE_DATA_PATH, self.object_detection_json)
        name = fileName[:fileName.rfind('.')]
        type = fileName[fileName.rfind('.'):]
        draw_detection_filename = name + '_' + self.current_files_tag + type
        segname = name + '_' + self.current_files_tag + '.png'
        boxes_list = []
        cropfiles = []
        cropsegfiles = []
        trimapfiles = []
        
        imagecopy = image.copy()
        width, height = image.size
        draw = ImageDraw.Draw(imagecopy)

        image.save(self.IMAGES_PATH+draw_detection_filename)
        
        if len(seg_map) > 0:
            cp_segmap = seg_map.copy()            
            cp_segmap = cv2.merge((cp_segmap,cp_segmap,cp_segmap))            
            cp_segmap[np.where((cp_segmap == [1,1,1]).all(axis = 2))] = [0,0,255]
            cp_segmap[np.where((cp_segmap == [2,2,2]).all(axis = 2))] = [0,255,0]
            cp_segmap[np.where((cp_segmap == [3,3,3]).all(axis = 2))] = [255,0,0]
            cv2.imwrite(join(self.SEGMENTATION_COLOR_IMAGE_PATH, segname), cp_segmap)
            

        print("Save data from file: " + fileName)
        #box[0] top=ymin, box[1] left=xmin, box[2] botton=ymax, box[3] right=xmax
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
            
            if len(seg_map) > 0:
                crop_seg = seg_map[ymin:ymin+(ymax - ymin), xmin:xmin+(xmax - xmin)]
                crop_seg_name = name+'_cropseg_'+str(i)+'_'+self.current_files_tag+'.png'
                cv2.imwrite(join(self.SEGMENTATION_CROP_PATH, crop_seg_name), crop_seg)
                cropsegfiles.append(crop_seg_name)
                
                trimap_name = name+'_trimap_'+str(i)+'_'+self.current_files_tag+".png"
                trimapfiles.append(trimap_name)
                
                trimap = self.__create_trimap(crop_seg.copy(), size=3)
                cv2.imwrite(join(self.SEGMENTATION_TRIMAP_PATH, trimap_name), trimap)
                
            
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
        
        matting_files = self.run_matting(cropfiles)
        
        data['files'].append({
            'filename': draw_detection_filename,
            'segfilename': segname,
            'boxes': np.array(boxes_list).tolist(),
            'scores': np.array(scores).tolist(),
            'cropfiles': cropfiles,
            'cropsegfiles': cropsegfiles,
            'trimapfiles': trimapfiles,
            'matting_files': matting_files,
            'oldwidth': old_width,
            'oldheight': old_height,
            'width': width,
            'height': height,
            'segmentation_resized': is_resized
        })

        with open(file,"w+") as json_file:
            json.dump(data, json_file, indent=4)
            json_file.close()
        
        return file
    
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
    
    