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


from matting import alpha_matting, load_image, save_image, stack_images,\
    estimate_foreground_background, METHODS, PRECONDITIONERS
from scipy.constants.constants import alpha

class DataManager:        
    
    OBJECT_DETECTION_MODEL_PATH = join(os.getcwd(), 'models/object_detection/')
    OBJECT_DETECTION_SAVE_PATH = join(os.getcwd(),'result/object_detection/')
    OBJECT_DETECTION_IMAGE_PATH = join(OBJECT_DETECTION_SAVE_PATH, 'images/')
    OBJECT_DETECTION_CROP_PATH = join(OBJECT_DETECTION_SAVE_PATH, 'crop/')
    OBJECT_DETECTION_CROP_IOU75_PATH = join(OBJECT_DETECTION_SAVE_PATH, 'crop_iou_75/')
    OBJECT_DETECTION_CROP_GROUND_TRUTH_PATH = join(OBJECT_DETECTION_SAVE_PATH, 'crop_ground_truth/')
    
    DEEPLAB_MODEL_PATH = join(os.getcwd(), 'models/deeplab/')
    SEGMENTATION_PATH = join(os.getcwd(), 'result/segmentation/')
    SEGMENTATION_IMAGE_PATH = join(SEGMENTATION_PATH, 'images/')
    SEGMENTATION_COLOR_IMAGE_PATH = join(SEGMENTATION_PATH, 'color_images/')
    SEGMENTATION_CROP_PATH = join(SEGMENTATION_PATH, 'crop/')
    SEGMENTATION_TRIMAP_4_6_PATH = join(SEGMENTATION_PATH, 'trimap_4_6/')
    SEGMENTATION_TRIMAP_8_14_PATH = join(SEGMENTATION_PATH, 'trimap_8_14/')    
    SEGMENTATION_CROP_IOU75_PATH = join(SEGMENTATION_PATH, 'crop_iou_75/')
    SEGMENTATION_TRIMAP_IOU75_SIZE_4_6_PATH = join(SEGMENTATION_PATH, 'trimap_iou_75_size_4_6/')
    SEGMENTATION_TRIMAP_IOU75_SIZE_8_14_PATH = join(SEGMENTATION_PATH, 'trimap_iou_75_size_8_14/')    
    
    JSON_SAVE_DATA_PATH = join(os.getcwd(),'result/')
    MATTING_PATH = join(os.getcwd(), 'result/matting/')
    TMP_PATH = join(os.getcwd(), 'result/tmp/')
    LIB_PATH = join(os.getcwd(), 'lib/')
    INPUT_SIZE = 2048
    matting_output_path = ''
    
    COLOR_MAP =[
        [255,0,0],
        [0,255,0],
        [0,0,255]
    ]
    
    threshold = .5
        
    def __init__(self, accuracy = .5, show_mask = False, obj_dection_model_path = '', deeplab_model_path = ''):
        self.accuracy = accuracy
        self.show_mask = show_mask        
        
        if obj_dection_model_path != '':
            self.OBJECT_DETECTION_MODEL_PATH = obj_dection_model_path
        
        if deeplab_model_path != '':
            self.DEEPLAB_MODEL_PATH = deeplab_model_path
        

    def run(self, f, box_magin = .10, calc_bb_iou = True):
        
        self.box_magin = box_magin
        self.calc_bb_iou = calc_bb_iou
        self.detection = ObjectDetection(self.OBJECT_DETECTION_MODEL_PATH)
        self.segmentation = DeepLabModel(self.DEEPLAB_MODEL_PATH)
        millis = int(round(time.time() * 1000))
        self.object_detection_json = 'object_detection_result_' + str(millis) + '.json'
        
        if os.path.isfile(f) and (f.rfind('jpg')>-1 or f.rfind('jpeg')>-1 or f.rfind('png')>-1):
            self.filespath = f[:f.rfind('/')+1]
            self.gtpath = self.filespath + '/gt/' 
            self.__run_for_single_image(f)            
        else:
            self.filespath = f
            self.gtpath = f + '/gt/'
            self.resized = f + '/resized/'
            self.__run_for_multiple_images(f) 
        

    def __run_for_single_image(self, file):
        image = Image.open(file)
        name = file[file.rfind('/')+1:]
        
        resized_im, seg_map = self.segmentation.run(image)        
        boxes, scores = self.__detect_object(image)
        json_file = self.__save_data(image, name, boxes, scores, seg_map)
                
        return json_file


    def __run_for_multiple_images(self, path):
        files = [f for f in listdir(path) if isfile(join(path, f)) and (f.rfind("jpg")>-1 or f.rfind("jpeg")>-1 or f.rfind("png")>-1 or f.rfind("JPG")>-1 or f.rfind("JPEG")>-1 or f.rfind("PNG")>-1)]
        
        """for file in files:
            image = Image.open(join(path, file))
            width, height = image.size
            
            if width > self.INPUT_SIZE or height > self.INPUT_SIZE:        
                resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
                target_size = (int(resize_ratio * width), int(resize_ratio * height))
                resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
                resized_image.save(self.resized + file)
        
        exit()"""
        
        
        """c = 0
        w = 0
        h = 0
        
        for file in files:
            img = cv2.imread(join(path, file))
            height, width, channel = img.shape
            if width > w:
                w = width
                
            if height > h:
                h = height

            if height > 2048 or width > 2048:
                im = self.__resize_image(join(path, file), 2048)
                wi, he = im.size
                c += 1
                print(file)
                print(str(width) + "X" + str(height))
                print(str(wi) + "X" + "" + str(he))
                
        
        print(w, h)
        print(c)
        exit()"""    

        for file in files:
            #if file.find('039004') == -1:
            #    continue

            image = Image.open(join(path, file))
            width, height = image.size
            #image = self.__resize_image(join(path, file), 2048)
            #if width > self.INPUT_SIZE or height > self.INPUT_SIZE:   
            print("Processing file: " + file)
            seg_map = []                  
            resized_im, seg_map, is_resized = self.segmentation.run(image)
            boxes, scores = self.__detect_object(resized_im)            
            json_file = self.__save_data(resized_im, file, boxes, scores, seg_map, width, height, is_resized)
        
        return json_file
    
    def __resize_image(self, img_path, max_px_size):
        with Image.open(img_path) as img:
            width_0, height_0 = img.size
            
            if max((width_0, height_0)) <= max_px_size:
                return img
    
            if width_0 > height_0:
                wpercent = max_px_size / float(width_0)
                hsize = int(float(height_0) * float(wpercent))
                img = img.resize((max_px_size, hsize), Image.ANTIALIAS)
                
    
            if width_0 < height_0:
                hpercent = max_px_size / float(height_0)
                wsize = int(float(width_0) * float(hpercent))
                img = img.resize((wsize, max_px_size), Image.ANTIALIAS)
            
            if width_0 == height_0:
                img = img.resize((max_px_size, max_px_size), Image.ANTIALIAS)
                
            return img
    def run_all_matting(self):
        ignore_files = [
            'Grampus griseus_5537064_image_trimap_3.png',
            'Orcinus orca_17806545_image_trimap_0.png',
            'Pontoporia blainvillei_028210_58402d4255aa5_trimap_0.png',
            'Pontoporia blainvillei_030057_59ac15097727b_trimap_0.png',
            'Pontoporia blainvillei_043922_59fdd56305591_trimap_0.png',
            'Sotalia guianensis_016226_579d042739631_trimap_0.png',
            'Stenella frontalis_458491_image_trimap_0.png',
            'Tursiops truncatus_4218717_image_trimap_0.png',
            'Tursiops truncatus_17370966_image_trimap_1.png',
            'Tursiops truncatus_18754086_image_trimap_0.png'
        ]

        #self.run_matting('result/segmentation/trimap_iou_75_3x3_size_1/', output_path='result/matting_trimap_iou_75_3x3_size_1_th50/', ignore_files=ignore_files)
        #self.run_matting('result/segmentation/trimap_iou_75_3x3_size_1/', threshold=-1, output_path='result/matting_trimap_iou_75_3x3_size_1_th_average/', ignore_files=ignore_files)

        #self.run_matting('result/segmentation/trimap_iou_75_3x3_size_2/', output_path='result/matting_trimap_iou_75_3x3_size_2_th50/', ignore_files=ignore_files)
        #self.run_matting('result/segmentation/trimap_iou_75_3x3_size_2/', threshold=-1, output_path='result/matting_trimap_iou_75_3x3_size_2_th_average/', ignore_files=ignore_files)

        #self.run_matting('result/segmentation/trimap_iou_75_3x3_size_3/', output_path='result/matting_trimap_iou_75_3x3_size_3_th50/', ignore_files=ignore_files)
        #self.run_matting('result/segmentation/trimap_iou_75_3x3_size_3/', threshold=-1, output_path='result/matting_trimap_iou_75_3x3_size_3_th_average/', ignore_files=ignore_files)


        self.run_matting('result/segmentation/trimap_iou_75_5x5_size_1/', output_path='result/matting_trimap_iou_75_5x5_size_1_th50/', ignore_files=ignore_files)
        self.run_matting('result/segmentation/trimap_iou_75_5x5_size_1/', threshold=-1, output_path='result/matting_trimap_iou_75_5x5_size_1_th_average/', ignore_files=ignore_files)

        self.run_matting('result/segmentation/trimap_iou_75_5x5_size_2/', output_path='result/matting_trimap_iou_75_5x5_size_2_th50/', ignore_files=ignore_files)
        self.run_matting('result/segmentation/trimap_iou_75_5x5_size_2/', threshold=-1, output_path='result/matting_trimap_iou_75_5x5_size_2_th_average/', ignore_files=ignore_files)

        self.run_matting('result/segmentation/trimap_iou_75_5x5_size_3/', output_path='result/matting_trimap_iou_75_5x5_size_3_th50/', ignore_files=ignore_files)
        self.run_matting('result/segmentation/trimap_iou_75_5x5_size_3/', threshold=-1, output_path='result/matting_trimap_iou_75_5x5_size_3_th_average/', ignore_files=ignore_files)




    def run_matting(self, path, matting = 'all', threshold=.5, output_path='', ignore_files=[]):
        self.matting_output_path = output_path

        if self.matting_output_path == '':
           self.matting_output_path = self.MATTING_PATH

        folders = ['cf_None',
            'cf_jacobi',
            'cf_vcycle',
            'cf_ichol',
            'knn_None',
            'knn_jacobi',
            'knn_vcycle',
            'knn_ichol',
            'lkm_None',
            'lkm_jacobi',
            'ifm_None',
            'ifm_jacobi',
            'ifm_vcycle',
            'bayesian',
            'closed_form',
            'knn',
            'learding_based']
        
        folders = ['cf_None',
            'knn_None',
            'lkm_None',
            'ifm_None',
            'bayesian',
            'closed_form',
            'knn',
            'learding_based']

        self.threshold = threshold

        if os.path.isdir(self.matting_output_path):
            shutil.rmtree(self.matting_output_path)
        os.mkdir(self.matting_output_path)
        
        for fl in folders:
            if os.path.isdir(self.matting_output_path +fl):
                shutil.rmtree(self.matting_output_path +fl)
            os.mkdir(self.matting_output_path + fl)

        files = [f for f in listdir(path) if isfile(join(path, f)) and (f.rfind("jpg")>-1 or f.rfind("jpeg")>-1 or f.rfind("png")>-1 or f.rfind("JPG")>-1 or f.rfind("JPEG")>-1 or f.rfind("PNG")>-1)]
        
        for file in files:
            if file in ignore_files:
                continue
            name = file[:file.rfind("_trimap_")]
            index = file[file.rfind("_")+1:file.rfind(".")]
            #print(name, index)
            np_image = np.array([])
            type = ".jpg"
            image_partial_name = join(self.OBJECT_DETECTION_CROP_IOU75_PATH, name +"_crop_"+index)
            
            if os.path.isfile(image_partial_name+".JPG"):
                type = ".JPG"
            elif os.path.isfile(image_partial_name+".jpeg"):
                type = ".jpeg"
            elif os.path.isfile(image_partial_name+".JPEG"):
                type = ".JPEG"
            elif os.path.isfile(image_partial_name+".png"):
                type = ".png"
            elif os.path.isfile(image_partial_name+".PNG"):
                type = ".PNG"
                
            image_path = image_partial_name+type
            np_image = cv2.imread(image_path)
            image = Image.open(image_path)
            
            if len(np_image) < 0:
                print('file error ' + name)
            
            trimap = cv2.imread(path + file, 0)
            trimap = trimap.astype(float)
            new_name = name + '_' + index  
            
            
            if matting == 'all' or matting == 'learding_based':
                alpha, binary_image, canny_binary = self.__learding_based(np_image, trimap)
                self.__save_matting_files('learding_based', new_name, trimap, alpha, binary_image, canny_binary, np_image)
            
            if matting == 'all' or matting == 'bayesian':
                alpha, binary_image, canny_binary = self.__bayesian(np_image, trimap)
                self.__save_matting_files('bayesian', new_name, trimap, alpha, binary_image, canny_binary, np_image)
                
            if matting == 'all' or matting == 'knn':
                alpha, binary_image, canny_binary = self.__knn(image, trimap, type)
                self.__save_matting_files('knn', new_name, trimap, alpha, binary_image, canny_binary, np_image)
                
            if matting == 'all' or matting == 'closed_form':
                alpha, binary_image, canny_binary = self.__closed_form(np_image, trimap)
                self.__save_matting_files('closed_form', new_name, trimap, alpha, binary_image, canny_binary, np_image)
                
            if matting == 'all' or matting == 'new_matting':
                self.__new_matting(image_path, path + file, new_name)
            
        """exit()
        
        
        
        with open(file) as json_file:  
            data = json.load(json_file)
        
        for d in data['files']:
            filename = d['filename']
            crop_files = d['cropfiles']
            trimap_files = d['trimapfiles']
            name = filename[:filename.rfind('.')]
            type = filename[filename.rfind('.'):]
            
            if len(crop_files) == len(trimap_files):
                for i, crop in enumerate(crop_files):
                    trimap_file = trimap_files[i]
                    crop_path = join(self.OBJECT_DETECTION_CROP_IOU75_PATH, crop)
                    trimap_path = join(self.SEGMENTATION_TRIMAP_IOU75_PATH, trimap_file)
                    
                    if os.path.isfile(crop_path) == False and os.path.isfile(trimap_path) == False:
                        continue
                                        
                    np_image = cv2.imread(crop_path)
                    trimap = cv2.imread(trimap_path, 0)
                    trimap = trimap.astype(float)
                    new_name = name + '_' + str(i)
                    
                    if matting == 'all' or matting == 'learding_based':
                        alpha, binary_image, canny_binary = self.__learding_based(np_image, trimap)
                        self.__save_matting_files('learding_based', new_name, trimap, alpha, binary_image, canny_binary, np_image)
                    
                    if matting == 'all' or matting == 'bayesian':
                        alpha, binary_image, canny_binary = self.__bayesian(np_image, trimap)
                        self.__save_matting_files('bayesian', new_name, trimap, alpha, binary_image, canny_binary, np_image)
                        
                    if matting == 'all' or matting == 'knn':
                        alpha, binary_image, canny_binary = self.__knn(image, trimap, img_type)
                        self.__save_matting_files('knn', new_name, trimap, alpha, binary_image, canny_binary, np_image)
                        
                    if matting == 'all' or matting == 'closed_form':
                        alpha, binary_image, canny_binary = self.__closed_form(np_image, trimap)
                        self.__save_matting_files('closed_form', new_name, trimap, alpha, binary_image, canny_binary, np_image)
                            
            else:
                print("File error: " + d['filename'])"""
        
    
    def create_trimap_files(self, path, save_path, replace_name='', interations = 1, kernel=3):
        files = [f for f in listdir(path) if isfile(join(path, f)) and (f.rfind("jpg")>-1 or f.rfind("jpeg")>-1 or f.rfind("png")>-1 or f.rfind("JPG")>-1 or f.rfind("JPEG")>-1 or f.rfind("PNG")>-1)]    
        if os.path.isdir(save_path) == False:
            os.mkdir(save_path)        

        for file in files:
            name = file[:file.rfind('.')]
            if replace_name == '':
                name = name + '_trimap.png'
            else:
                name = name.replace(replace_name, 'trimap') + '.png'
                
            seg_map = cv2.imread(path + file, 0)
            trimap = self.__create_trimap(seg_map, interations, kernel)
            cv2.imwrite(save_path + name, trimap)
        
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
    
    def __apply_matting(self, image, seg_map, name, img_type):
        np_image = ut.load_image_into_numpy_array(image)

        seg_map[seg_map>0] = 255
        canny = ut.auto_canny(np.uint8(seg_map))
        dilatation = ut.dilation(canny, 1)

        trimap = seg_map.copy()
        trimap[dilatation == 255] = 128
        trimap = trimap.astype(float)
        
        alpha, binary_image, canny_binary = self.__learding_based(np_image, trimap)
        self.__save_matting_files('learding_based', name, trimap, alpha, binary_image, canny_binary, np_image)

        #alpha, binary_image, canny_binary = self.__bayesian(np_image, trimap)
        #self.__save_matting_files('bayesian', name, trimap, alpha, binary_image, canny_binary, np_image)
                
        alpha, binary_image, canny_binary = self.__knn(image, trimap, img_type)
        self.__save_matting_files('knn', name, trimap, alpha, binary_image, canny_binary, np_image)
        
        alpha, binary_image, canny_binary = self.__closed_form(np_image, trimap)
        self.__save_matting_files('closed_form', name, trimap, alpha, binary_image, canny_binary, np_image)
             
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
        cv2.imwrite(join(self.matting_output_path + path + '/', name+"_trimap.png"), trimap)
        cv2.imwrite(join(self.matting_output_path + path + '/', name+"_alpha.png"), alpha)
        cv2.imwrite(join(self.matting_output_path + path + '/', name+"_binary.png"), binary_image)
        cv2.imwrite(join(self.matting_output_path + path + '/', name+"_canny.png"), canny_binary)
        cv2.imwrite(join(self.matting_output_path + path + '/', name+"_final.png"), np_image)
    
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

        #alpha = lib.matting.knn.run(image, newtrimap)
        command = 'python '+self.LIB_PATH+'knn_matting.py -image='+img_name+' -trimap='+trimap_name+' -output='+alpha_name
        print(command)
        os.system(command)
        alpha = cv2.imread(alpha_name, 0)
        binary_image = alpha.copy()
        binary_image[binary_image < 128] = 0
        binary_image[binary_image >= 128] = 255
        canny_binary = ut.auto_canny(np.uint8(binary_image))

        return alpha, binary_image, canny_binary
        
        #return self.__convert_alpha(alpha)

    def __closed_form(self, image, trimap):
        cf = ClosedForm()
        alpha = cf.run(image, trimap)
        return self.__convert_alpha(alpha)
    
    def __new_matting(self, image_path, trimap_path, name):
        height = 128
        # Load input images
        # shape (height, width, 3) of data type numpy.float64 in range [0, 1]
        image = load_image(image_path, "RGB", "BILINEAR")
        # shape (height, width) of data type numpy.float64 in range [0, 1]
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
                cv2.imwrite(join(self.matting_output_path + p + '/', name+"_trimap.png"), trimap_copy)
                cv2.imwrite(join(self.matting_output_path + p + '/', name+"_alpha.png"), alpha)
                cv2.imwrite(join(self.matting_output_path + p + '/', name+"_binary.png"), binary_image)
                cv2.imwrite(join(self.matting_output_path + p + '/', name+"_canny.png"), canny_binary)
                cv2.imwrite(join(self.matting_output_path + p + '/', name+"_final.png"), np_image)
        
            

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
    
    



    def __save_data(self, image, fileName, boxes, scores, seg_map, or_width, or_height, is_resized):

        file = join(self.JSON_SAVE_DATA_PATH, self.object_detection_json)
        name = fileName[:fileName.rfind('.')]
        type = fileName[fileName.rfind('.'):]
        segname = name+'.png'
        boxes_list = []
        cropfiles = []
        cropsegfiles = []
        trimapfiles = []
        cropgroundtruthfiles = []
        gt_bb_list = []
        iou_list = []
        imagecopy = image.copy()
        width, height = image.size
        draw = ImageDraw.Draw(imagecopy)
        
        ground_truth_image = np.array([])
        
        if os.path.isfile(join(self.gtpath, name+'.png')):
            ground_truth_image = cv2.imread(join(self.gtpath, name+'.png'), 0)            
        
        if len(seg_map) > 0:
            cv2.imwrite(join(self.SEGMENTATION_IMAGE_PATH, segname), seg_map)
            
            cp_segmap = seg_map.copy()            
            cp_segmap = cv2.merge((cp_segmap,cp_segmap,cp_segmap))            
            cp_segmap[np.where((cp_segmap == [1,1,1]).all(axis = 2))] = [0,0,255]
            cp_segmap[np.where((cp_segmap == [2,2,2]).all(axis = 2))] = [0,255,0]
            cp_segmap[np.where((cp_segmap == [3,3,3]).all(axis = 2))] = [255,0,0]
            cv2.imwrite(join(self.SEGMENTATION_COLOR_IMAGE_PATH, segname), cp_segmap)
            

        print("Save data from file: " + fileName)
        if self.calc_bb_iou and os.path.isfile(join(self.filespath, name+'.xml')):
            tree = ET.parse(join(self.filespath, name+'.xml'))
            root = tree.getroot()
            add = False
            for member in root.findall('object'):
                #if member.find('name').text == 'dorsal' or member.find('name').text == 'dorsal_overlap':
                if member.find('name').text == 'dorsal':
                    xmin = int(member.find('bndbox').find('xmin').text)
                    ymin = int(member.find('bndbox').find('ymin').text)
                    xmax = int(member.find('bndbox').find('xmax').text)
                    ymax = int(member.find('bndbox').find('ymax').text)
                    
                    if is_resized:
                        xmin = int(round(width * (xmin / or_width)))
                        ymin = int(round(height * (ymin / or_height)))
                        xmax = int(round(width * (xmax / or_width)))
                        ymax = int(round(height * (ymax / or_height)))
                    
                    gt_bb_list.append([xmin, ymin, xmax, ymax])

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
            is_iou = False
            
            if len(gt_bb_list) > 0:
                for boxA in gt_bb_list:
                    iou, overlap_percent = self.__bb_intersection_over_union(boxA, bb)
                    
                    if iou >= .75:
                        is_iou = True
                            
                    if iou > .0:
                        iou_list.append({
                            'ground_truth_bb': np.array(boxA).tolist(),
                            'object_detection_bb': np.array(bb).tolist(),                        
                            'od_bb_index': i,
                            'iou': iou,
                            'overlap_percent':overlap_percent
                        })
                        draw.rectangle(((boxA[0], boxA[1]), (boxA[2], boxA[3])), outline = '#000000')
            
            
            if len(seg_map) > 0:
                crop_seg = seg_map[ymin:ymin+(ymax - ymin), xmin:xmin+(xmax - xmin)]
                crop_seg_name = name+'_cropseg_'+str(i)+'.png'
                cv2.imwrite(join(self.SEGMENTATION_CROP_PATH, crop_seg_name), crop_seg)
                cropsegfiles.append(crop_seg_name)
                
                trimap_name = name+'_trimap_'+str(i)+".png"
                trimapfiles.append(trimap_name)
                
                trimap_4_6 = self.__create_trimap(crop_seg.copy())
                cv2.imwrite(join(self.SEGMENTATION_TRIMAP_4_6_PATH, trimap_name), trimap_4_6)
                
                trimap_8_14 = self.__create_trimap(crop_seg.copy(), size=2)
                cv2.imwrite(join(self.SEGMENTATION_TRIMAP_8_14_PATH, trimap_name), trimap_8_14)
                
                if is_iou:
                    cv2.imwrite(join(self.SEGMENTATION_CROP_IOU75_PATH, crop_seg_name), crop_seg)
                    cv2.imwrite(join(self.SEGMENTATION_TRIMAP_IOU75_SIZE_4_6_PATH, trimap_name), trimap_4_6)
                    cv2.imwrite(join(self.SEGMENTATION_TRIMAP_IOU75_SIZE_8_14_PATH, trimap_name), trimap_8_14)
                    
            
            crop = image.crop((xmin, ymin, xmax, ymax))
            crop_name = name + '_crop_' + str(i) + type
            crop.save(join(self.OBJECT_DETECTION_CROP_PATH, crop_name))
            cropfiles.append(crop_name)
            
            if len(ground_truth_image) > 0:
                crop_ground_truth = ground_truth_image[ymin:ymin+(ymax - ymin), xmin:xmin+(xmax - xmin)]                
                crop_ground_truth_name = name + '_crop_ground_truth_' + str(i) + ".png"
                cv2.imwrite(join(self.OBJECT_DETECTION_CROP_GROUND_TRUTH_PATH, crop_ground_truth_name), crop_ground_truth)
                cropgroundtruthfiles.append(crop_ground_truth_name)                
            
            if is_iou:
                crop.save(join(self.OBJECT_DETECTION_CROP_IOU75_PATH, crop_name))

            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline = '#ffff00')
            draw.rectangle(((xmin, ymin - p), (xmin + 45, ymin - p + 15)), fill = '#ffff00')
            draw.text((xmin+3, ymin+3 - p), str(score) + '%', fill='black')
            
            
            #self.__apply_matting(crop, crop_seg, name + '_' + str(i), type)
            
        imagecopy.save(join(self.OBJECT_DETECTION_IMAGE_PATH, fileName))

        if os.path.isfile(file):
            with open(file) as json_file:
                data = json.load(json_file)
        else:
            data = {}
            data['files'] = []
        
        
        data['files'].append({
            'filename': fileName,
            'segmapfilename': segname,
            'boxes': np.array(boxes_list).tolist(),
            'scores': np.array(scores).tolist(),
            'cropfiles': cropfiles,
            'cropsegfiles': cropsegfiles,
            'trimapfiles': trimapfiles,
            'cropgroundtruthfiles': cropgroundtruthfiles,
            'iouresults': iou_list,
            'is_resized': is_resized,
            'oldwidth': width,
            'oldheight': height
        })

        with open(file,"w+") as json_file:
            json.dump(data, json_file)
            json_file.close()
        
        return file

    def __bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
     
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
     
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
     
        # return the intersection over union value
        return iou , interArea/boxAArea




