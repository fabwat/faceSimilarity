import cv2
import glob
import os
import itertools as it, glob
import time
from threading import Thread
from pkg_resources import resource_filename
import dlib
import numpy as np

# ---------- FACE SIMILARITY   -------------------------------------------------------------------

# FaceRecognition static class
class FaceRecognition:        
    @staticmethod
    def trim_bounds(bbox, image_shape):
        return max(bbox[0], 0), max(bbox[1], 0), min(bbox[2], image_shape[1]), min(bbox[3], image_shape[0])

    @staticmethod
    def face_locations(image, upsample=1):
        face_detector = dlib.get_frontal_face_detector()  # use HOG
        # number of times to upsample = 1
        # face_detector returns dlib.fhog_object_detector which returns dlib.rectangles == face

        _ret = []
        for face in face_detector(image, upsample):
            _ret.append(FaceRecognition.trim_bounds((face.left(), face.top(), face.right(), face.bottom()), image.shape))

        return _ret

    @staticmethod
    def load_image(file, pixeltype=cv2.IMREAD_COLOR):
        _image = cv2.imread(file, pixeltype)
        return np.array(_image)

    @staticmethod
    def face_encodings(image, locations=None, upsample=1, jitter=1):
        # Generate the face encodings
        if locations is None:
            face_detector = dlib.get_frontal_face_detector()  # use HOG
            _raw_face_locations = face_detector(image, upsample)  # returns dlib *** RECT *** objects
        else:
            #  left: location[0], top: location[1], right: location[2], bottom: location[3]
            _raw_face_locations = [dlib.rectangle(location[0], location[1], location[2], location[3]) for location in locations]

        # small 5 points landmarks
        predictor_5_model_location = resource_filename(__name__, "models/shape_predictor_5_face_landmarks.dat")
        pose_predictor = dlib.shape_predictor(predictor_5_model_location)
        _raw_landmarks = [pose_predictor(image, face_location) for face_location in _raw_face_locations]

        # face recognition model v1 from dlib
        face_recognition_model_location = resource_filename(__name__, "models/dlib_face_recognition_resnet_model_v1.dat")
        face_encoder = dlib.face_recognition_model_v1(face_recognition_model_location)

        # compute_face_descriptor returns dlib.vectors; convert them to the numpy array
        ret_ar= [np.array(face_encoder.compute_face_descriptor(image, raw_landmark, jitter))
                for raw_landmark in _raw_landmarks]
        return ret_ar

    @staticmethod
    def encoding_distance(known_encodings, encoding_check):
        if len(known_encodings) == 0:
            return np.empty(0)

        return np.linalg.norm(known_encodings - encoding_check, axis=1)

    @staticmethod
    def compare_encodings(known_encodings, encoding_check, tolerance=0.5):       
        return list(FaceRecognition.encoding_distance(known_encodings, encoding_check) <= tolerance)  
    
    
class FaceSimilarity():
    num_ranking = 3 

    def __init__(self,capture, path, pattern ):        
        self.process_this_frame = True 
        self.list_faces_similars = None 
        
        self.t_face = Thread(target=self.update_face, args=())   
        self.t_face.daemon=True
        self.path =path
        self.capture = capture
        self.stopped = False
        
        #read all images in images folder, that will be the known faces, and filename will be used as face name
        path_pattern = [ (path+"/"+s) for s in pattern.split(",")]
        
        image_files= [s for s in self.multiple_file_types(  path_pattern )  ]
        print(image_files)
        self.face_img_path= [s for s in image_files]
        
        self.known_face_encodings = [ FaceRecognition.face_encodings(FaceRecognition.load_image(s))[0] for s in image_files]        
        self.known_face_names = [ (os.path.splitext(os.path.basename(s))[0]).upper()  for s in image_files]        
       
        for i in self.known_face_names:
            print ( i)
           
    def start(self):      
        self.t_face.start()
        print("Starting detecting face ..")
        return self
                   
    def multiple_file_types(self, patterns):
        return it.chain.from_iterable(glob.iglob(pattern) for pattern in patterns) 
    
    def update_face(self):       
       
        delay = 0.05 # delay value in seconds
        
        while True :
            if self.stopped is True :
                break      
  
            result=[]
            if self.process_this_frame:
                frame = self.capture.get()
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                #rgb_small_frame = small_frame[:, :, ::-1]
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Find all the faces and face encodings in the current frame of video
                _face_locations = FaceRecognition.face_locations(rgb_small_frame)
                
                if len(self.known_face_encodings) > 0:
                    
                    _face_encodings = FaceRecognition.face_encodings(rgb_small_frame, locations=_face_locations)

                    for face_encoding in _face_encodings:
                        # use the known face with the smallest distance to the new face
                        #face_distances = FaceRecognition.face_distance(self.known_face_encodings, face_encoding)
                        face_distances = FaceRecognition.encoding_distance(known_encodings=self.known_face_encodings,
                                            encoding_check=face_encoding)
 
                        _name = "Unknown"

                        # If found, use the first one                      
                
                        #face_distance_ranking= enumerate(face_distances )
                        face_distance_ranking_sorted = sorted( enumerate(face_distances ), key=lambda x:x[1]) 
                        
                        # get the 5 faces with the smallest distance, append name and percentual of similarity
                        list_names=[]
                        list_perc=[]
                        list_path=[]
                        for index,distance in face_distance_ranking_sorted[:self.num_ranking]:
                            list_names.append(self.known_face_names[index])
                            list_perc.append( round( ((1- distance)*100) ,2) )
                            list_path.append(self.face_img_path[index])
                            
                        list_top =zip(list_names,list_perc,list_path)                                                     
                        
                        result.append(list_top)
                                            
                    if ( len( result) )> 0:
                        self.list_faces_similars =zip(_face_locations,result)    
                                
                time.sleep(delay) 
                
            self.process_this_frame = not self.process_this_frame     
                       
   
    def get_faces(self):           
        return self.list_faces_similars
    
        
    def stop(self):
        self.stopped = True
   
         