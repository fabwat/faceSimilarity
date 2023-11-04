from threading import Thread
import cv2

# ---------- WEB CAM CLAS -------------------------------------------------------------------
class WebcamStream :
    # initialization method 
    def __init__(self, stream_id=0):
        self.stream_id = stream_id # default is 0 for main camera 
        
        # opening video capture stream 
        self.vcap      = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False :            
            exit(0)
        fps_input_stream = int(self.vcap.get(5)) # hardware fps
        print("FPS of input stream: {}".format(fps_input_stream))
        self.ocr_txt=""
        
        self.cam_width  = self.vcap.get(3)  # float `width`
        self.cam_height = self.vcap.get(4)  # float `height`       
            
        # reading a single frame from vcap stream for initializing 
        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)
        # self.stopped is initialized to False 
        self.stopped = True
        # thread instantiation  
        self.t_cam = Thread(target=self.update, args=())
        self.t_cam.daemon = True # daemon threads run in background 
        
    # method to start thread 
    def start(self):
        self.stopped = False
        self.t_cam.start()
    
    def get_cam_dim(self):
        return (self.cam_width,self.cam_height)    
        
    # method passed to thread to read next available frame  
    def update(self):
        while True :
            if self.stopped is True :
                break
            self.grabbed , self.frame = self.vcap.read()
         
            #print(pytesseract.image_to_string(self.frame))
            if self.grabbed is False :
                print('[Exiting] No more frames to read')
                self.stopped = True
                break 
            
        self.vcap.release()        
           
    # method to return latest read frame 
    def get(self):
        return self.frame
    
    # method to stop reading frames 
    def stop(self):
        self.stopped = True
