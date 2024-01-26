from threading import Thread
import cv2

# ---------- WEB CAM CLAS -------------------------------------------------------------------
class WebcamStream :
    # initialization method 
    def __init__(self, stream_id=0):
        self.stream_id = stream_id  # default id is 0
        
        # opening video capture stream 
        self.vcap      = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False :    
            print('[Exiting] Error openning the camera')        
            exit(0)
        
        self.ocr_txt=""        
        self.cam_width  = self.vcap.get(3)  # float `width`
        self.cam_height = self.vcap.get(4)  # float `height`       
            
        # Checking frame from vcap stream
        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)

        # Flag for stop thread 
        self.stopped = True

        # thread instantiation  
        self.t_cam = Thread(target=self.update, args=())
        self.t_cam.daemon = True # daemon threads run in background 
        
    # Start thread 
    def start(self):
        self.stopped = False
        self.t_cam.start()

    # Camera frame dimension
    def get_cam_dim(self):
        return (self.cam_width,self.cam_height)    
        
    # Read next available frame  
    def update(self):
        while True :
            if self.stopped is True :
                break

            self.grabbed , self.frame = self.vcap.read()         
            
            if self.grabbed is False :
                print('[Exiting] No more frames to read')
                self.stopped = True
                break 

        #Stopping thread    
        self.vcap.release()        
           
    # method to return latest read frame 
    def get(self):
        return self.frame
    
    # Signaling to stop thread
    def stop(self):
        self.stopped = True
