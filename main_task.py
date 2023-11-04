import cv2
import time
from facesimilarity import FaceSimilarity as face_similarity

from webcamstream import WebcamStream as webcam
import faulthandler

# -----------------TASK FACE    ----------------------------------------------------------------------------          
def draw_faces(frame, cam_stream, dim,icon_size):
    list_faces=cam_stream.get_faces()   
        
    person_rect_size=6
    
    if( list_faces != None ):
        count=1
        # Display the results
       
        posx_1 = 1
        posx_2 = posx_1+icon_size
        posy_1=person_rect_size
        posy_2=icon_size+person_rect_size
        
        for (left, top, right, bottom), list_names in list_faces:                              
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4  
            
            font = cv2.FONT_HERSHEY_DUPLEX
                    
            # Draw a box around the face
            color = (0,0,255)
            cv2.rectangle(frame, (left, top), (right, bottom+10), color, 2)              
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom ), (right, bottom+14), color, cv2.FILLED)
            
            # identifying each face as person1, person2, in order to associate to the image icon
            text_person="Person " + str(count)
            cv2.putText(frame, text_person, (left + int((right-left)/2)-28, bottom+10), font, 0.4, (255,255,255), 1)
            
            
            #font = cv2.FONT_HERSHEY_PLAIN        
            posy_1_tmp = posy_1-person_rect_size     
           
            # Draw the 5 most similar images with the percentual
            # Image icon related to person 1 will be drawn on the left side , second person will be drawn below person 1 icons
            # Third person will be drawn on the right side, and the fourth person below third.
            # In case of presence more than 4 person, theirs icons will not be drawn, otherwise the frame could be mixed with too much information.
            for name, perc, path in list_names:
                text = str(perc) + "%" + ":" + name    
                img= cv2.imread(path,cv2.IMREAD_UNCHANGED)
                img_resized = cv2.resize(img,(icon_size,icon_size), interpolation=cv2.INTER_AREA)
                new_icon= cv2.addWeighted(frame[posy_1:posy_2,posx_1:posx_2], 0.4, img_resized[0:icon_size,0:icon_size,:],1-0.4,0)
                cv2.putText(new_icon, text, (1, 68), font, 0.3, (255,255,255), 1)                
               
                frame[ posy_1:posy_2,posx_1:posx_2] = new_icon                
                
                posy_1 =posy_1 + icon_size +1
                posy_2=posy_1 + icon_size
                
                #check if is out of frame
                if ( posy_2 > dim[1]):
                    break             
               
            cv2.rectangle(frame, (posx_1, posy_1_tmp), (posx_2, posy_2-icon_size+1), color, 2) 
            cv2.rectangle(frame, (posx_1, posy_1_tmp), (posx_2, posy_1_tmp+4), color, cv2.FILLED)
            cv2.putText(frame, text_person, (posx_1 + 7, posy_1_tmp+ 5), font, 0.3, (255,255,255), 1)   
            
            posy_1 = posy_1 + person_rect_size +13
            posy_2 = posy_1 + icon_size                         
            count = count+1
            
            # if left side is full, change to rigth.
            if ( posy_2 > dim[1]):
                # check if its already drawn on right side
                if posx_1 == int(dim[0]-icon_size-1):
                    break  
                posx_1 = int(dim[0]-icon_size-1)
                posx_2 = posx_1+icon_size
                posy_1 = person_rect_size
                posy_2 = icon_size+person_rect_size         
            

def main_task():         
    icon_size = 68
    cam= webcam(0)
    cam.start()
    
    # all images in img_path folder with the extension 'img_pattern' will be used as known face to be recognized    
    img_pattern="*.jpeg,*.jpg,*.webp,*.png,*.bmp"
    img_path = "./images"
    
    face_stream = face_similarity(capture=cam, path= img_path, pattern=img_pattern) # 0 id for main camera    
    face_stream.start()   
    
    cv2.namedWindow("window")    
    
    while True :
        if cam.stopped is True :
            break
        else :            
            frame = cam.get()                               
                                        
        # adding a delay for simulating video processing time 
        delay = 0.03 # delay value in seconds
                
        # displaying frame                 
        draw_faces(frame, face_stream, cam.get_cam_dim(),icon_size)            
        #time.sleep(delay) 
        time.sleep(delay) 
            
        # Display the resulting image#
        cv2.imshow('window', frame)

        # Hit 'q'  or esc to quit
        if(  cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27 ) :
            break

    # Release handle to the webcam
    cam.stop()  # stop the webcam stream
    cv2.destroyAllWindows()



if __name__=="__main__": 
    faulthandler.enable()
    main_task()