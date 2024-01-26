# Face similarity
## It detects the faces at camera and calculates the percentual similarity with some known faces.

    This program detects faces in the video, and calculates the similarity with the faces located 
  at folder images.  
    It shows the 3 highest similarity faces in the corners of the screen.  
    Each face detected in the is denominated as person 1-4 and the similarity faces will be 
  showed in a vertical list .


1) Steps:

- Install pip and virtual env
  ->python3 -m pip install --user --upgrade pip
  ->python3 -m pip install --user virtualenv

- create a virtual env and install packages needed to run the program:
  - >python3 -m venv face
  - >source face/bin/activate
  - >pip install -r requirements.txt
              
              
     
2) How to run:

  - >python3 main_task.py
  
