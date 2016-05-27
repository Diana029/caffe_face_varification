from Tkinter import *
from tkFileDialog import *
from tkMessageBox import *
import caffe
import numpy as np
from PIL import Image, ImageTk
import cv2
import copy
from datetime import datetime
master = Tk()
def load_file1():
  global fname1
  fname1 = askopenfilename( filetypes = [("Image Files", ("*.jpg", "*.gif","*.png","*.jpeg")),
                                         ('All','*')
                                        ] )
  if fname1:
    print(fname1)
    pilImage = Image.open(fname1)
    photo = pilImage.resize((250, 250), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(photo)
    l2.configure(image=image)
    refresh()
           
def load_file2():
  global fname2
  fname2 = askopenfilename( filetypes = [("Image Files", ("*.jpg", "*.gif","*.png","*.jpeg")),
                                         ('All','*')
                                        ] )
  if fname2:
    print(fname2)
    pilImage2 = Image.open(fname2)
    photo2 = pilImage2.resize((250, 250), Image.ANTIALIAS)
    image2 = ImageTk.PhotoImage(photo2)
    l3.configure(image=image2)
    refresh()

def refresh():
  mainloop()

def compare():
  global date_start_1
  date_start_1 = datetime.now()
  l4.configure(text="")
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  
  img = cv2.imread(fname1)
  #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  faces = face_cascade.detectMultiScale(img, 1.3, 5)
  crop_img = img[faces[0,1]:faces[0,1]+faces[0,3], faces[0,0]:faces[0,0]+faces[0,2]] 
  cv2.imwrite(fname1+".jpg", crop_img)

  img2 = cv2.imread(fname2)
  gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
  faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)
  #faces2 = face_cascade.detectMultiScale(img2, 1.3, 5)
  crop_img2 = img2[faces2[0,1]:faces2[0,1]+faces2[0,3], faces2[0,0]:faces2[0,0]+faces2[0,2]] 
  cv2.imwrite(fname2+".jpg", crop_img2)

  caffe.set_mode_cpu()
  #load the model
  net = caffe.Net('/home/diana/Desktop/vgg_face_caffe/LightenedCNN_A_deploy.prototxt',
                '/home/diana/Desktop/vgg_face_caffe/LightenedCNN_A.caffemodel',
                caffe.TEST)

  # load input and configure preprocessing
  transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
  transformer.set_input_scale('data',1)
  transformer.set_transpose('data', (2,0,1))#Defining the order of the channels in input data
  #transformer.set_input_scale('data',1)
  #transformer.set_channel_swap('data', (2,1,0))#model has channels in BGR order instead of RGB
  #transformer.set_raw_scale('data', 255.0)# the reference model operates on images in [0,255] range instead of [0,1]
  transformer.set_input_scale('data',1)
  #note we can change the batch size on-the-fly
  #since we classify only one image, we change batch size from 10 to 1
  net.blobs['data'].reshape(20,1,128,128)

  global date2
  date_start_2 = datetime.now()
  im=caffe.io.load_image(fname1+".jpg", False)
  net.blobs['data'].data[...] = transformer.preprocess('data', im)
  out = net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
  global caffe_ft
  caffe_ft = copy.copy(net.blobs['fc2'].data[0])

  im2=caffe.io.load_image(fname2+".jpg", False)
  net.blobs['data'].data[...] = transformer.preprocess('data', im2)
  out2 = net.forward_all(data=np.asarray([transformer.preprocess('data', im2)]))
  global caffe_ft2
  caffe_ft2 = copy.copy(net.blobs['fc2'].data[0])
  date_end_2 = datetime.now()
  date2 = date_end_2 - date_start_2
  print_comparison()

def print_comparison():
  global caffe_ft
  global caffe_ft2
  date_start_3 = datetime.now()
  dist = np.linalg.norm(caffe_ft-caffe_ft2)
  date_end_3 = datetime.now()
  date3 = date_end_3 - date_start_3
  print(date2)
  print(date3)
  print(dist)
  distance = 0.88096
  if dist<distance:
    l4.config(text="Faces are from the same person")
  else:
    l4.config(text="Faces are from different people")
  date_end_1 = datetime.now()
  date1 = date_end_1 - date_start_1
  print(date1)
master.geometry('600x400+300+300') 
master.title('Face verification system') 
l1 = Label(master, text="Please choose 2 photos")
l1.pack()

b1 = Button(master, text="Choose first photo", command=load_file1)
b1.pack()
b1.place(x = 80, y = 20)

l2 = Label(master)
l2.pack() 
l2.place(x = 30, y = 60); 

b2 = Button(master, text="Choose second photo", command=load_file2)
b2.pack()
b2.place(x = 350, y = 20)

l3 = Label(master)
l3.pack()
l3.place(x = 300, y = 60); 

b3 = Button(master, text="Copmpare", command=compare)
b3.pack()
b3.place(x = 250, y = 330)

l4 = Label(master)
l4.pack()
l4.place(x = 220, y = 370)

mainloop()


