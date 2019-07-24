import sys, os
import cv2
import numpy as np
import traceback
import time
import keras
import darknet.python.darknet as dn

from darknet.python.darknet import detect
from darknet.python.darknet import nparray_to_image
from src.label 				import Label, lwrite ,Shape, writeShapes, dknet_label_conversion, lread,readShapes
from src.keras_utils 		import load_model, detect_lp
from src.utils 				import crop_region, image_files_from_folder,im2single, nms
from src.drawing_utils		import draw_label, draw_losangle, write2img
from os.path 				import splitext, basename, isdir, isfile
from os 					import makedirs
from glob 					import glob
from pdb                    import set_trace as pause
from keras.preprocessing.image import img_to_array, array_to_img

def get_time_stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    return time_stamp
    #print(time_stamp)
    #stamp = ("".join(time_stamp.split()[0].split("-"))+"".join(time_stamp.split()[1].split(":"))).replace('.', '')
    #print(stamp)


def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))



if __name__ == '__main__':


    input_device  = 0
    output_dir = "output"
    OCR_SUCCEED = False

    count = 0
    fps = 5
    pre_frame = None
    cur_lp_str = ""
    last_lp_str = ""

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)

    #lp detection model load 
    lp_threshold = .5
    wpod_net_path = 'data/lp-detector/wpod-net_update1.h5'
    wpod_net = load_model(wpod_net_path)
    
    #lp ocr model load 
    ocr_threshold = .4
    ocr_weights = 'data/ocr/ocr-net.weights'
    ocr_netcfg  = 'data/ocr/ocr-net.cfg'
    ocr_dataset = 'data/ocr/ocr-net.data'
    ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
    ocr_meta = dn.load_meta(ocr_dataset)


    if not isdir(output_dir):
        makedirs(output_dir)

    while True:
        count = count + 1
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
            
        ret,frame = cap.read()
        atom_time_str = get_time_stamp()  #time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
          
        img2gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
        imageVarStr = "%d"%(imageVar)
        img2gray = cv2.putText(img2gray, imageVarStr, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)
        
        cv2.imshow("image",img2gray)

        if count > 5 :
            count = 0
        else:
            continue


        if ret == True :

        
            if imageVar<150. :
                print("read the image is not clear ")
                continue

            #cv2.imwrite('output/test.jpg',frame,[int(cv2.IMWRITE_JPEG_QUALITY),70])
            print("read the image at: %s"%atom_time_str)
            bname = atom_time_str
            #img_path = 'output/test.jpg'
            #license plate detection
            print ('Searching for license plates using WPOD-NET')
            #print ("Processing:%s"% img_path)
            #Ivehicle = cv2.imread(img_path)
            Ivehicle = frame
            ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
            side  = int(ratio*288.)
            bound_dim = min(side + (side%(2**4)),608)
            print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))
            Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)
            if len(LlpImgs):
                Ilp = LlpImgs[0]
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
                s = Shape(Llp[0].pts)
                cv2.imwrite("output/test_lp.jpg",Ilp*255.)
                #license plate ocr
                print ('Performing OCR...')

                #Ilp_image = nparray_to_image(Ilp)

                image_path = "output/test_lp.jpg"
    
                R,(width,height) = detect(ocr_net, ocr_meta, image_path ,thresh=ocr_threshold, nms=None)
            
                if len(R):
                    L = dknet_label_conversion(R,width,height)
                    L = nms(L,.45)
                    L.sort(key=lambda x: x.tl()[0])
                    lp_str = ''.join([chr(l.cl()) for l in L])

                    
                    cur_lp_str = lp_str
                    if len(lp_str) < 5:
                        print("the lp_str too short ")
                        continue

                    if (cur_lp_str == last_lp_str):
                        print("the lp_str repeate")
                        continue
                    

                    with open('%s/%s_str.txt' % (output_dir,bname),'w') as f:
                        f.write(lp_str + '\n')
                    Ilp = cv2.putText(Ilp, lp_str, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)        
                    cv2.imshow("Ilp",Ilp)
                    print ('\t\tFound LP: %s' % lp_str)
                    cv2.imwrite('%s/%s_lp.png' % (output_dir,bname),Ilp*255.)

                    last_lp_str = cur_lp_str 

                else:
                    print ('No characters found')  
            else:
                print("can't detect license plate ")
        else:
            print("can't read the image")
        


    cv2.destroyAllWindows()
    cap.release()

    sys.exit(0)


    




