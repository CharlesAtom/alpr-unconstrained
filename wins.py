import sys, os
import cv2
import numpy as np
import traceback
import time
import keras
import darknet.python.darknet as dn

from darknet.python.darknet import detect
from src.label 				import Label, lwrite ,Shape, writeShapes, dknet_label_conversion
from src.keras_utils 		import load_model, detect_lp
from src.utils 				import crop_region, image_files_from_folder,im2single, nms
from os.path 				import splitext, basename, isdir
from os 					import makedirs
from glob 					import glob



def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))



if __name__ == '__main__':


    input_device  = sys.argv[1]
    output_dir = sys.argv[2]

    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    #time_length = 30.0
    #fps=25
    #frame_seq = 749
    #frame_no = (frame_seq /(time_length*fps))
    #cap.set(2,frame_no)
    #cap = cv2.VideoCapture("rtsp://admin:admin@192.168.2.64:554//Streaming/Channels/1")
    #cap.set(3,640)
    #cap.set(4,480)
    #width=cap.get(3)
    #height=cap.get(4)
    #print ("width:%d"%width)
    #print ("height:%d"%height)

    #src=cv2.imread('samples/Images/0048.jpg')       
    #cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
    #cv2.imshow('input_image', src)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #vehicle detection model load
    vehicle_threshold = .5
    vehicle_weights = 'data/vehicle-detector/yolo-voc.weights'
    vehicle_netcfg  = 'data/vehicle-detector/yolo-voc.cfg'
    vehicle_dataset = 'data/vehicle-detector/voc.data'
    vehicle_net  = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
    vehicle_meta = dn.load_meta(vehicle_dataset)

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
        ret,frame = cap.read()
        atom_time_str = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        if ret == True :
            cv2.imshow("image",frame)
            cv2.imwrite('output/test.jpg',frame,[int(cv2.IMWRITE_JPEG_QUALITY),70])
            print("read the image at: %s"%atom_time_str)

            #vehicle detection
            print ('Searching for vehicles using YOLO...')
            print ('Scanning test.jpg...')
            #name = basename(splitext(img_path)[0])
            bname = atom_time_str
            img_path = 'output/test.jpg'
            R,_ = detect(vehicle_net, vehicle_meta, img_path ,thresh=vehicle_threshold)
            R = [r for r in R if r[0] in ['car','bus']]
            print ('\t\t%d cars found' % len(R))
            if len(R):
                Iorig = cv2.imread(img_path)
                cv2.imwrite('output/%s.png'%bname,frame)
                WH = np.array(Iorig.shape[1::-1],dtype=float)
                Lcars = []
                for i,r in enumerate(R):
                    cx,cy,w,h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
                    tl = np.array([cx - w/2., cy - h/2.])
                    br = np.array([cx + w/2., cy + h/2.])
                    label = Label(0,tl,br)
                    Icar = crop_region(Iorig,label)
                    Lcars.append(label)
                    cv2.imwrite('%s/%s_%dcar.png' % (output_dir,bname,i),Icar)
                lwrite('%s/%s_cars.txt' % (output_dir,bname),Lcars)

                #license plate detection
                print ('Searching for license plates using WPOD-NET')
                imgs_paths = glob('%s/*car.png' % output_dir)
                for img_path in enumerate(imgs_paths):
                    print ("Processing:%s"% img_path)
                    bname = splitext(basename(img_path))[0]
                    Ivehicle = cv2.imread(img_path)
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
                        cv2.imwrite('%s/%s_lp.png' % (output_dir,bname),Ilp*255.)
                        writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])

                        #license plate ocr
                        print ('Performing OCR...')
                        imgs_paths = "%s/%s_lp.png"%output_dir,bname
                        #imgs_paths = sorted(glob('%s/*lp.png' % output_dir))
                        #for img_path in enumerate(imgs_paths):
                        print ('\tScanning %s' % img_path)
                        bname = basename(splitext(img_path)[0])
                        R,(width,height) = detect(ocr_net, ocr_meta, img_path ,thresh=ocr_threshold, nms=None)
                        if len(R):
                            L = dknet_label_conversion(R,width,height)
                            L = nms(L,.45)
                            L.sort(key=lambda x: x.tl()[0])
                            lp_str = ''.join([chr(l.cl()) for l in L])
                            with open('%s/%s_str.txt' % (output_dir,bname),'w') as f:
                                f.write(lp_str + '\n')
                            print ('\t\tLP: %s' % lp_str)
                        else:
                            print ('No characters found')  
                    else:
                        print("can't detect license plate ")
            else:
                print("can't find the vehicles")
        
        else:
            print("can't read the image")

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

    sys.exit(0)