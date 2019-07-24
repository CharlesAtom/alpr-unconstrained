import sys, os
import cv2
import numpy as np
import traceback
import time
import keras
import darknet.python.darknet as dn

from darknet.python.darknet import detect
from src.label 				import Label, lwrite ,Shape, writeShapes, dknet_label_conversion, lread,readShapes
from src.keras_utils 		import load_model, detect_lp
from src.utils 				import crop_region, image_files_from_folder,im2single, nms
from src.drawing_utils		import draw_label, draw_losangle, write2img
from os.path 				import splitext, basename, isdir, isfile
from os 					import makedirs
from glob 					import glob
from pdb                    import set_trace as pause



def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))



if __name__ == '__main__':


    input_device  = sys.argv[1]
    output_dir = sys.argv[2]
    OCR_SUCCEED = False

    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('output_image', cv2.WINDOW_AUTOSIZE)

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

                    #license plate detection
                    print ('Searching for license plates using WPOD-NET')
                    img_path_lp = "{dir}/{name}_{num}car.png".format(dir=output_dir,name=bname,num=i)
                    print ("Processing:%s"% img_path_lp)
                    bname_lp = splitext(basename(img_path_lp))[0]
                    Ivehicle = cv2.imread(img_path_lp)
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
                        cv2.imwrite('%s/%s_lp.png' % (output_dir,bname_lp),Ilp*255.)
                        writeShapes('%s/%s_lp.txt' % (output_dir,bname_lp),[s])
                        
                        #license plate ocr
                        print ('Performing OCR...')
                        img_path_ocr = "{dir}/{name}_lp.png".format(dir = output_dir,name = bname_lp)
                        print ("\tScanning %s"%img_path_ocr)
                        bname_ocr = basename(splitext(img_path_ocr)[0])

                        R,(width,height) = detect(ocr_net, ocr_meta, img_path_ocr ,thresh=ocr_threshold, nms=None)

                        if len(R):
                            L = dknet_label_conversion(R,width,height)
                            L = nms(L,.45)
                            L.sort(key=lambda x: x.tl()[0])
                            lp_str = ''.join([chr(l.cl()) for l in L])
                            with open('%s/%s_str.txt' % (output_dir,bname_ocr),'w') as f:
                                f.write(lp_str + '\n')
                            print ('\t\tLP: %s' % lp_str)
                            OCR_SUCCEED = True
                            
                        else:
                            print ('No characters found')  
                    else:
                        print("can't detect license plate ")
                lwrite('%s/%s_cars.txt' % (output_dir,bname),Lcars) 

            else:
                print("can't find the vehicles")
        else:
            print("can't read the image")
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        if OCR_SUCCEED:
            OCR_SUCCEED = False
            YELLOW = (  0,255,255)
            RED    = (  0,  0,255)

            img_file  = "output/{name}.png'".format(name = bname)
            I = cv2.imread(img_file)
            #detected_cars_labels = '%s/%s_cars.txt' % (output_dir,bname)
            #Lcar = lread(detected_cars_labels)
            Lcar = Lcars
            sys.stdout.write('%s' % bname)

            if Lcar:
                for j,lcar in enumerate(Lcar):
                    draw_label(I,lcar,color=YELLOW,thickness=3)
                    lp_label 		= '%s/%s_%dcar_lp.txt'		% (output_dir,bname,j)
                    lp_label_str 	= '%s/%s_%dcar_lp_str.txt'	% (output_dir,bname,j)
                    if isfile(lp_label):
                        Llp_shapes = readShapes(lp_label)
                        pts = Llp_shapes[0].pts*lcar.wh().reshape(2,1) + lcar.tl().reshape(2,1)
                        ptspx = pts*np.array(I.shape[1::-1],dtype=float).reshape(2,1)
                        draw_losangle(I,ptspx,RED,3)
                        if isfile(lp_label_str):
                            with open(lp_label_str,'r') as f:
                                lp_str = f.read().strip()
                            llp = Label(0,tl=pts.min(1),br=pts.max(1))
                            write2img(I,llp,lp_str)
                            sys.stdout.write(',%s' % lp_str)
                cv2.imshow("output_image",I)
                cv2.imwrite('%s/%s_output.png' % (output_dir,bname),I)

    cv2.destroyAllWindows()
    cap.release()

    sys.exit(0)


    




