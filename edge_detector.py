import os,cv2
import numpy as np

class detector(object):
    """Detecting edges from multi-source images with multi-control boards"""
    def __init__(self):
        self.stockImg = []      # image stock
        self.img_path = "./imgs/"
        self.imgRef = cv2.imread(self.img_path+"base.png")
        self.img0 = cv2.imread(self.img_path+"315.png")
        self.img1 = cv2.imread(self.img_path+"slope.png")
        self.imgList = [self.img0,self.img1]
        self.edgesColor = [(0,255,0),(0,0,255)]
        self.blurList = []
        self.contoursList = []
        self.drawingList = []
        self.count = 0
        # changing trackbar values
        self.a, self.b, self.c, self.d, self.e, self.f = {},{},{},{},{},{}
        # old trackbar values
        self.a_old, self.b_old, self.c_old, self.d_old, self.e_old, self.f_old = {},{},{},{},{},{}
        # Get grayscale images
        self.get_grayscale()

    def get_grayscale(self):
        """Transform RGB to GRAY"""
        l = 0
        while l < len(self.imgList):
            grayVal = cv2.cvtColor(self.imgList[l],cv2.COLOR_BGR2GRAY)
            self.stockImg.append(grayVal)
            l += 1

    def main(self,*args):
        """Main function called by trackbars in created windows each time a modifier value is changed"""
        # create trackbars for images
        k = 0
        while k <= len(self.stockImg):
            # trackbars for modifiers
            contrastV = cv2.getTrackbarPos('contrast:', 'input{}'.format(k))
            blurV = cv2.getTrackbarPos('blur:', 'input{}'.format(k))
            edgesV = cv2.getTrackbarPos('edges:', 'input{}'.format(k))
            ratioV = cv2.getTrackbarPos('ratio:', 'input{}'.format(k))
            angleMinV = cv2.getTrackbarPos('angle_min:', 'input{}'.format(k))
            angleMaxV = cv2.getTrackbarPos('angle_max:', 'input{}'.format(k))
            # value of modifiers
            self.a[k] = contrastV
            self.b[k] = blurV
            self.c[k] = edgesV
            self.d[k] = ratioV
            self.e[k] = angleMinV
            self.f[k] = angleMaxV
            k += 1
        if self.count == 0:
            # if initial state...
            self.storedValues()         # initiate modifier values
            self.contrast_callback()    # get initial blurs
            self.count += 1
        else:
            self.checkFunction()        # call this function every time trackbar is modified to apply modifiers
        self.storedValues()             # store new values each time a change is applied to image(s)

    def checkFunction(self):
        """Check if a value has changed i.e. what trackbar from which input window has been changed
        by comparing new and old values (triggers corresponding function to be called)"""
        for i in range(len(self.stockImg)):
            if self.a[i] != self.a_old[i]:
                self.contrast_callback()
            elif self.b[i] != self.b_old[i]:
                self.blur_callback()
            elif self.c[i] != self.c_old[i]:
                self.thresh_Edges()
            elif self.d[i] != self.d_old[i]:
                self.ratio_callback()
            elif self.e[i] != self.e_old[i]:
                self.angle_callback()
            elif self.f[i] != self.f_old[i]:
                self.angle_callback()
            else:
                pass

    def contrast_callback(self):
        """Contrast modifier"""
        self.blur_callback()

    def blur_callback(self):
        """Blur modifier"""
        # Parameters
        maxIntensity = 255.0 # depends on dtype of image data
        x = np.arange(maxIntensity)
        phi,theta = 1,1
        self.blurList = []
        for i in range(len(self.stockImg)):
            grayMod = (maxIntensity/phi)*(self.stockImg[i]/(maxIntensity/theta))**self.a[i]
            grayMod = np.array(grayMod,np.uint8)
            if self.b[i] % 2 == 1:
                blur = cv2.GaussianBlur(grayMod,(self.b[i],self.b[i]),0)
                self.blurList.append(blur)
            else:
                blurValMod = self.b[i] - 1
                blur = cv2.GaussianBlur(grayMod,(blurValMod,blurValMod),0)
                self.blurList.append(blur)
        self.thresholding()

    def thresh_Edges(self):
        """Edges threshold modifier"""
        self.thresholding()

    def thresholding(self):
        """Contouring function"""    
        for i in range(len(self.blurList)):
            edges = cv2.Canny(self.blurList[i],self.c[i]*0.5,self.c[i]*1.5)
            #ret,threshImg = cv2.threshold(blurList[i],c[i],255,0)
            contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            self.contoursList.append(contours)
        self.ratio()

    def ratio_callback(self):
        """Ratio modifier"""
        self.ratio()

    def angle_callback(self):
        self.ratio()

    def ratio(self):
        """Drawing function"""
        for i in range(len(self.blurList)):
            drawing_edges = np.zeros(self.img0.shape,np.uint8)
            for cnt in self.contoursList[i]:
                rect = cv2.minAreaRect(cnt)
                width = rect[1][0]
                height = rect[1][1]
                ratio = width/height if height != 0 else 0
                ratio2 = height/width if width != 0 else 0
                angle = rect[2]
                points = cv2.boxPoints(rect)
                points = np.int0(np.around(points))
                if ratio > self.d[i] and -self.e[i] >= angle >= -self.f[i]:
                    cv2.drawContours(drawing_edges,[cnt],0,self.edgesColor[i],2)
                if ratio2 > self.d[i] and -self.e[i] >= angle >= -self.f[i]:
                    cv2.drawContours(drawing_edges,[cnt],0,self.edgesColor[i],2)
            self.drawingList.append(drawing_edges)
        alpha = 1.0
        beta = 1.0
        gamma = 100
        imgEdges = np.zeros(self.img0.shape,np.uint8)
        dst_ref = 0
        for i in range(1,len(self.blurList)):
            dst_ref += self.drawingList[i-1] * beta + self.drawingList[i] * beta + gamma
            imgEdges += cv2.addWeighted(self.drawingList[i-1], beta, self.drawingList[i], beta, 0, dst_ref)
        dst = self.imgRef*alpha + imgEdges*beta + gamma
        imgSuperimposed = cv2.addWeighted(self.imgRef, alpha, imgEdges, beta, 0, dst)
        self.drawingList = []
        self.img_show(imgSuperimposed)
    
    def storedValues(self):
        """Store current values as old values"""
        k = 0
        while k <= len(self.stockImg): 
            self.a_old[k] = self.a[k]
            self.b_old[k] = self.b[k]
            self.c_old[k] = self.c[k]
            self.d_old[k] = self.d[k]
            self.e_old[k] = self.e[k]
            self.f_old[k] = self.f[k]
            k += 1

    def img_show(self,img):            
        cv2.imshow('Edges',img)
        for i in range(len(self.imgList)):
            cv2.imshow('img{}'.format(i),self.blurList[i])

    def detect(self):
        # Create windows (for controllers)
        for i in range(len(self.imgList)):
            cv2.namedWindow('input{}'.format(i),cv2.WINDOW_NORMAL) #cv2.WINDOW_NORMAL is resize; WINDOW_AUTOSIZE
            cv2.resizeWindow('input{}'.format(i), 500, 200)
            cv2.namedWindow('img{}'.format(i),cv2.WINDOW_NORMAL)
            cv2.resizeWindow('img{}'.format(i), 200, 75)
        # Create window to display edges
        cv2.namedWindow('Edges',cv2.WINDOW_NORMAL)
        # Default parameters
        threshEdges = 184
        max_threshEdges = 255
        blurVal = 7
        blurMax = 9
        ratioVal = 5
        ratioMax = 10
        contrastVal = 2
        contrastMax = 8
        angle_minVal = 30
        angle_min_Max = 360
        angle_maxVal = 70
        angle_max_Max = 360
        # Create controllers
        for i in range(len(self.imgList)):
            cv2.createTrackbar('blur:','input{}'.format(i),blurVal,blurMax,self.main)
            cv2.createTrackbar('edges:','input{}'.format(i),threshEdges,max_threshEdges,self.main)
            cv2.createTrackbar('ratio:','input{}'.format(i),ratioVal,ratioMax,self.main)
            cv2.createTrackbar('contrast:','input{}'.format(i),contrastVal,contrastMax,self.main)
            cv2.createTrackbar('angle_min:','input{}'.format(i),angle_minVal,angle_min_Max,self.main)
            cv2.createTrackbar('angle_max:','input{}'.format(i),angle_maxVal,angle_max_Max,self.main)
        # Initiate the main function
        self.main()
        if cv2.waitKey(0) == 27: # if you press ESC, it will close all windows
            cv2.destroyAllWindows()
