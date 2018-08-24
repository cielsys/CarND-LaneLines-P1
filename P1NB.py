
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Project: **Finding Lane Lines on the Road** 

# ## Import Packages

# In[1]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

get_ipython().run_line_magic('matplotlib', 'inline')

import datetime
import pprint
pp = pprint.PrettyPrinter(indent=4)


# In[2]:


get_ipython().run_cell_magic('HTML', '', '<style> code {background-color : orange !important;} </style>\nfrom IPython.core.display import display, HTML\ndisplay(HTML("<style>.container { width:100% !important; }</style>"))')


# ## Helper Functions

# I am not currently using these helpers but I have left them here for future development

# In[3]:


import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


# In[4]:


def CalcSlope(lineSegIn):
    x1,y1,x2,y2 = lineSegIn
    slopeOut = (y2-y1)/(x2-x1)
    return(slopeOut)

def ExtendLineSegment(lineSegIn, extFactor = 0.25):
    x1,y1,x2,y2 = lineSegIn
    dx = x2 - x1
    dy = y2 - y1
    
    # This here is deeply cheesy!
    # Only extend the segment toward the bottom of the image
    if (dy > 0):
        x3 = int(round(x2 + extFactor * dx))
        y3 = int(round(y2 + extFactor * dy))
        x0 = x1
        y0 = y1
    else:
        x3 = x2
        y3 = y2
        x0 = int(round(x1 - extFactor * dx))
        y0 = int(round(y1 - extFactor * dy))
        
    lineSegOut = (x0,y0,x3,y3)
    return(lineSegOut)
                       


# In[5]:


def PlotImageRecords(imgRecords):
    fig = plt.gcf()
    fig.set_size_inches(12,4)
    fig.set_dpi(180)

    numImages = len(imgRecords)
    numCols = 4
    numRows = math.ceil(numImages/numCols) 
    for recIndex, imgRecord in enumerate(imgRecords):
        name, img = imgRecord

        plt.subplot(numRows, numCols, recIndex+1)
        plt.title(name)
        plt.axis('off')
        plt.imshow(img)
        
    #plt.tight_layout()
    #plt.show()        

#PlotImageRecords(pipelineImgRecords)
    


# ## Test Images
# 

# In[6]:


import os
from glob import glob
g_outputDir = "test_videos_output"

glob_pattern = os.path.join("test_images/", "*.jpg")
tmpFileNames = glob(glob_pattern)
g_testImgFileNames = tmpFileNames
g_testImgFileNamesFQ = [os.path.abspath(fileName) for fileName in tmpFileNames]


# ## Main Pipeline Implementation

# In[7]:


def ProcessImage(imgIn):
    """
    Wrapper function - because moviepy process function does not like the extra return obj
    """
    pipelineImages, imgOut = ProcessImageLowlevel(imgIn)
    return imgOut
    
def ProcessImageLowlevel(imgIn, imgInFileName="imageIn"):
    """
    Create the pipeline processed image on the input image.
    :param imgIn: Input matplotlib image
    :param imgInFileName: Optional filename of the input image - for debug display
    :returns imgOut: The processed image final result
    :returns pipelineImages: A list of the intermediate images of the pipeline for debug display
    """
    imgGray = cv2.cvtColor(imgIn, cv2.COLOR_RGB2GRAY)

    # Define a kernel size and apply Gaussian smoothing
    kernelSize = (5, 5)
    imgBlurGray = cv2.GaussianBlur(imgGray, kernelSize, 0)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    imgEdges = cv2.Canny(imgBlurGray, low_threshold, high_threshold)

    # This time we are defining a four sided polygon to mask
    # Next we'll create a masked edges image using cv2.fillPoly()
    maskROI = np.zeros_like(imgEdges)   
    ignore_mask_color = 255   
    
    # Region of interest mask polygon
    iHeight = imgIn.shape[0]
    iWidth = imgIn.shape[1]
    trapTop = 320
    leftbottom = (110, iHeight)
    lefttop = (440, trapTop)
    righttop = (520, trapTop)
    rightbottom = (iWidth-75, iHeight)
    
    vertices = np.array([[ leftbottom, lefttop, righttop, rightbottom]], dtype=np.int32)
    cv2.fillPoly(maskROI, vertices, ignore_mask_color)
    imgMaskedEdges = cv2.bitwise_and(imgEdges, maskROI)
    
    maskROI3Ch = np.dstack((maskROI, maskROI, maskROI)) 
    imgMaskedInput = cv2.bitwise_and(imgIn, maskROI3Ch)
    
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta =   0.1 * np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 50    # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50#minimum number of pixels making up a line
    max_line_gap = 200    # maximum gap in pixels between connectable line segments
    imgLines = np.zeros_like(imgIn)   
    
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(imgMaskedEdges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    # Iterate over the output "lines" and draw lines on a blank image
    print("numLines: ", len(lines))
    slopesP = []
    slopesN = []
    
    for line in lines:  
        #print(line[0])
        for x1,y1,x2,y2 in line:            
            cv2.line(imgLines, (x1,y1), (x2,y2), (255,0,0), 10)
            lineExt = ExtendLineSegment(line[0], extFactor = 1)
            x1,y1,x2,y2 = lineExt
            cv2.line(imgLines, (x1,y1), (x2,y2), (255,0,0), 10)
            
            slope = CalcSlope(line[0])          
            if (slope > 0):
                slopesP.append(slope)
            else:
                slopesN.append(slope)            
            
    # Create a "color" binary image to combine with line image
    imgEdges3Ch = np.dstack((imgEdges, imgEdges, imgEdges)) 
    
    # Draw the lines on the edge image
    imgLinesEdges = cv2.addWeighted(imgEdges3Ch, 1, imgLines, 1.0, 0)
    
    # Overlay lines on input image
    imgLinesOnInput = cv2.addWeighted(imgLines, 1, imgIn, 0.7, 20) 
    #imgLinesOnInput = imgIn.copy()
    #lineMaskPixels = np.where(imgLines[:, :, 0] != 0)
    #imgLinesOnInput[lineMaskPixels] = imgLines[lineMaskPixels]
    
    imgOut = imgLinesOnInput
    # Create list of images for each step for dev/debug
    doDebugImages = True
    if doDebugImages :
        pipelineImages = [
            (imgInFileName, imgIn),
            ("imgMaskedInput", imgMaskedInput),
            #("imgGray", np.dstack((imgGray, imgGray, imgGray)) ),
            #("imgBlurGray", cv2.cvtColor(imgBlurGray,cv2.COLOR_GRAY2RGB)),
            ("imgEdges", cv2.cvtColor(imgEdges,cv2.COLOR_GRAY2RGB)),
            ("imgMaskedEdges", cv2.cvtColor(imgMaskedEdges,cv2.COLOR_GRAY2RGB)),
            #("imgLines", imgLines),
            ("imgLinesEdges", imgLinesEdges),
            ("imgLinesOnInput", imgLinesOnInput),
            ("imgOut", imgOut),        
        ]
    else:
        pipelineImages = None        
    return pipelineImages, imgOut


# In[8]:


def ProcessSingleImage(imgInFileName):
    imgIn = mpimg.imread(imgInFileName)
    pipelineImgRecords, imgOut = ProcessImageLowlevel(imgIn, imgInFileName)
    PlotImageRecords(pipelineImgRecords)

imgInFileName = 'test_images/solidYellowLeft.jpg'
#imgInFileName = 'test_images/traingleQuiz.jpg'
#imgInFileName = 'test_images/whiteCarLaneSwitch.jpg'
#imgInFileName = 'test_images/solidYellowCurve2.jpg'
#imgInFileName = 'test_images/solidWhiteCurve.jpg'
imgInFileName = 'test_images/solidWhiteRight.jpg'
#imgInFileName = 'test_images/solidYellowCurve.jpg'

ProcessSingleImage(imgInFileName)


# In[9]:


from moviepy.editor import VideoFileClip
from IPython.display import HTML
import datetime

def ProcessVideo():
    fileNameBase = "solidWhiteRight"
    fileNameBase = "solidYellowLeft"
    #fileNameBase = "challenge"
    fileExt = ".mp4"
    folderInput = "test_videos/"
    folderOutput = "test_videos_output/"

    dt = datetime.datetime.now()
    strDT = "_{:%Y-%m-%dT%H:%M:%S}".format(dt)

    fileNameIn  = folderInput + fileNameBase + fileExt
    fileNameOut = folderOutput + fileNameBase + strDT + fileExt

    #clip1 = VideoFileClip(fileNameIn).subclip(0,3)
    clip1 = VideoFileClip(fileNameIn)
    white_clip = clip1.fl_image(ProcessImage) #NOTE: this function expects color images!!
    white_clip.write_videofile(fileNameOut, audio=False)
    
ProcessVideo()

