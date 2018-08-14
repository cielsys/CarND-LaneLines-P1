{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Self-Driving Car Engineer Nanodegree\n",
    "\n",
    "\n",
    "## Project: **Finding Lane Lines on the Road** \n",
    "***\n",
    "In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip \"raw-lines-example.mp4\" (also contained in this repository) to see what the output should look like after using the helper functions below. \n",
    "\n",
    "Once you have a result that looks roughly like \"raw-lines-example.mp4\", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video \"P1_example.mp4\".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.\n",
    "\n",
    "In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.\n",
    "\n",
    "---\n",
    "Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the \"play\" button above) to display the image.\n",
    "\n",
    "**Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the \"Kernel\" menu above and selecting \"Restart & Clear Output\".**\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**\n",
    "\n",
    "---\n",
    "\n",
    "<figure>\n",
    " <img src=\"examples/line-segments-example.jpg\" width=\"380\" alt=\"Combined Image\" />\n",
    " <figcaption>\n",
    " <p></p> \n",
    " <p style=\"text-align: center;\"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> \n",
    " </figcaption>\n",
    "</figure>\n",
    " <p></p> \n",
    "<figure>\n",
    " <img src=\"examples/laneLines_thirdPass.jpg\" width=\"380\" alt=\"Combined Image\" />\n",
    " <figcaption>\n",
    " <p></p> \n",
    " <p style=\"text-align: center;\"> Your goal is to connect/average/extrapolate line segments to get output like this</p> \n",
    " </figcaption>\n",
    "</figure>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, consult the forums for more troubleshooting tips.**  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Import Packages"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "#importing some useful packages\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.image as mpimg\n",
    "import numpy as np\n",
    "import cv2\n",
    "import math\n",
    "\n",
    "%matplotlib inline\n",
    "\n",
    "import datetime\n",
    "import pprint\n",
    "pp = pprint.PrettyPrinter(indent=4)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style> code {background-color : orange !important;} </style>\n",
       "from IPython.core.display import display, HTML\n",
       "display(HTML(\"<style>.container { width:100% !important; }</style>\"))"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "%%HTML\n",
    "<style> code {background-color : orange !important;} </style>\n",
    "from IPython.core.display import display, HTML\n",
    "display(HTML(\"<style>.container { width:100% !important; }</style>\"))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Ideas for Lane Detection Pipeline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**\n",
    "\n",
    "`cv2.inRange()` for color selection  \n",
    "`cv2.fillPoly()` for regions selection  \n",
    "`cv2.line()` to draw lines on an image given endpoints  \n",
    "`cv2.addWeighted()` to coadd / overlay two images\n",
    "`cv2.cvtColor()` to grayscale or change color\n",
    "`cv2.imwrite()` to output images to file  \n",
    "`cv2.bitwise_and()` to apply a mask to an image\n",
    "\n",
    "**Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Helper Functions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Below are some helper functions to help get you started. They should look familiar from the lesson!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "import math\n",
    "\n",
    "def grayscale(img):\n",
    "    \"\"\"Applies the Grayscale transform\n",
    "    This will return an image with only one color channel\n",
    "    but NOTE: to see the returned image as grayscale\n",
    "    (assuming your grayscaled image is called 'gray')\n",
    "    you should call plt.imshow(gray, cmap='gray')\"\"\"\n",
    "    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)\n",
    "    # Or use BGR2GRAY if you read an image with cv2.imread()\n",
    "    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n",
    "    \n",
    "def canny(img, low_threshold, high_threshold):\n",
    "    \"\"\"Applies the Canny transform\"\"\"\n",
    "    return cv2.Canny(img, low_threshold, high_threshold)\n",
    "\n",
    "def gaussian_blur(img, kernel_size):\n",
    "    \"\"\"Applies a Gaussian Noise kernel\"\"\"\n",
    "    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)\n",
    "\n",
    "def region_of_interest(img, vertices):\n",
    "    \"\"\"\n",
    "    Applies an image mask.\n",
    "    \n",
    "    Only keeps the region of the image defined by the polygon\n",
    "    formed from `vertices`. The rest of the image is set to black.\n",
    "    `vertices` should be a numpy array of integer points.\n",
    "    \"\"\"\n",
    "    #defining a blank mask to start with\n",
    "    mask = np.zeros_like(img)   \n",
    "    \n",
    "    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image\n",
    "    if len(img.shape) > 2:\n",
    "        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image\n",
    "        ignore_mask_color = (255,) * channel_count\n",
    "    else:\n",
    "        ignore_mask_color = 255\n",
    "        \n",
    "    #filling pixels inside the polygon defined by \"vertices\" with the fill color    \n",
    "    cv2.fillPoly(mask, vertices, ignore_mask_color)\n",
    "    \n",
    "    #returning the image only where mask pixels are nonzero\n",
    "    masked_image = cv2.bitwise_and(img, mask)\n",
    "    return masked_image\n",
    "\n",
    "\n",
    "def draw_lines(img, lines, color=[255, 0, 0], thickness=2):\n",
    "    \"\"\"\n",
    "    NOTE: this is the function you might want to use as a starting point once you want to \n",
    "    average/extrapolate the line segments you detect to map out the full\n",
    "    extent of the lane (going from the result shown in raw-lines-example.mp4\n",
    "    to that shown in P1_example.mp4).  \n",
    "    \n",
    "    Think about things like separating line segments by their \n",
    "    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left\n",
    "    line vs. the right line.  Then, you can average the position of each of \n",
    "    the lines and extrapolate to the top and bottom of the lane.\n",
    "    \n",
    "    This function draws `lines` with `color` and `thickness`.    \n",
    "    Lines are drawn on the image inplace (mutates the image).\n",
    "    If you want to make the lines semi-transparent, think about combining\n",
    "    this function with the weighted_img() function below\n",
    "    \"\"\"\n",
    "    for line in lines:\n",
    "        for x1,y1,x2,y2 in line:\n",
    "            cv2.line(img, (x1, y1), (x2, y2), color, thickness)\n",
    "\n",
    "def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):\n",
    "    \"\"\"\n",
    "    `img` should be the output of a Canny transform.\n",
    "        \n",
    "    Returns an image with hough lines drawn.\n",
    "    \"\"\"\n",
    "    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)\n",
    "    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)\n",
    "    draw_lines(line_img, lines)\n",
    "    return line_img\n",
    "\n",
    "# Python 3 has support for cool math symbols.\n",
    "\n",
    "def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):\n",
    "    \"\"\"\n",
    "    `img` is the output of the hough_lines(), An image with lines drawn on it.\n",
    "    Should be a blank image (all black) with lines drawn on it.\n",
    "    \n",
    "    `initial_img` should be the image before any processing.\n",
    "    \n",
    "    The result image is computed as follows:\n",
    "    \n",
    "    initial_img * α + img * β + γ\n",
    "    NOTE: initial_img and img must be the same shape!\n",
    "    \"\"\"\n",
    "    return cv2.addWeighted(initial_img, α, img, β, γ)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Test Images\n",
    "\n",
    "Build your pipeline to work on the images in the directory \"test_images\"  \n",
    "**You should make sure your pipeline works well on these images before you try the videos.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 393,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['test_images/IMG_1131.JPG',\n",
       " 'test_images/IMG_1111.JPG',\n",
       " 'test_images/solidYellowLeft.jpg',\n",
       " 'test_images/DSC_7315.JPG',\n",
       " 'test_images/traingleQuiz.jpg',\n",
       " 'test_images/IMG_1117.JPG',\n",
       " 'test_images/whiteCarLaneSwitch.jpg',\n",
       " 'test_images/solidYellowCurve2.jpg',\n",
       " 'test_images/solidWhiteCurve.jpg',\n",
       " 'test_images/solidWhiteRight.jpg',\n",
       " 'test_images/solidYellowCurve.jpg']"
      ]
     },
     "execution_count": 393,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import os\n",
    "from glob import glob\n",
    "g_outputDir = \"test_videos_output\"\n",
    "\n",
    "glob_pattern = os.path.join(\"test_images/\", \"*\")\n",
    "tmpFileNames = glob(glob_pattern)\n",
    "g_testImgFileNames = tmpFileNames\n",
    "#g_testImgFileNamesFQ = [os.path.abspath(fileName) for fileName in tmpFileNames]\n",
    "#g_testImgFileNamesFQ\n",
    "g_testImgFileNames"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Build a Lane Finding Pipeline\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.\n",
    "\n",
    "Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Test on Videos\n",
    "\n",
    "You know what's cooler than drawing lanes over images? Drawing lanes over video!\n",
    "\n",
    "We can test our solution on two provided videos:\n",
    "\n",
    "`solidWhiteRight.mp4`\n",
    "\n",
    "`solidYellowLeft.mp4`\n",
    "\n",
    "**Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**\n",
    "\n",
    "**If you get an error that looks like this:**\n",
    "```\n",
    "NeedDownloadError: Need ffmpeg exe. \n",
    "You can download it by calling: \n",
    "imageio.plugins.ffmpeg.download()\n",
    "```\n",
    "**Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import everything needed to edit/save/watch video clips\n",
    "from moviepy.editor import VideoFileClip\n",
    "from IPython.display import HTML"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 304,
   "metadata": {},
   "outputs": [],
   "source": [
    "import math\n",
    "def PlotImageRecords(imgRecords):\n",
    "    fig = plt.gcf()\n",
    "    fig.set_size_inches(12,4)\n",
    "    fig.set_dpi(180)\n",
    "\n",
    "    numImages = len(imgRecords)\n",
    "    numCols = 4\n",
    "    numRows = math.ceil(numImages/numCols) \n",
    "    for recIndex, imgRecord in enumerate(imgRecords):\n",
    "        name, img = imgRecord\n",
    "\n",
    "        plt.subplot(numRows, numCols, recIndex+1)\n",
    "        plt.title(name)\n",
    "        plt.axis('off')\n",
    "        plt.imshow(img)\n",
    "        \n",
    "    #plt.tight_layout()\n",
    "    #plt.show()        \n",
    "\n",
    "#PlotImageRecords(pipelineImgRecords)\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 414,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  6\n",
      "slopesPavg = 0.640206588759\n",
      "slopesNavg = -0.700225499756\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fc1e98ae8d0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def ProcessImage(imgIn):\n",
    "    pipelineImages, imgOut = ProcessImageLowlevel(imgIn)\n",
    "    return imgOut\n",
    "    \n",
    "def ProcessImageLowlevel(imgIn, imgInFileName=\"imageIn\"):\n",
    "    \"\"\"\n",
    "    Create the pipeline processed image on the input image.\n",
    "    :param imgIn: Input matplotlib image\n",
    "    :param imgInFileName: Optional filename of the input image - for debug display\n",
    "    :returns imgOut: The processed image final result\n",
    "    :returns pipelineImages: A list of the intermediate images of the pipeline for debug display\n",
    "    \"\"\"\n",
    "    imgGray = cv2.cvtColor(imgIn, cv2.COLOR_RGB2GRAY)\n",
    "\n",
    "    # Define a kernel size and apply Gaussian smoothing\n",
    "    kernelSize = (5, 5)\n",
    "    imgBlurGray = cv2.GaussianBlur(imgGray, kernelSize, 0)\n",
    "\n",
    "    # Define our parameters for Canny and apply\n",
    "    low_threshold = 50Region\n",
    "    high_threshold = 150\n",
    "    imgEdges = cv2.Canny(imgBlurGray, low_threshold, high_threshold)\n",
    "\n",
    "    # This time we are defining a four sided polygon to mask\n",
    "    # Next we'll create a masked edges image using cv2.fillPoly()\n",
    "    maskROI = np.zeros_like(imgEdges)   \n",
    "    ignore_mask_color = 255   \n",
    "    \n",
    "    # Region of interest mask polygon\n",
    "    iHeight = imgIn.shape[0]\n",
    "    iWidth = imgIn.shape[1]\n",
    "    trapTop = 320\n",
    "    leftbottom = (110, iHeight)\n",
    "    lefttop = (440, trapTop)\n",
    "    righttop = (520, trapTop)\n",
    "    rightbottom = (iWidth-75, iHeight)\n",
    "    \n",
    "    vertices = np.array([[ leftbottom, lefttop, righttop, rightbottom]], dtype=np.int32)\n",
    "    cv2.fillPoly(maskROI, vertices, ignore_mask_color)\n",
    "    imgMaskedEdges = cv2.bitwise_and(imgEdges, maskROI)\n",
    "    \n",
    "    maskROI3Ch = np.dstack((maskROI, maskROI, maskROI)) \n",
    "    imgMaskedInput = cv2.bitwise_and(imgIn, maskROI3Ch)\n",
    "    \n",
    "    # Define the Hough transform parameters\n",
    "    # Make a blank the same size as our image to draw on\n",
    "    rho = 1 # distance resolution in pixels of the Hough grid\n",
    "    theta =   0.1 * np.pi/180 # angular resolution in radians of the Hough grid\n",
    "    threshold = 50    # minimum number of votes (intersections in Hough grid cell)\n",
    "    min_line_length = 50#minimum number of pixels making up a line\n",
    "    max_line_gap = 200    # maximum gap in pixels between connectable line segments\n",
    "    imgLines = np.zeros_like(imgIn)   \n",
    "    \n",
    "    # Run Hough on edge detected image\n",
    "    # Output \"lines\" is an array containing endpoints of detected line segments\n",
    "    lines = cv2.HoughLinesP(imgMaskedEdges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)\n",
    "\n",
    "    # Iterate over the output \"lines\" and draw lines on a blank image\n",
    "    print(\"numLines: \", len(lines))\n",
    "    slopesP = []\n",
    "    slopesN = []\n",
    "    \n",
    "    for line in lines:  \n",
    "        #print(line[0])\n",
    "        for x1,y1,x2,y2 in line:            \n",
    "            cv2.line(imgLines, (x1,y1), (x2,y2), (255,0,0), 10)\n",
    "            lineExt = ExtendLineSegment(line[0], extFactor = 1)\n",
    "            x1,y1,x2,y2 = lineExt\n",
    "            cv2.line(imgLines, (x1,y1), (x2,y2), (255,0,0), 10)\n",
    "            \n",
    "            slope = CalcSlope(line[0])          \n",
    "            if (slope > 0):\n",
    "                slopesP.append(slope)\n",
    "            else:\n",
    "                slopesN.append(slope)            \n",
    "            \n",
    "            \n",
    "    print(\"slopesPavg =\", sum(slopesP)/len(slopesP))\n",
    "    print(\"slopesNavg =\", sum(slopesN)/len(slopesN))\n",
    "\n",
    "    # Create a \"color\" binary image to combine with line image\n",
    "    imgEdges3Ch = np.dstack((imgEdges, imgEdges, imgEdges)) \n",
    "    \n",
    "    # Draw the lines on the edge image\n",
    "    imgLinesEdges = cv2.addWeighted(imgEdges3Ch, 1, imgLines, 1.0, 0)\n",
    "    \n",
    "    # Overlay lines on input image\n",
    "    imgLinesOnInput = cv2.addWeighted(imgLines, 1, imgIn, 0.7, 20) \n",
    "    #imgLinesOnInput = imgIn.copy()\n",
    "    #lineMaskPixels = np.where(imgLines[:, :, 0] != 0)\n",
    "    #imgLinesOnInput[lineMaskPixels] = imgLines[lineMaskPixels]\n",
    "    \n",
    "    imgOut = imgLinesOnInput\n",
    "    # Create list of images for each step for dev/debug\n",
    "    doDebugImages = True\n",
    "    if doDebugImages :\n",
    "        pipelineImages = [\n",
    "            (imgInFileName, imgIn),\n",
    "            (\"imgMaskedInput\", imgMaskedInput),\n",
    "            #(\"imgGray\", np.dstack((imgGray, imgGray, imgGray)) ),\n",
    "            #(\"imgBlurGray\", cv2.cvtColor(imgBlurGray,cv2.COLOR_GRAY2RGB)),\n",
    "            (\"imgEdges\", cv2.cvtColor(imgEdges,cv2.COLOR_GRAY2RGB)),\n",
    "            (\"imgMaskedEdges\", cv2.cvtColor(imgMaskedEdges,cv2.COLOR_GRAY2RGB)),\n",
    "            #(\"imgLines\", imgLines),\n",
    "            (\"imgLinesEdges\", imgLinesEdges),\n",
    "            (\"imgLinesOnInput\", imgLinesOnInput),\n",
    "            (\"imgOut\", imgOut),        \n",
    "        ]\n",
    "    else:\n",
    "        pipelineImages = None        \n",
    "    return pipelineImages, imgOut\n",
    "\n",
    "imgInFileName = 'test_images/solidYellowLeft.jpg'\n",
    "#imgInFileName = 'test_images/traingleQuiz.jpg'\n",
    "#imgInFileName = 'test_images/whiteCarLaneSwitch.jpg'\n",
    "#imgInFileName = 'test_images/solidYellowCurve2.jpg'\n",
    "#imgInFileName = 'test_images/solidWhiteCurve.jpg'\n",
    "imgInFileName = 'test_images/solidWhiteRight.jpg'\n",
    "#imgInFileName = 'test_images/solidYellowCurve.jpg'\n",
    "#imgInFileName = 'test_images/IMG_1131.JPG',\n",
    "#imgInFileName = 'test_images/IMG_1111.JPG',\n",
    "#imgInFileName = 'test_images/DSC_7315.JPG',\n",
    "#imgInFileName = 'test_images/a.jpg',\n",
    "ProcessSingleImage(imgInFileName)\n",
    "#ProcessVideo()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 402,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 411,
   "metadata": {},
   "outputs": [],
   "source": [
    "def OpenIphonejpg():\n",
    "    from PIL import Image\n",
    "    imgInPil = Image.open('test_images/a.jpg')\n",
    "    imgIn = np.array(imgInPil)\n",
    "    \n",
    "def ProcessSingleImage(imgInFileName):\n",
    "\n",
    "    imgIn = mpimg.imread(imgInFileName)\n",
    "    pipelineImgRecords, imgOut = ProcessImageLowlevel(imgIn, imgInFileName)\n",
    "    PlotImageRecords(pipelineImgRecords)\n",
    "\n",
    "imgInFileName = 'test_images/solidYellowLeft.jpg'\n",
    "#imgInFileName = 'test_images/traingleQuiz.jpg'\n",
    "#imgInFileName = 'test_images/whiteCarLaneSwitch.jpg'\n",
    "#imgInFileName = 'test_images/solidYellowCurve2.jpg'\n",
    "#imgInFileName = 'test_images/solidWhiteCurve.jpg'\n",
    "imgInFileName = 'test_images/solidWhiteRight.jpg'\n",
    "#imgInFileName = 'test_images/solidYellowCurve.jpg'\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 406,
   "metadata": {},
   "outputs": [],
   "source": [
    "def CalcSlope(lineSegIn):\n",
    "    x1,y1,x2,y2 = lineSegIn\n",
    "    slopeOut = (y2-y1)/(x2-x1)\n",
    "    return(slopeOut)\n",
    "\n",
    "def ExtendLineSegment(lineSegIn, extFactor = 0.25):\n",
    "    x1,y1,x2,y2 = lineSegIn\n",
    "    dx = x2 - x1\n",
    "    dy = y2 - y1\n",
    "    \n",
    "    # This here is deeply cheesy!\n",
    "    # Only extend the segment toward the bottom of the image\n",
    "    if (dy > 0):\n",
    "        x3 = int(round(x2 + extFactor * dx))\n",
    "        y3 = int(round(y2 + extFactor * dy))\n",
    "        x0 = x1\n",
    "        y0 = y1\n",
    "    else:\n",
    "        x3 = x2\n",
    "        y3 = y2\n",
    "        x0 = int(round(x1 - extFactor * dx))\n",
    "        y0 = int(round(y1 - extFactor * dy))\n",
    "        \n",
    "    lineSegOut = (x0,y0,x3,y3)\n",
    "    return(lineSegOut)\n",
    "                       "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 384,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  27\n",
      "slopesPavg = 0.337749488219\n",
      "slopesNavg = -0.309368004367\n",
      "[MoviePy] >>>> Building video test_videos_output/challenge_2018-08-13T22:02:11.mp4\n",
      "[MoviePy] Writing video test_videos_output/challenge_2018-08-13T22:02:11.mp4\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  0%|          | 0/251 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  27\n",
      "slopesPavg = 0.337749488219\n",
      "slopesNavg = -0.309368004367\n",
      "numLines:  27\n",
      "slopesPavg = 0.372380158346\n",
      "slopesNavg = -0.296593024868\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  1%|          | 2/251 [00:00<00:23, 10.58it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  31\n",
      "slopesPavg = 0.431205575622\n",
      "slopesNavg = -0.343539912313\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  2%|▏         | 4/251 [00:00<00:27,  8.95it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  25\n",
      "slopesPavg = 0.240108506712\n",
      "slopesNavg = -0.391270472829\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  2%|▏         | 6/251 [00:00<00:24, 10.00it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  26\n",
      "slopesPavg = 0.218087209943\n",
      "slopesNavg = -0.313474791075\n",
      "numLines:  26\n",
      "slopesPavg = 0.260899850999\n",
      "slopesNavg = -0.291017801172\n",
      "numLines:  27\n",
      "slopesPavg = 0.276643931398\n",
      "slopesNavg = -0.283298044515\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  3%|▎         | 8/251 [00:00<00:24,  9.95it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  25\n",
      "slopesPavg = 0.216114819429\n",
      "slopesNavg = -0.349105365047\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  4%|▎         | 9/251 [00:00<00:25,  9.67it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  26\n",
      "slopesPavg = 0.320343829093\n",
      "slopesNavg = -0.290383424925\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  4%|▍         | 10/251 [00:01<00:24,  9.66it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  26\n",
      "slopesPavg = 0.31448039321\n",
      "slopesNavg = -0.329268006827\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  4%|▍         | 11/251 [00:01<00:25,  9.42it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  30\n",
      "slopesPavg = 0.395690346927\n",
      "slopesNavg = -0.31941875868\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  5%|▍         | 12/251 [00:01<00:25,  9.33it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  28\n",
      "slopesPavg = 0.321430737549\n",
      "slopesNavg = -0.33981627992\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  6%|▌         | 14/251 [00:01<00:24,  9.55it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  29\n",
      "slopesPavg = 0.486518381629\n",
      "slopesNavg = -0.328114551449\n",
      "numLines:  27\n",
      "slopesPavg = 0.44666619956\n",
      "slopesNavg = -0.253965513397\n",
      "numLines:  25\n",
      "slopesPavg = 0.266991088066\n",
      "slopesNavg = -0.297292516939\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  6%|▋         | 16/251 [00:01<00:25,  9.28it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  28\n",
      "slopesPavg = 0.279672783809\n",
      "slopesNavg = -0.261894963888\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  7%|▋         | 17/251 [00:01<00:25,  9.28it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  25\n",
      "slopesPavg = 0.242946263214\n",
      "slopesNavg = -0.271402989782\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  7%|▋         | 18/251 [00:01<00:25,  9.24it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  27\n",
      "slopesPavg = 0.388612066759\n",
      "slopesNavg = -0.281678809767\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  8%|▊         | 19/251 [00:02<00:25,  9.28it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  27\n",
      "slopesPavg = 0.318222124491\n",
      "slopesNavg = -0.300820447983\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  8%|▊         | 20/251 [00:02<00:24,  9.25it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  29\n",
      "slopesPavg = 0.340162588986\n",
      "slopesNavg = -0.293684813727\n",
      "numLines:  23\n",
      "slopesPavg = 0.211840172709\n",
      "slopesNavg = -0.333156394498\n",
      "numLines:  28\n",
      "slopesPavg = 0.409245763619\n",
      "slopesNavg = -0.246088112405\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  9%|▉         | 22/251 [00:02<00:24,  9.41it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  29\n",
      "slopesPavg = 0.363803963877\n",
      "slopesNavg = -0.306606286963\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 10%|▉         | 25/251 [00:02<00:23,  9.58it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  30\n",
      "slopesPavg = 0.415040728268\n",
      "slopesNavg = -0.350575015469\n",
      "numLines:  25\n",
      "slopesPavg = 0.383342690085\n",
      "slopesNavg = -0.324180600995\n",
      "numLines:  29\n",
      "slopesPavg = 0.432448425316\n",
      "slopesNavg = -0.324530202374\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 11%|█         | 27/251 [00:02<00:23,  9.73it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  27\n",
      "slopesPavg = 0.270877420687\n",
      "slopesNavg = -0.340043882173\n",
      "numLines:  27\n",
      "slopesPavg = 0.232482422893\n",
      "slopesNavg = -0.347794332477\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 12%|█▏        | 29/251 [00:02<00:22,  9.94it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  26\n",
      "slopesPavg = 0.389556808376\n",
      "slopesNavg = -0.298754061099\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 12%|█▏        | 31/251 [00:03<00:21, 10.11it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  28\n",
      "slopesPavg = 0.398809644594\n",
      "slopesNavg = -0.279715282903\n",
      "numLines:  29\n",
      "slopesPavg = 0.35773770409\n",
      "slopesNavg = -0.322937066547\n",
      "numLines:  31\n",
      "slopesPavg = 0.330386751131\n",
      "slopesNavg = -0.278652219082\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 13%|█▎        | 33/251 [00:03<00:21, 10.15it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  28\n",
      "slopesPavg = 0.3666777682\n",
      "slopesNavg = -0.349695884661\n",
      "numLines:  26\n",
      "slopesPavg = 0.426942020139\n",
      "slopesNavg = -0.313549340278\n",
      "numLines:  32\n",
      "slopesPavg = 0.411252668317\n",
      "slopesNavg = -0.372743212159\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 14%|█▍        | 35/251 [00:03<00:21, 10.13it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  33\n",
      "slopesPavg = 0.360602384421\n",
      "slopesNavg = -0.289171224866\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 15%|█▍        | 37/251 [00:03<00:21, 10.19it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  32\n",
      "slopesPavg = 0.284469991638\n",
      "slopesNavg = -0.374937879711\n",
      "numLines:  31\n",
      "slopesPavg = 0.333870959268\n",
      "slopesNavg = -0.311582671726\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 16%|█▌        | 39/251 [00:03<00:20, 10.28it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  27\n",
      "slopesPavg = 0.324896610626\n",
      "slopesNavg = -0.317478307643\n",
      "numLines:  24\n",
      "slopesPavg = 0.228287912341\n",
      "slopesNavg = -0.353889077214\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 16%|█▋        | 41/251 [00:04<00:20, 10.23it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  25\n",
      "slopesPavg = 0.196293703981\n",
      "slopesNavg = -0.356127437996\n",
      "numLines:  29\n",
      "slopesPavg = 0.394322154516\n",
      "slopesNavg = -0.37928464262\n",
      "numLines:  36\n",
      "slopesPavg = 0.374281918527\n",
      "slopesNavg = -0.334196715158\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 18%|█▊        | 45/251 [00:04<00:20, 10.05it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  31\n",
      "slopesPavg = 0.317677499814\n",
      "slopesNavg = -0.334568273407\n",
      "numLines:  33\n",
      "slopesPavg = 0.23157357778\n",
      "slopesNavg = -0.297830163732\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 19%|█▊        | 47/251 [00:04<00:20, 10.10it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  33\n",
      "slopesPavg = 0.382409716064\n",
      "slopesNavg = -0.304919422153\n",
      "numLines:  28\n",
      "slopesPavg = 0.350346223399\n",
      "slopesNavg = -0.344355580094\n",
      "numLines:  34\n",
      "slopesPavg = 0.290190765384\n",
      "slopesNavg = -0.315880962131\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 20%|█▉        | 49/251 [00:04<00:19, 10.12it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  35\n",
      "slopesPavg = 0.278317600226\n",
      "slopesNavg = -0.31646106216\n",
      "numLines:  38\n",
      "slopesPavg = 0.285138441365\n",
      "slopesNavg = -0.256032667584\n",
      "numLines:  37\n",
      "slopesPavg = 0.390075324399\n",
      "slopesNavg = -0.264070561502\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 21%|██        | 53/251 [00:05<00:19, 10.03it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  34\n",
      "slopesPavg = 0.374658681548\n",
      "slopesNavg = -0.267095741407\n",
      "numLines:  31\n",
      "slopesPavg = 0.3082361435\n",
      "slopesNavg = -0.278666951868\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 22%|██▏       | 55/251 [00:05<00:19,  9.94it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  34\n",
      "slopesPavg = 0.46702969214\n",
      "slopesNavg = -0.268722614061\n",
      "numLines:  33\n",
      "slopesPavg = 0.449563982581\n",
      "slopesNavg = -0.27074280861\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 23%|██▎       | 57/251 [00:05<00:19,  9.82it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  36\n",
      "slopesPavg = 0.303710803573\n",
      "slopesNavg = -0.315199344182\n",
      "numLines:  35\n",
      "slopesPavg = 0.401464080199\n",
      "slopesNavg = -0.252288795814\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 24%|██▎       | 59/251 [00:06<00:19,  9.79it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  38\n",
      "slopesPavg = 0.31635714273\n",
      "slopesNavg = -0.259329266542\n",
      "numLines:  43\n",
      "slopesPavg = 0.345009461203\n",
      "slopesNavg = -0.288911398021\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 24%|██▍       | 61/251 [00:06<00:19,  9.74it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  36\n",
      "slopesPavg = 0.390777081943\n",
      "slopesNavg = -0.3299247577\n",
      "numLines:  37\n",
      "slopesPavg = 0.300049772609\n",
      "slopesNavg = -0.362232628625\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 25%|██▌       | 63/251 [00:06<00:19,  9.69it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  40\n",
      "slopesPavg = 0.295906747977\n",
      "slopesNavg = -0.354467654586\n",
      "numLines:  33\n",
      "slopesPavg = 0.169602634468\n",
      "slopesNavg = -0.327959208529\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 26%|██▌       | 65/251 [00:06<00:19,  9.58it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  47\n",
      "slopesPavg = 0.221669220869\n",
      "slopesNavg = -0.362269576274\n",
      "numLines:  51\n",
      "slopesPavg = 0.274095347645\n",
      "slopesNavg = -0.416108510283\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 27%|██▋       | 67/251 [00:07<00:19,  9.50it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  40\n",
      "slopesPavg = 1.32421837306\n",
      "slopesNavg = -0.3034747386\n",
      "numLines:  38\n",
      "slopesPavg = 0.251251475831\n",
      "slopesNavg = -0.295793327269\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 27%|██▋       | 69/251 [00:07<00:19,  9.48it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  35\n",
      "slopesPavg = 0.192885559251\n",
      "slopesNavg = -0.347083111176\n",
      "numLines:  35\n",
      "slopesPavg = 0.325111774308\n",
      "slopesNavg = -0.253006896971\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 28%|██▊       | 71/251 [00:07<00:18,  9.49it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  36\n",
      "slopesPavg = 0.271580930011\n",
      "slopesNavg = -0.28452741572\n",
      "numLines:  36\n",
      "slopesPavg = 0.281452391885\n",
      "slopesNavg = -0.240294726123\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 29%|██▉       | 73/251 [00:07<00:18,  9.48it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  40\n",
      "slopesPavg = 2.56332926644\n",
      "slopesNavg = -0.261385745591\n",
      "numLines:  39\n",
      "slopesPavg = 2.88003865841\n",
      "slopesNavg = -0.244453492778\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 30%|██▉       | 75/251 [00:07<00:18,  9.47it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  39\n",
      "slopesPavg = 3.04305199467\n",
      "slopesNavg = -0.349457191428\n",
      "numLines:  40\n",
      "slopesPavg = 3.76516106499\n",
      "slopesNavg = -0.270810646049\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 31%|███       | 77/251 [00:08<00:18,  9.46it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  40\n",
      "slopesPavg = 4.78234975284\n",
      "slopesNavg = -0.218405598001\n",
      "numLines:  39\n",
      "slopesPavg = 2.63137881326\n",
      "slopesNavg = -0.243850820938\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 31%|███▏      | 79/251 [00:08<00:18,  9.44it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  33\n",
      "slopesPavg = 4.08071432939\n",
      "slopesNavg = -0.203025916115\n",
      "numLines:  43\n",
      "slopesPavg = 2.71636101604\n",
      "slopesNavg = -0.211946913591\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 32%|███▏      | 81/251 [00:08<00:18,  9.40it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  38\n",
      "slopesPavg = 4.07748784574\n",
      "slopesNavg = -0.203061097273\n",
      "numLines:  39\n",
      "slopesPavg = 3.21991914909\n",
      "slopesNavg = -0.241411146497\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 33%|███▎      | 83/251 [00:08<00:17,  9.40it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  47\n",
      "slopesPavg = 4.79435981324\n",
      "slopesNavg = -0.160103912869\n",
      "numLines:  51\n",
      "slopesPavg = 1.31488519954\n",
      "slopesNavg = -0.197050420822\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 34%|███▍      | 85/251 [00:09<00:17,  9.33it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  52\n",
      "slopesPavg = 2.4052964767\n",
      "slopesNavg = -0.211435095223\n",
      "numLines:  48\n",
      "slopesPavg = 1.81526744158\n",
      "slopesNavg = -0.210342439464\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 35%|███▍      | 87/251 [00:09<00:17,  9.27it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  51\n",
      "slopesPavg = 1.70132458856\n",
      "slopesNavg = -0.183239226953\n",
      "numLines:  52\n",
      "slopesPavg = 1.59287915835\n",
      "slopesNavg = -0.185776641183\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 35%|███▌      | 89/251 [00:09<00:17,  9.19it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  57\n",
      "slopesPavg = 2.57809803197\n",
      "slopesNavg = -0.221313102415\n",
      "numLines:  59\n",
      "slopesPavg = 2.70175160942\n",
      "slopesNavg = -0.190643491176\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 36%|███▋      | 91/251 [00:09<00:17,  9.12it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  57\n",
      "slopesPavg = 2.40591037014\n",
      "slopesNavg = -0.200442395962\n",
      "numLines:  64\n",
      "slopesPavg = 2.55351057885\n",
      "slopesNavg = -0.194056383396\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 37%|███▋      | 92/251 [00:10<00:17,  9.05it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  67\n",
      "slopesPavg = 2.533316501\n",
      "slopesNavg = -0.147289339414\n",
      "numLines:  57\n",
      "slopesPavg = 2.93637209546\n",
      "slopesNavg = -0.185558076121\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 37%|███▋      | 94/251 [00:10<00:17,  8.99it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  68\n",
      "slopesPavg = 1.97537414924\n",
      "slopesNavg = -0.16168810599\n",
      "numLines:  61\n",
      "slopesPavg = 1.34259844866\n",
      "slopesNavg = -0.151630894547\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 39%|███▊      | 97/251 [00:10<00:17,  8.87it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  59\n",
      "slopesPavg = 3.22428597948\n",
      "slopesNavg = -0.154331059344\n",
      "numLines:  66\n",
      "slopesPavg = 0.182118435983\n",
      "slopesNavg = -0.187552087153\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 39%|███▉      | 99/251 [00:11<00:17,  8.83it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  60\n",
      "slopesPavg = 0.0572804669052\n",
      "slopesNavg = -0.163151750203\n",
      "numLines:  50\n",
      "slopesPavg = 0.0154400287676\n",
      "slopesNavg = -0.175784490172\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 40%|████      | 101/251 [00:11<00:17,  8.82it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  52\n",
      "slopesPavg = 0.230258702424\n",
      "slopesNavg = -0.171430192277\n",
      "numLines:  49\n",
      "slopesPavg = 0.171590655557\n",
      "slopesNavg = -0.17313994059\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 41%|████      | 103/251 [00:11<00:16,  8.81it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  42\n",
      "slopesPavg = 0.181541474507\n",
      "slopesNavg = -0.215777670139\n",
      "numLines:  41\n",
      "slopesPavg = 0.219586073899\n",
      "slopesNavg = -0.198361458922\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 42%|████▏     | 105/251 [00:11<00:16,  8.81it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  36\n",
      "slopesPavg = 0.214175098643\n",
      "slopesNavg = -0.217581649505\n",
      "numLines:  39\n",
      "slopesPavg = 0.342044548574\n",
      "slopesNavg = -0.206253981653\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 43%|████▎     | 107/251 [00:12<00:16,  8.81it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  37\n",
      "slopesPavg = 0.382419661127\n",
      "slopesNavg = -0.200308347079\n",
      "numLines:  46\n",
      "slopesPavg = 0.307585212693\n",
      "slopesNavg = -0.116031348416\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 43%|████▎     | 108/251 [00:12<00:16,  8.77it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  44\n",
      "slopesPavg = 0.347317131823\n",
      "slopesNavg = -0.131687754291\n",
      "numLines:  48\n",
      "slopesPavg = 0.412106504467\n",
      "slopesNavg = -0.0964373426182\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 44%|████▍     | 111/251 [00:12<00:16,  8.73it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  52\n",
      "slopesPavg = 0.291290433193\n",
      "slopesNavg = -0.115247469349\n",
      "numLines:  34\n",
      "slopesPavg = 0.372237152303\n",
      "slopesNavg = -1.54596016573\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 45%|████▌     | 114/251 [00:13<00:15,  8.76it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  32\n",
      "slopesPavg = 0.381545330095\n",
      "slopesNavg = -0.17927821031\n",
      "numLines:  32\n",
      "slopesPavg = 0.253371479148\n",
      "slopesNavg = -0.227278160594\n",
      "numLines:  34\n",
      "slopesPavg = 0.516853681414\n",
      "slopesNavg = -0.170653837731\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 46%|████▌     | 116/251 [00:13<00:15,  8.79it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  26\n",
      "slopesPavg = 0.23420131441\n",
      "slopesNavg = -0.242239451821\n",
      "numLines:  31\n",
      "slopesPavg = 0.285468298699\n",
      "slopesNavg = -0.103868744396\n",
      "numLines: "
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 47%|████▋     | 118/251 [00:13<00:15,  8.80it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " 27\n",
      "slopesPavg = 0.240206318928\n",
      "slopesNavg = -0.107436961693\n",
      "numLines:  29\n",
      "slopesPavg = 0.267894259392\n",
      "slopesNavg = -0.145706012221\n",
      "numLines:  33\n",
      "slopesPavg = 0.360898961096\n",
      "slopesNavg = -0.165061282765\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 48%|████▊     | 121/251 [00:13<00:14,  8.81it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  35\n",
      "slopesPavg = 0.274214270178\n",
      "slopesNavg = -0.155240293311\n",
      "numLines:  32\n",
      "slopesPavg = 0.283062193911\n",
      "slopesNavg = -0.104652438525\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 49%|████▉     | 123/251 [00:14<00:14,  8.77it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  31\n",
      "slopesPavg = 0.345992775842\n",
      "slopesNavg = -0.0880793269824\n",
      "numLines:  24\n",
      "slopesPavg = 0.321606469076\n",
      "slopesNavg = -0.110829988966\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 49%|████▉     | 124/251 [00:14<00:14,  8.76it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  29\n",
      "slopesPavg = 0.411626213518\n",
      "slopesNavg = -0.198532282429\n",
      "numLines:  27\n",
      "slopesPavg = 0.11454541891\n",
      "slopesNavg = -0.264708693149\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 51%|█████     | 127/251 [00:14<00:14,  8.72it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  31\n",
      "slopesPavg = 0.191922455227\n",
      "slopesNavg = -0.198747948578\n",
      "numLines:  30\n",
      "slopesPavg = 0.209563236142\n",
      "slopesNavg = -0.257950184455\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 51%|█████     | 128/251 [00:14<00:14,  8.70it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  29\n",
      "slopesPavg = 0.223162182851\n",
      "slopesNavg = -0.224887672152\n",
      "numLines:  31\n",
      "slopesPavg = 2.7406323917\n",
      "slopesNavg = -0.158696710104\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 52%|█████▏    | 131/251 [00:15<00:13,  8.68it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  30\n",
      "slopesPavg = 0.221191743123\n",
      "slopesNavg = -0.209308567574\n",
      "numLines:  30\n",
      "slopesPavg = 0.258465775918\n",
      "slopesNavg = -0.253634228855\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 53%|█████▎    | 133/251 [00:15<00:13,  8.68it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  31\n",
      "slopesPavg = 0.227580707991\n",
      "slopesNavg = -0.194875892413\n",
      "numLines:  31\n",
      "slopesPavg = 0.237652333148\n",
      "slopesNavg = -0.199817169641\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 53%|█████▎    | 134/251 [00:15<00:13,  8.69it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  37\n",
      "slopesPavg = 0.364660696856\n",
      "slopesNavg = -0.143603045933\n",
      "numLines:  38\n",
      "slopesPavg = 0.32524783324\n",
      "slopesNavg = -0.165754140849\n",
      "numLines:  37\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 54%|█████▍    | 136/251 [00:15<00:13,  8.71it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "slopesPavg = 0.217523436287\n",
      "slopesNavg = -0.32060112618\n",
      "numLines:  28\n",
      "slopesPavg = 0.151305694989\n",
      "slopesNavg = -0.181120295072\n",
      "numLines:  30\n",
      "slopesPavg = 0.175064349203\n",
      "slopesNavg = -0.112995030992\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 55%|█████▌    | 139/251 [00:15<00:12,  8.72it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  28\n",
      "slopesPavg = 0.215892261427\n",
      "slopesNavg = -0.145420649419\n",
      "numLines:  38\n",
      "slopesPavg = 0.170090355974\n",
      "slopesNavg = -0.181067101893\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 56%|█████▌    | 141/251 [00:16<00:12,  8.68it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  40\n",
      "slopesPavg = 0.167962961839\n",
      "slopesNavg = -0.158985962547\n",
      "numLines:  43\n",
      "slopesPavg = 0.202744109064\n",
      "slopesNavg = -0.142378878264\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 57%|█████▋    | 144/251 [00:16<00:12,  8.60it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  40\n",
      "slopesPavg = 0.18306251461\n",
      "slopesNavg = -0.126831096785\n",
      "numLines:  42\n",
      "slopesPavg = 0.242872776477\n",
      "slopesNavg = -0.135182279201\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 58%|█████▊    | 146/251 [00:17<00:12,  8.58it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  48\n",
      "slopesPavg = 0.166174093782\n",
      "slopesNavg = -0.140146804631\n",
      "numLines:  55\n",
      "slopesPavg = 0.129227325881\n",
      "slopesNavg = -0.14139003332\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 59%|█████▊    | 147/251 [00:17<00:12,  8.54it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  69\n",
      "slopesPavg = 0.136048512068\n",
      "slopesNavg = -0.150391158255\n",
      "numLines:  72\n",
      "slopesPavg = 0.109883496084\n",
      "slopesNavg = -0.14615471685\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 59%|█████▉    | 149/251 [00:17<00:11,  8.51it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  66\n",
      "slopesPavg = 0.120405571692\n",
      "slopesNavg = -0.140169094105\n",
      "numLines:  53\n",
      "slopesPavg = 0.126844468166\n",
      "slopesNavg = -0.167348074049\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 60%|██████    | 151/251 [00:17<00:11,  8.43it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  41\n",
      "slopesPavg = 0.223444218096\n",
      "slopesNavg = -0.201772574317\n",
      "numLines:  35\n",
      "slopesPavg = 0.210239471789\n",
      "slopesNavg = -0.227393821455\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 61%|██████    | 153/251 [00:18<00:11,  8.40it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  42\n",
      "slopesPavg = 0.190787792096\n",
      "slopesNavg = -0.23959901017\n",
      "numLines:  42\n",
      "slopesPavg = 0.175620922639\n",
      "slopesNavg = -0.222136373005\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 62%|██████▏   | 156/251 [00:18<00:11,  8.34it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  41\n",
      "slopesPavg = 0.267639306158\n",
      "slopesNavg = -0.230696268891\n",
      "numLines:  35\n",
      "slopesPavg = 0.267032562669\n",
      "slopesNavg = -0.264542306813\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 63%|██████▎   | 157/251 [00:18<00:11,  8.32it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  32\n",
      "slopesPavg = 0.354728048095\n",
      "slopesNavg = -0.251874330687\n",
      "numLines:  29\n",
      "slopesPavg = 0.189400263272\n",
      "slopesNavg = -0.221336314001\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 64%|██████▎   | 160/251 [00:19<00:11,  8.27it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  26\n",
      "slopesPavg = 0.248074415838\n",
      "slopesNavg = -0.295422018236\n",
      "numLines:  33\n",
      "slopesPavg = 0.159256634759\n",
      "slopesNavg = -0.231546338462\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 64%|██████▍   | 161/251 [00:19<00:10,  8.25it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  25\n",
      "slopesPavg = 0.440844048019\n",
      "slopesNavg = -0.294448601212\n",
      "numLines:  31\n",
      "slopesPavg = 0.374790340367\n",
      "slopesNavg = -0.239803801152\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 65%|██████▌   | 164/251 [00:19<00:10,  8.25it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  28\n",
      "slopesPavg = 0.43088790042\n",
      "slopesNavg = -0.326660623316\n",
      "numLines:  24\n",
      "slopesPavg = 0.524795894155\n",
      "slopesNavg = -0.331295712552\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 67%|██████▋   | 167/251 [00:20<00:10,  8.27it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  25\n",
      "slopesPavg = 0.561714031847\n",
      "slopesNavg = -0.336011756114\n",
      "numLines:  25\n",
      "slopesPavg = 0.558645422015\n",
      "slopesNavg = -0.282445839615\n",
      "numLines:  31\n",
      "slopesPavg = 0.410601097559\n",
      "slopesNavg = -0.29005252022\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 67%|██████▋   | 169/251 [00:20<00:09,  8.29it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  31\n",
      "slopesPavg = 0.384773546941\n",
      "slopesNavg = -0.233354592064\n",
      "numLines:  27\n",
      "slopesPavg = 0.336893907444\n",
      "slopesNavg = -0.238367480914\n",
      "numLines:  30\n",
      "slopesPavg = 0.285208672291\n",
      "slopesNavg = -0.225560080002\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 69%|██████▊   | 172/251 [00:20<00:09,  8.32it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  24\n",
      "slopesPavg = 0.245026108912\n",
      "slopesNavg = -0.258450690081\n",
      "numLines:  29\n",
      "slopesPavg = 0.223511947676\n",
      "slopesNavg = -0.226621084615\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 69%|██████▉   | 174/251 [00:20<00:09,  8.35it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  26\n",
      "slopesPavg = 0.381855808936\n",
      "slopesNavg = -0.260464094256\n",
      "numLines:  25\n",
      "slopesPavg = 0.427451391287\n",
      "slopesNavg = -0.301795348313\n",
      "numLines:  28\n",
      "slopesPavg = 0.272294441137\n",
      "slopesNavg = -0.273384703178\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 70%|███████   | 176/251 [00:21<00:08,  8.36it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  32\n",
      "slopesPavg = 0.422258081177\n",
      "slopesNavg = -0.251614222845\n",
      "numLines:  31\n",
      "slopesPavg = 0.368867284349\n",
      "slopesNavg = -0.294427793839\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 71%|███████▏  | 179/251 [00:21<00:08,  8.36it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  33\n",
      "slopesPavg = 0.361353369351\n",
      "slopesNavg = -0.243437188045\n",
      "numLines:  33\n",
      "slopesPavg = 0.306542588974\n",
      "slopesNavg = -0.275969444992\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 72%|███████▏  | 181/251 [00:21<00:08,  8.36it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  26\n",
      "slopesPavg = 0.352762506887\n",
      "slopesNavg = -0.261743669154\n",
      "numLines:  28\n",
      "slopesPavg = 0.272332318728\n",
      "slopesNavg = -0.327424377575\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 73%|███████▎  | 183/251 [00:21<00:08,  8.39it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  32\n",
      "slopesPavg = 0.365605811785\n",
      "slopesNavg = -0.328342128575\n",
      "numLines:  30\n",
      "slopesPavg = 0.213616538218\n",
      "slopesNavg = -0.318033353342\n",
      "numLines:  32\n",
      "slopesPavg = 0.274774004743\n",
      "slopesNavg = -0.297980386156\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 75%|███████▍  | 187/251 [00:22<00:07,  8.43it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  33\n",
      "slopesPavg = 0.327369585265\n",
      "slopesNavg = -0.316994903553\n",
      "numLines:  29\n",
      "slopesPavg = 0.311168743553\n",
      "slopesNavg = -0.251556859323\n",
      "numLines:  29\n",
      "slopesPavg = 0.425889970354\n",
      "slopesNavg = -0.257886671977\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 75%|███████▌  | 189/251 [00:22<00:07,  8.44it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  34\n",
      "slopesPavg = 0.308784953269\n",
      "slopesNavg = -0.252663121005\n",
      "numLines:  34\n",
      "slopesPavg = 0.412770787707\n",
      "slopesNavg = -0.261257646911\n",
      "numLines:  37\n",
      "slopesPavg = 0.308763030418\n",
      "slopesNavg = -0.292527171834\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 77%|███████▋  | 193/251 [00:22<00:06,  8.48it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  31\n",
      "slopesPavg = 0.362252130703\n",
      "slopesNavg = -0.249465385372\n",
      "numLines:  28\n",
      "slopesPavg = 0.408944351866\n",
      "slopesNavg = -0.326729961335\n",
      "numLines:  27\n",
      "slopesPavg = 0.244100883145\n",
      "slopesNavg = -0.287972797632\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 78%|███████▊  | 195/251 [00:22<00:06,  8.50it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  30\n",
      "slopesPavg = 0.410860634425\n",
      "slopesNavg = -0.247746871295\n",
      "numLines:  32\n",
      "slopesPavg = 0.396423318988\n",
      "slopesNavg = -0.251070662545\n",
      "numLines:  28\n",
      "slopesPavg = 0.405103389009\n",
      "slopesNavg = -0.28752404199\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 79%|███████▉  | 199/251 [00:23<00:06,  8.53it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  31\n",
      "slopesPavg = 0.360526163319\n",
      "slopesNavg = -0.28587643111\n",
      "numLines:  28\n",
      "slopesPavg = 0.356508776459\n",
      "slopesNavg = -0.305622877642\n",
      "numLines:  31\n",
      "slopesPavg = 0.346133780644\n",
      "slopesNavg = -0.323699187652\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 80%|████████  | 201/251 [00:23<00:05,  8.55it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  31\n",
      "slopesPavg = 0.427011305476\n",
      "slopesNavg = -0.330438456053\n",
      "numLines:  32\n",
      "slopesPavg = 0.374335198095\n",
      "slopesNavg = -0.329687804887\n",
      "numLines:  35\n",
      "slopesPavg = 0.382896997047\n",
      "slopesNavg = -0.342613089584\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 82%|████████▏ | 205/251 [00:23<00:05,  8.60it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  28\n",
      "slopesPavg = 0.337394064168\n",
      "slopesNavg = -0.306300004941\n",
      "numLines:  28\n",
      "slopesPavg = 0.308878183893\n",
      "slopesNavg = -0.298931181562\n",
      "numLines:  30\n",
      "slopesPavg = 0.184427956445\n",
      "slopesNavg = -0.306194707438\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 82%|████████▏ | 207/251 [00:24<00:05,  8.62it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  27\n",
      "slopesPavg = 0.217507432482\n",
      "slopesNavg = -0.290137768979\n",
      "numLines:  33\n",
      "slopesPavg = 0.406256346727\n",
      "slopesNavg = -0.275763724612\n",
      "numLines:  33\n",
      "slopesPavg = 0.358970231442\n",
      "slopesNavg = -0.320237122517\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 84%|████████▍ | 211/251 [00:24<00:04,  8.65it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  27\n",
      "slopesPavg = 0.361970785751\n",
      "slopesNavg = -0.290084517946\n",
      "numLines:  28\n",
      "slopesPavg = 0.312663415899\n",
      "slopesNavg = -0.299636618779\n",
      "numLines:  31\n",
      "slopesPavg = 0.337942595594\n",
      "slopesNavg = -0.28978556968\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 85%|████████▍ | 213/251 [00:24<00:04,  8.67it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  32\n",
      "slopesPavg = 0.340270177222\n",
      "slopesNavg = -0.291746297491\n",
      "numLines:  34\n",
      "slopesPavg = 0.486368946131\n",
      "slopesNavg = -0.277740228716\n",
      "numLines:  29\n",
      "slopesPavg = 0.41668168315\n",
      "slopesNavg = -0.269787205159\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 86%|████████▋ | 217/251 [00:24<00:03,  8.69it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  26\n",
      "slopesPavg = 0.414937658157\n",
      "slopesNavg = -0.287913567023\n",
      "numLines:  29\n",
      "slopesPavg = 0.23971262093\n",
      "slopesNavg = -0.259946849176\n",
      "numLines:  31\n",
      "slopesPavg = 0.218352900269\n",
      "slopesNavg = -0.319209103954\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 87%|████████▋ | 219/251 [00:25<00:03,  8.71it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  26\n",
      "slopesPavg = 0.288549357542\n",
      "slopesNavg = -0.32200200633\n",
      "numLines:  30\n",
      "slopesPavg = 0.297598533459\n",
      "slopesNavg = -0.300156191879\n",
      "numLines:  32\n",
      "slopesPavg = 0.388714560758\n",
      "slopesNavg = -0.293903549616\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 88%|████████▊ | 221/251 [00:25<00:03,  8.72it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  28\n",
      "slopesPavg = 0.331328828308\n",
      "slopesNavg = -0.298811007794\n",
      "numLines:  27\n",
      "slopesPavg = 0.441710972134\n",
      "slopesNavg = -0.35577359938\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 89%|████████▉ | 223/251 [00:25<00:03,  8.71it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  32\n",
      "slopesPavg = 0.338204208342\n",
      "slopesNavg = -0.289994793248\n",
      "numLines:  30\n",
      "slopesPavg = 0.321597931386\n",
      "slopesNavg = -0.335349356769\n",
      "numLines:  32\n",
      "slopesPavg = 0.383225552653\n",
      "slopesNavg = -0.412377058332\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 90%|█████████ | 227/251 [00:25<00:02,  8.74it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  33\n",
      "slopesPavg = 0.438455685177\n",
      "slopesNavg = -0.309525627079\n",
      "numLines:  27\n",
      "slopesPavg = 0.384971757948\n",
      "slopesNavg = -0.325492736428\n",
      "numLines:  32\n",
      "slopesPavg = 0.308570372485\n",
      "slopesNavg = -0.297750724722\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 91%|█████████ | 229/251 [00:26<00:02,  8.75it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  27\n",
      "slopesPavg = 0.323085677584\n",
      "slopesNavg = -0.369202337443\n",
      "numLines:  28\n",
      "slopesPavg = 0.306135403491\n",
      "slopesNavg = -0.324655935075\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 93%|█████████▎| 233/251 [00:26<00:02,  8.78it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  32\n",
      "slopesPavg = 0.342901527489\n",
      "slopesNavg = -0.315782412286\n",
      "numLines:  30\n",
      "slopesPavg = 0.226815637961\n",
      "slopesNavg = -0.358040744618\n",
      "numLines:  29\n",
      "slopesPavg = 0.334463548529\n",
      "slopesNavg = -0.344048334116\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 94%|█████████▎| 235/251 [00:26<00:01,  8.77it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  31\n",
      "slopesPavg = 0.370108792622\n",
      "slopesNavg = -0.306655774588\n",
      "numLines:  31\n",
      "slopesPavg = 0.373377027508\n",
      "slopesNavg = -0.331326269628\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 94%|█████████▍| 237/251 [00:27<00:01,  8.77it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  35\n",
      "slopesPavg = 0.413253553154\n",
      "slopesNavg = -0.338659442402\n",
      "numLines:  32\n",
      "slopesPavg = 0.400014449919\n",
      "slopesNavg = -0.284366443019\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 95%|█████████▌| 239/251 [00:27<00:01,  8.79it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  26\n",
      "slopesPavg = 0.410820452687\n",
      "slopesNavg = -0.36288287293\n",
      "numLines:  26\n",
      "slopesPavg = 0.292636385124\n",
      "slopesNavg = -0.312308808798\n",
      "numLines:  33\n",
      "slopesPavg = 0.390984549754\n",
      "slopesNavg = -0.257502470035\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " 97%|█████████▋| 243/251 [00:27<00:00,  8.82it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  25\n",
      "slopesPavg = 0.35869082471\n",
      "slopesNavg = -0.316507104904\n",
      "numLines:  32\n",
      "slopesPavg = 0.32148896119\n",
      "slopesNavg = -0.320692341199\n",
      "numLines:  33\n",
      "slopesPavg = 0.266389538002\n",
      "slopesNavg = -0.269666671982\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 98%|█████████▊| 245/251 [00:27<00:00,  8.84it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  31\n",
      "slopesPavg = 0.460696608054\n",
      "slopesNavg = -0.287882950531\n",
      "numLines:  31\n",
      "slopesPavg = 0.438923141826\n",
      "slopesNavg = -0.350494143028\n",
      "numLines:  30\n",
      "slopesPavg = 0.380421522406\n",
      "slopesNavg = -0.335983203663\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 98%|█████████▊| 247/251 [00:27<00:00,  8.84it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  33\n",
      "slopesPavg = 0.435952469171\n",
      "slopesNavg = -0.298454926495\n",
      "numLines:  35\n",
      "slopesPavg = 0.370041500418\n",
      "slopesNavg = -0.266680866612\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      " 99%|█████████▉| 249/251 [00:28<00:00,  8.86it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numLines:  34\n",
      "slopesPavg = 0.272882430283\n",
      "slopesNavg = -0.322172998215\n",
      "numLines:  30\n",
      "slopesPavg = 0.336857530916\n",
      "slopesNavg = -0.29145550089\n",
      "numLines:  29\n",
      "slopesPavg = 0.342791822979\n",
      "slopesNavg = -0.313064622828\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 251/251 [00:28<00:00,  8.86it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[MoviePy] Done.\n",
      "[MoviePy] >>>> Video ready: test_videos_output/challenge_2018-08-13T22:02:11.mp4 \n",
      "\n"
     ]
    }
   ],
   "source": [
    "def ProcessVideo():\n",
    "    fileNameBase = \"solidWhiteRight\"\n",
    "    fileNameBase = \"solidYellowLeft\"\n",
    "    fileNameBase = \"challenge\"\n",
    "    fileExt = \".mp4\"\n",
    "    folderInput = \"test_videos/\"\n",
    "    folderOutput = \"test_videos_output/\"\n",
    "\n",
    "    dt = datetime.datetime.now()\n",
    "    strDT = \"_{:%Y-%m-%dT%H:%M:%S}\".format(dt)\n",
    "\n",
    "    fileNameIn  = folderInput + fileNameBase + fileExt\n",
    "    fileNameOut = folderOutput + fileNameBase + strDT + fileExt\n",
    "\n",
    "    #clip1 = VideoFileClip(fileNameIn).subclip(0,3)\n",
    "    clip1 = VideoFileClip(fileNameIn)\n",
    "    white_clip = clip1.fl_image(ProcessImage) #NOTE: this function expects color images!!\n",
    "    white_clip.write_videofile(fileNameOut, audio=False)\n",
    "    \n",
    "ProcessVideo()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's try the one with the solid white lane on the right first ..."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 289,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "\n",
       "<video width=\"960\" height=\"540\" controls>\n",
       "  <source src=\"test_videos_output/solidWhiteRight.mp4\">\n",
       "</video>\n"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "execution_count": 289,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "HTML(\"\"\"\n",
    "<video width=\"960\" height=\"540\" controls>\n",
    "  <source src=\"{0}\">\n",
    "</video>\n",
    "\"\"\".format(fileNameOut))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Improve the draw_lines() function\n",
    "\n",
    "**At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video \"P1_example.mp4\".**\n",
    "\n",
    "**Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now for the one with the solid yellow lane on the left. This one's more tricky!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "yellow_output = 'test_videos_output/solidYellowLeft.mp4'\n",
    "## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video\n",
    "## To do so add .subclip(start_second,end_second) to the end of the line below\n",
    "## Where start_second and end_second are integer values representing the start and end of the subclip\n",
    "## You may also uncomment the following line for a subclip of the first 5 seconds\n",
    "##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)\n",
    "clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')\n",
    "yellow_clip = clip2.fl_image(process_image)\n",
    "%time yellow_clip.write_videofile(yellow_output, audio=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "HTML(\"\"\"\n",
    "<video width=\"960\" height=\"540\" controls>\n",
    "  <source src=\"{0}\">\n",
    "</video>\n",
    "\"\"\".format(yellow_output))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Writeup and Submission\n",
    "\n",
    "If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "## Optional Challenge\n",
    "\n",
    "Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "challenge_output = 'test_videos_output/challenge.mp4'\n",
    "## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video\n",
    "## To do so add .subclip(start_second,end_second) to the end of the line below\n",
    "## Where start_second and end_second are integer values representing the start and end of the subclip\n",
    "## You may also uncomment the following line for a subclip of the first 5 seconds\n",
    "##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)\n",
    "clip3 = VideoFileClip('test_videos/challenge.mp4')\n",
    "challenge_clip = clip3.fl_image(process_image)\n",
    "%time challenge_clip.write_videofile(challenge_output, audio=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "HTML(\"\"\"\n",
    "<video width=\"960\" height=\"540\" controls>\n",
    "  <source src=\"{0}\">\n",
    "</video>\n",
    "\"\"\".format(challenge_output))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 385,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "General configuration for OpenCV 3.1.0 =====================================\n",
      "  Version control:               unknown\n",
      "\n",
      "  Platform:\n",
      "    Host:                        Linux 4.8.12-040812-generic x86_64\n",
      "    CMake:                       3.6.3\n",
      "    CMake generator:             Unix Makefiles\n",
      "    CMake build tool:            /usr/bin/make\n",
      "    Configuration:               Release\n",
      "\n",
      "  C/C++:\n",
      "    Built as dynamic libs?:      YES\n",
      "    C++ Compiler:                /usr/bin/c++  (ver 4.6.3)\n",
      "    C++ flags (Release):         -fsigned-char -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -msse -msse2 -mno-avx -msse3 -mno-ssse3 -mno-sse4.1 -mno-sse4.2 -ffunction-sections -fvisibility=hidden -fvisibility-inlines-hidden -fopenmp -O3 -DNDEBUG  -DNDEBUG\n",
      "    C++ flags (Debug):           -fsigned-char -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -msse -msse2 -mno-avx -msse3 -mno-ssse3 -mno-sse4.1 -mno-sse4.2 -ffunction-sections -fvisibility=hidden -fvisibility-inlines-hidden -fopenmp -g  -O0 -DDEBUG -D_DEBUG\n",
      "    C Compiler:                  /usr/bin/cc\n",
      "    C flags (Release):           -fsigned-char -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -msse -msse2 -mno-avx -msse3 -mno-ssse3 -mno-sse4.1 -mno-sse4.2 -ffunction-sections -fvisibility=hidden -fopenmp -O3 -DNDEBUG  -DNDEBUG\n",
      "    C flags (Debug):             -fsigned-char -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -msse -msse2 -mno-avx -msse3 -mno-ssse3 -mno-sse4.1 -mno-sse4.2 -ffunction-sections -fvisibility=hidden -fopenmp -g  -O0 -DDEBUG -D_DEBUG\n",
      "    Linker flags (Release):\n",
      "    Linker flags (Debug):\n",
      "    Precompiled headers:         YES\n",
      "    Extra dependencies:          gtk-x11-2.0 gdk-x11-2.0 atk-1.0 gio-2.0 pangoft2-1.0 pangocairo-1.0 gdk_pixbuf-2.0 cairo pango-1.0 freetype fontconfig gobject-2.0 gthread-2.0 glib-2.0 dl m pthread rt\n",
      "    3rdparty dependencies:       zlib libjpeg libwebp libpng libtiff libjasper IlmImf libprotobuf\n",
      "\n",
      "  OpenCV modules:\n",
      "    To be built:                 core flann imgproc ml photo reg surface_matching video dnn fuzzy imgcodecs shape videoio highgui objdetect plot superres xobjdetect xphoto bgsegm bioinspired dpm face features2d line_descriptor saliency text calib3d ccalib datasets java rgbd stereo structured_light tracking videostab xfeatures2d ximgproc aruco optflow stitching python3\n",
      "    Disabled:                    world contrib_world\n",
      "    Disabled by dependency:      -\n",
      "    Unavailable:                 cudaarithm cudabgsegm cudacodec cudafeatures2d cudafilters cudaimgproc cudalegacy cudaobjdetect cudaoptflow cudastereo cudawarping cudev python2 ts viz cvv hdf matlab sfm\n",
      "\n",
      "  GUI: \n",
      "    QT:                          NO\n",
      "    GTK+ 2.x:                    YES (ver 2.24.10)\n",
      "    GThread :                    YES (ver 2.32.4)\n",
      "    GtkGlExt:                    NO\n",
      "    OpenGL support:              NO\n",
      "    VTK support:                 NO\n",
      "\n",
      "  Media I/O: \n",
      "    ZLib:                        build (ver 1.2.8)\n",
      "    JPEG:                        build (ver 90)\n",
      "    WEBP:                        build (ver 0.3.1)\n",
      "    PNG:                         build (ver 1.6.19)\n",
      "    TIFF:                        build (ver 42 - 4.0.2)\n",
      "    JPEG 2000:                   build (ver 1.900.1)\n",
      "    OpenEXR:                     build (ver 1.7.1)\n",
      "    GDAL:                        NO\n",
      "\n",
      "  Video I/O:\n",
      "    DC1394 1.x:                  NO\n",
      "    DC1394 2.x:                  NO\n",
      "    FFMPEG:                      NO\n",
      "      codec:                     NO\n",
      "      format:                    NO\n",
      "      util:                      NO\n",
      "      swscale:                   NO\n",
      "      resample:                  NO\n",
      "      gentoo-style:              NO\n",
      "    GStreamer:                   NO\n",
      "    OpenNI:                      NO\n",
      "    OpenNI PrimeSensor Modules:  NO\n",
      "    OpenNI2:                     NO\n",
      "    PvAPI:                       NO\n",
      "    GigEVisionSDK:               NO\n",
      "    UniCap:                      NO\n",
      "    UniCap ucil:                 NO\n",
      "    V4L/V4L2:                    NO/YES\n",
      "    XIMEA:                       NO\n",
      "    Xine:                        NO\n",
      "    gPhoto2:                     NO\n",
      "\n",
      "  Parallel framework:            OpenMP\n",
      "\n",
      "  Other third-party libraries:\n",
      "    Use IPP:                     9.0.1 [9.0.1]\n",
      "         at:                     /home/travis/miniconda/conda-bld/conda_1486587071158/work/opencv-3.1.0/3rdparty/ippicv/unpack/ippicv_lnx\n",
      "    Use IPP Async:               NO\n",
      "    Use VA:                      NO\n",
      "    Use Intel VA-API/OpenCL:     NO\n",
      "    Use Eigen:                   YES (ver 3.2.7)\n",
      "    Use Cuda:                    NO\n",
      "    Use OpenCL:                  NO\n",
      "    Use custom HAL:              NO\n",
      "\n",
      "  Python 2:\n",
      "    Interpreter:                 /usr/bin/python2.7 (ver 2.7.3)\n",
      "\n",
      "  Python 3:\n",
      "    Interpreter:                 /home/cl/apps/anaconda3/envs/carnd-term1/bin/python (ver 3.5.2)\n",
      "    Libraries:                   /home/cl/apps/anaconda3/envs/carnd-term1/lib/libpython3.5m.so (ver 3.5.2)\n",
      "    numpy:                       /home/cl/apps/anaconda3/envs/carnd-term1/lib/python3.5/site-packages/numpy/core/include (ver 1.11.3)\n",
      "    packages path:               lib/python3.5/site-packages\n",
      "\n",
      "  Python (for build):            /usr/bin/python2.7\n",
      "\n",
      "  Java:\n",
      "    ant:                         /usr/bin/ant (ver 1.8.2)\n",
      "    JNI:                         /usr/lib/jvm/java-7-openjdk-amd64/include /usr/lib/jvm/java-7-openjdk-amd64/include /usr/lib/jvm/java-7-openjdk-amd64/include\n",
      "    Java wrappers:               YES\n",
      "    Java tests:                  NO\n",
      "\n",
      "  Matlab:                        Matlab not found or implicitly disabled\n",
      "\n",
      "  Tests and samples:\n",
      "    Tests:                       NO\n",
      "    Performance tests:           NO\n",
      "    C/C++ Examples:              NO\n",
      "\n",
      "  Install path:                  /home/cl/apps/anaconda3/envs/carnd-term1\n",
      "\n",
      "  cvconfig.h is in:              /home/travis/miniconda/conda-bld/conda_1486587071158/work/opencv-3.1.0/build\n",
      "-----------------------------------------------------------------\n",
      "\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(cv2.getBuildInformation())"
   ]
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  },
  "widgets": {
   "state": {},
   "version": "1.1.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
