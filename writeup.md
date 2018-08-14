# **Finding Lane Lines on the Road** 
### Submission writeup: ChrisL

---

## Overview
The purpose of this project was understand and utilize some basic image processing techniques
to enable isolation of highway lane lines from images. The techinques and 'pipeline'sequence of those
techniques were outlined in the course modules.

## Implementation Notes
### 1. Overview 
The project consists of a Jupyter python notebook ```P1.ipynb```
The primary function ```ProcessImageLowlevel(imgIn, ...)``` takes an image object
numpy array and runs the lane finding pipeline (as detailed below) and returns
an image object which is a copy of original image superimposed with the lane lines
determined by the pipeline.
 
 
The function ```ProcessVideo()``` opens an input mp4 video file using
moviepy.VideoFileClip()
and calls ```ProcessImageLowlevel() ``` (via wrapper function ProcessImage())
on each video frame using moviepy.fl_image() and assembles those frames
into a new mp4 video file and saves it to disk with moviepy.write_videofile().

Additonally the are some dev/debug utilities detailed below.


### 2. Lane finding pipeline implementation

My pipeline followed the instruction closely with a few enhancements. The 
individual processing steps consisted primarily of calls to OpenCV with 
properly prepared inputs.
It consisted of these processing steps


1. **Convert to grayscale**<br/>
Using  cv2.cvtColor()<br/>
2. **Blur**<br/>
Using  cv2.GaussianBlur()<br/>
3. **Edge detection**<br/>
Using  cv2.Canny()<br/>
4. **Region of Interest Masking**<br/>
Using cv2.fillPoly()<br/>
and cv2.bitwise_and()<br/>
5. **Line segment extraction**<br/>
Using  cv2.HoughLinesP()<br/>
Selecting parameters was by trial and error until good results were achieved.
6. **Line Extraplolation**<br/>
In the case of dashed road lines there are gaps in the lane line segments.
In order to completely define the lanes it is needed to extrapolate the line segments.
This is perfomed using custom code:<br/>
The implementation loops through each Hough line and calls ExtendLineSegment()
which creates a new line that has been extended in increasing Y direction (ie toward
the bottom of the image) so that line extends to the near field.<br/>
7.**Lane Line image overlay**
Creates an image of the found and extrapolated lines using cv2.line()
then superimposes that image on the original input image using cv2.addWeighted()
to create the final output image.

### 3. Dev/Debug utilities
```PlotImageRecords()``` Plots a list of images inline in the notebook.
```ProcessSingleImage()``` Calls the image pipeline on a single image file then 
calls ```PlotImageRecords()``` to show all the intermediate pipeline images 
generated in PlotImageRecords() inline.

### 4.Implementation weaknesses, areas for improvement.<br/>
The parameters for various stages of the pipeline are tunred empirically 
for a particular input to get good resulst. The Region of interest in particular
depends on the scene unfolding. For example curves or intersections can not be handled
by a static trapezoid. Probably analysis of the segments could be performed to
determine when the are of interest is changing.

Line extrapolation is naive. A better algorithm would sort the segments according to which line
they bleonged to (by slope) then interpolate connecting segements where there are gaps
and extrapolate segments to the boundaries of the ROI.

## Environment Notes
I had some difficulty initially with environment.

I attempted to use the jupyter/conda environment carnd-term1
At notebook step

    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip
    from IPython.display import HTML

I got error ```ImportError: No module named request```

so I tried this

    conda install --name carnd-term1 requests

It found the package and gave an ominous message about downgrading some packages...
but it seemed to work... For now.

If it can't find the package in the normal repositories. 
Some poking around leads through a series of links

(https://conda.io/docs/user-guide/tasks/manage-pkgs.html#installing-packages)
(https://anaconda.org/search?q=requests)

That page will(may) provide options. Choose one that matches your
platform and seems popular and has a higher version. 
For example the link to this package info page

(https://anaconda.org/conda-forge/requests)

provides this instruction (although I think you need to have
your conda environment activated when you invoke to get it into 
that environment)

    conda install -c conda-forge requests 

### Links
[rubric](https://review.udacity.com/#!/rubrics/322/view)


[EnvConfig](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md)
