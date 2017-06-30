## Project - Advanced Lane Detection

---
## Overview
The goal of this project is to write a software pipeline to identify the lane boundaries in a video. The following animation shows how the pipeline detects the lane.

![gif](docs/project_video.gif)

Check out the full video [here](https://youtu.be/ipXSKLb3tWE).

My lane detection pipeline consists of the following steps:
1. Compute camera calibration matrix and distortion coefficients.
1. Undistort raw images.
1. Use color transforms, gradients, etc., to create a thresholded binary image.
1. Transform image to "birds-eye view".
1. Detect lane pixels and fit to find the lane boundary.
1. Determine the curvature of the lane and vehicle position with respect to center.
1. Warp the detected lane boundaries back onto the original image.
1. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


### Computing camera calibration matrix
Using the chess borad images in the *camera_cal/* folder and the calibrateCamera function from opencv, we could get the following calibration matrix:  

    [[  1.05757268e+03   0.00000000e+00   6.57590638e+02]
     [  0.00000000e+00   9.70566278e+02   4.17880306e+02]
     [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]


### Undistort raw images
After we computed the calibration matrix we could use it to undistort the image.  

![png](docs/output_3_2.png)

### Perspective transform
To convert perspective of the image to "bird's eye view". We need to define the source and target points on the image. In the chess board case, we could simply take advantage of the *findChessboardCorners()* function in cv2 to automatically define the source coner points. The following code shows how I tranform the perspective of the chess board image.

![png](docs/output_5_0.png)


But in the case of lane finding, there is no corners like chessboard so we need to define our own source and target points. Assuming the ground is flat, I selected a trapozoidal shape in the image as the source points and a rectangle shape as the target points.


```python
# X,Y
img_size = (1280, 720)
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
            
# MMa 2017-06-27 Why it failed immediately when top source points are less then half of the height ??!
src = np.float32([ [img_size[0]/2-116,img_size[1]/2+150],
                    [img_size[0]/2+116,img_size[1]/2+150],
                    [img_size[0]/2+320,img_size[1]],
                    [img_size[0]/2-320,img_size[1]]])

offset_x = 300 # offset for dst points
offset_y = 400
dst = np.float32([ [offset_x, offset_y], 
                    [img_size[0]-offset_x, offset_y], 
                    [img_size[0]-offset_x, img_size[1]], 
                    [offset_x, img_size[1]]])

M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)
ploty = np.linspace(0, img_size[1]-1, img_size[1]/10 )
```

![png](docs/output_8_5.png)


There are many ways to generate a thresholded binary images. Here I first use a gaunssian filter to blur the image and smooth the noise. Then I use sobel operator to find the horizontal edge of the image as lanes are mostly vertical. I also converted the image to HLS space and extract the saturation layer. It works really well for yellow lines because they are high in saturation. I also use the magnitude and direction of gradient to find the edge of the lanes. Finally, I combine three of these approaches and generate a binary image for lane extraction. The following picture shows the result of binary image generation as well as perspective transformation. 


```python
import utils
def select(img):
    filtered = utils.gaussian_blur(img,9)
    s = utils.hls_select(filtered,thresh=(100, 255))
    
    grad_x= utils.abs_sobel_thresh(filtered, 'x',thresh=(32, 255))
    grad_y= utils.abs_sobel_thresh(filtered, 'y',thresh=(32, 255))
    
    dir_bin = utils.dir_thresh(filtered,15, thresh=(0.63, 1.))
    mag_bin = utils.mag_thresh(filtered,15, thresh=(70, 78))

    combined = np.zeros_like(mag_bin)
    combined[((grad_x == 1) ) | 
             ((mag_bin == 1) & (dir_bin == 1)) |
             s > 0] = 1
    
    return combined

for img in images:
    img_bin = select(img)
    warped = cv2.warpPerspective(img_bin, M, img_size, flags=cv2.INTER_LINEAR)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(warped,cmap='gray')
    ax2.set_title('Processed Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.show()
```

![png](docs/output_10_5.png)


I define a Line class to store the information of each frame. In each update, I add the new detection into the previous detection and fuse them together to filter out the bad detections and make the result more stable. The method I use is basically a moving average filter. I use a threshold to determine if a new detection is good or bad. If it is good then I update the value with larger weight, if not, then update with smaller weight. The threshold is determined by running the the pipeline without any filtering and calculating the mean polynomial fit coefficient values.

|Scenario | Lane |Mean polynomial coefficients |
--- | --- | ---
|**curve road**| 	L 	|	[  1.18180920e-05   7.80891282e-03   1.51736282e+00]
|        	|R 	|	[  4.46696746e-05   3.21652172e-02   7.67916775e+00]
|**straight road**| 	L 	|	[  7.57506233e-06   7.05633821e-03   3.65587183e+00]
|             	   | R 	|	[  3.13976960e-05   2.05890272e-02   5.05038335e+00]
|**different round**| 	L 	|	[  2.59277283e-05   2.85054620e-02   1.19792455e+01]
| 				    |R 	|	[  5.60560312e-05   3.47728916e-02   8.93921383e+00]
|**shadow road** 	|L 	|	[  3.52839926e-05   3.00916186e-02   1.03260225e+01]
| 				    |R 	|	[  1.39808593e-04   1.25237837e-01   3.13900015e+01]


After all the basic function blocks were built I started to assembled the pipeline. Here are the steps in the pipeline:

1. Undistort the image
2. Binarize the image
3. Transform the image to bird's eye view
4. Use search window and histogram to find the lane pixels
5. Use polynomial fitting algorithm to fit the lane line
6. Calculate the curvature
7. Unwarp the perspective and draw the information


```python
def pipeline(img):
    undistort = cv2.undistort(img, mtx, dist, None, mtx)
    img_bin = select(undistort)
    warped = cv2.warpPerspective(img_bin, M, img_size, flags=cv2.INTER_LINEAR)
    
    if l_line.best_fit == None or r_line.best_fit == None:
        left_fit,right_fit = poly_fit(warped)
    else:
        left_fit,right_fit = poly_fit_update(warped,l_line.best_fit,r_line.best_fit)
     
    l_line.add_poly_fit(left_fit)
    r_line.add_poly_fit(right_fit)
    
    
    # Generate x and y values for plotting
    left_fitx = l_line.best_fit[0]*ploty**2 + l_line.best_fit[1]*ploty + l_line.best_fit[2]
    right_fitx = r_line.best_fit[0]*ploty**2 + r_line.best_fit[1]*ploty + r_line.best_fit[2]
    
    result = draw_lane(undistort,ploty,left_fitx,right_fitx)
    result = draw_info(result,l_line.radius_of_curvature,r_line.radius_of_curvature,
                       l_line.line_base_pos,r_line.line_base_pos)
    return result

# img_bin = select(images[0])
# warped = cv2.warpPerspective(img_bin, M, img_size, flags=cv2.INTER_LINEAR)
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(img_bin,cmap='gray')
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(warped,cmap='gray')
# ax2.set_title('Processed Image', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# plt.show()
```


```python
l_line = Line()
r_line = Line()
for image in images:
    image_new = pipeline(image)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(image_new)
    ax2.set_title('Processed Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    plt.show()
    
print(l_line.mean_diff())
print(r_line.mean_diff())
```

![png](docs/output_16_4.png)
