# IMAGE-TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
<br>
Import numpy module as np and pandas as pd.

### Step2:
<br>
Assign the values to variables in the program.

### Step3:
<br>
Get the values from the user appropriately.

### Step4:
<br>
Continue the program by implementing the codes of required topics.

### Step5:
<br>
Thus the program is executed in google colab.


## Program:
```python
Developed By:KRITHIGA U 
Register Number:212223240076
```
```
import numpy as np
import cv2
import matplotlib.pyplot as plt
# Read the input image
input_image = cv2.imread("exp_4.png")
# Convert from BGR to RGB so we can plot using matplotlib
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
plt.imshow(input_image)
plt.title("Original Image")  
# Disable x & y axis
plt.axis('off')
# Show the image
plt.show()
```
### Original image

![Screenshot 2025-04-11 112228](https://github.com/user-attachments/assets/6c2ea592-2da1-426c-9f96-1082c3b32d5c)

i)Image Translation
```
# Get the image shape
rows, cols, dim = input_image.shape
# [1, 0, tx] - Horizontal shift by tx;here tx=50
# [0, 1, ty] - Vertical shift by ty;here ty=50
# Transformation matrix for translation
M = np.float32([[1, 0, 50],
                [0, 1, 50],
                [0, 0, 1]]) # Fixed the missing '0' and added correct dimensions
# Apply a perspective transformation to the image
translated_image = cv2.warpPerspective(input_image, M, (cols, rows))
```
```
plt.imshow(translated_image)
plt.title("Translated Image")
# Disable x & y axis
plt.axis('off')
# Show the resulting image
plt.show()
```
### i)Image Translation

![Screenshot 2025-04-11 112312](https://github.com/user-attachments/assets/5cee8407-8105-4421-91f6-ba2c03cb6d3f)

ii) Image Scaling
```
# Define scale factors
scale_x = 5  # Scaling factor along x-axis
scale_y = 3  # Scaling factor along y-axis

# Apply scaling to the image
scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
# resize: Resize the image by scaling factors fx, fy
# INTER_LINEAR: Uses bilinear interpolation for resizing
plt.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))  # Display the scaled image
plt.title("Scaled Image")  # Set title
plt.axis('off')
```
### ii) Image Scaling

![Screenshot 2025-04-11 112318](https://github.com/user-attachments/assets/98e94cdb-8f39-419e-b6f4-8ebd64cd2503)

iii)Image shearing
```
# Define shear parameters
#shear_factor_x = 0.5  # Shear factor along x-axis
#shear_factor_y = 0.2  # Shear factor along y-axis

# Define shear matrix
shear_matrix = np.float32([[1,0.5, 0], [0.2, 1, 0]])

# Apply shear to the image
sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))
plt.imshow(cv2.cvtColor(sheared_image, cv2.COLOR_BGR2RGB))  # Display the sheared image
plt.title("Sheared Image")  # Set title
plt.axis('off')
```
### iii)Image shearing

![Screenshot 2025-04-11 112325](https://github.com/user-attachments/assets/c1c41256-ceb8-4718-b67f-962ccea92e40)

iv)Image Reflection
```
reflected_image = cv2.flip(image, 2)  # Flip the image horizontally (1 means horizontal flip)
# flip: 1 means horizontal flip, 0 would be vertical flip, -1 would flip both axes
plt.imshow(cv2.cvtColor(reflected_image, cv2.COLOR_BGR2RGB))  # Display the reflected image
plt.title("Reflected Image")  # Set title
plt.axis('off')
```

### iv)Image Reflection

![Screenshot 2025-04-11 112330](https://github.com/user-attachments/assets/0fbb4f1b-392d-4707-95c3-5ebc73c9223f)

v)Image Rotation
```
# Define rotation angle in degrees
angle = 45

# Get image height and width
height, width = image.shape[:2]

# Calculate rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

# Perform image rotation
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
```
```
plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))  # Display the rotated image
plt.title("Rotated Image")  # Set title
plt.axis('off')
```

### v)Image Rotation

![Screenshot 2025-04-11 112339](https://github.com/user-attachments/assets/71890816-1427-44b8-b6fd-fbb033819342)

vi)Image Cropping
```# Define cropping coordinates (x, y, width, height)
x = 80  # Starting x-coordinate
y = 20   # Starting y-coordinate
width = 180  # Width of the cropped region
height = 100  # Height of the cropped region

# Perform image cropping
cropped_image = image[y:y+height, x:x+width]

# Cropped images
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))  # Display the cropped image
plt.title("Cropped Image")  # Set title
plt.axis('off')
```
### vi)Image Cropping

![Screenshot 2025-04-11 112345](https://github.com/user-attachments/assets/57050023-f14c-4590-bcab-4694bd46c0b9)

## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
