
Creating an AI model that detects diseases in plants
###################################################


#### 1- Prepare Data for Modeling

def convert_image_to_array(image_dir)
def image_hu_moments(image)
def convert_image_to_clor(image_dir):

### 2- Convert image to array 
### 3- divided our dataset into a training set, and a testing set: 80%, and 20%
### 4- Building and training the classifier

### 5- Model Evaluation Measure
### 6- Visualize the outcome of the classifier


Conclusion
############
our CNN model gave
loss: 0.0885 - acc: 0.9752 - val_loss: 0.2761 - val_acc: 0.9602

The model can be improved if we changed some paramaters and we can try using a different pretrained model.
The model can be improved by tuning it in future works, we can also train different models. 
For visualisation, we can optimize the algorithm's use of computational resources and reduce its complexity from O(n^2) to O(n)
We also implemented multiple images' features that can be also useful for image classification. Amongst that are:
	- HuMoments that characterize the shape in the image, those values are invariant to rotations and translation and 
	  other geometrical transformations
	- Histogram of Oriented Gradients that uses normalization, allowing better invariance to changes in illumination and shadowing 
Those features can be eventually integrated to our model in order to refine the results.
