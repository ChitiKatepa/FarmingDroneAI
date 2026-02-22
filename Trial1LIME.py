#Trial 1 using lime on the vgg16 model

import lime
import lime.lime_image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from skimage.segmentation import mark_boundaries
from PIL import Image  
import matplotlib.pyplot as plt
import numpy as np

model = VGG16(weights='imagenet')

explainer = lime.lime_image.LimeImageExplainer()

#resizing bc vgg16 uses a 224x224 pixel input
img = np.array(Image.open('any one of the images from the file or my own/internet').resize((224, 224)))
img = preprocess_input(img)

#will actually show how the model picked out the diseased zone
explanation = explainer.explain_instance(img, model.predict, top_labels=5, hide_color=0, num_samples=1000)

#superimposes the explanation onto the image
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
plt.axis('off')

plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
