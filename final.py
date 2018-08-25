# program works on a 6-digit postal code
import matplotlib.pyplot as plt
from keras.models import load_model
import matplotlib.image as mpimg
from PIL import Image

#load model
model = load_model('model.h5')

#image segmentation
images = []
#open test image
img = Image.open("postal.jpg")

#crop image in every 28 pixels, add each to a separate, cropped image
for x in range(0, 168, 28):
    img2 = img.crop((x, 0, x+28, 28)) # (0,0,28,28)
    img2.save(str(x) + "img.jpg")

img=mpimg.imread('0img.jpg')
img = img/ 255.0
img = img.reshape(1, 28,28,1)
images.append(img)

img=mpimg.imread('28img.jpg')
img = img/ 255.0
img = img.reshape(1, 28,28,1)

images.append(img)
img=mpimg.imread('56img.jpg')
img = img/ 255.0
img = img.reshape(1, 28,28,1)
images.append(img)

img=mpimg.imread('84img.jpg')
img = img/ 255.0
img = img.reshape(1, 28,28,1)
images.append(img)

img=mpimg.imread('112img.jpg')
img = img/ 255.0
img = img.reshape(1, 28,28,1)
images.append(img)

img=mpimg.imread('140img.jpg')
img = img/ 255.0
img = img.reshape(1, 28,28,1)
images.append(img)

prediction = []
#predict for each cropped image and append to a new array
for i in images:
    l = model.predict(i)[0]
    m = max(l)
    prediction.append(str([i for i, j in enumerate(l) if j == m][0]))
    #print("Predicted Value is: ", [i for i, j in enumerate(l) if j == m][0])
#join individual digits from the array to make the zip code
print(''.join(prediction))

# Plotting the loss function
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
