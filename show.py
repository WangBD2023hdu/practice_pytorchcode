import matplotlib.pyplot as plt
figure = plt.figure()
num_of_images = 60
for imgs, target in data_train_loader:
    break

for index in range(num_of_images):
    plt.subplot(6,10,index+1)
    plt.axis('off')
    img = imgs[index, ...]
    plt.imshow(img.)
