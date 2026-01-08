import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

if __name__ == "__main__":

    image_path = "/home/crslab/Grounded-Segment-Anything/images/top_down_img.png"

    fig, ax = plt.subplots()

    img = mpimg.imread(image_path)
    im_display = ax.imshow(img)

    plt.axis('off')

    while True:
        img = mpimg.imread(image_path)
        im_display.set_data(img)
        fig.canvas.draw()
        plt.pause(1)
    
    plt.show()