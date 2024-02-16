import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from shiprsimagenet.annotations import LabeledImage


def visualize_image(labeled_image: LabeledImage):
    # Load the image
    img = Image.open(labeled_image.file_path)

    # Draw each bounding box
    imgDraw = ImageDraw.Draw(img)
    for obj in labeled_image.objects:
        rotated_box = obj.rotated_bndbox
        imgDraw.polygon([(rotated_box.x1, rotated_box.y1), (rotated_box.x2, rotated_box.y2), (rotated_box.x3, rotated_box.y3), (rotated_box.x4, rotated_box.y4)], outline='lime', width=5)

    # Display the image
    plt.imshow(img)
    plt.axis('off')
    plt.show()