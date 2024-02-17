import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image

class AugmentationSelector:
    def __init__(self, directory):
        self.directory = directory
        self.image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.index = 0
        self.selected_images = []

        if os.path.exists(os.path.join(directory, 'selected')):
            shutil.rmtree(os.path.join(directory, 'selected'))
        
        os.makedirs(os.path.join(directory, 'selected'))

        self.fig, self.ax = plt.subplots()
        self.ax_select = plt.axes([0.3, 0.03, 0.3, 0.075])
        self.ax_skip = plt.axes([0.61, 0.03, 0.3, 0.075])

        self.btn_select = Button(self.ax_select, 'Select for Augmentation')
        self.btn_select.on_clicked(self.select_image)
        self.btn_skip = Button(self.ax_skip, 'Do not Augment')
        self.btn_skip.on_clicked(self.skip_image)

        self.show_image()

    def show_image(self):
        img = Image.open(os.path.join(self.directory, self.image_files[self.index]))
        print(f'Showing image: {self.index + 1} of {len(self.image_files)}')
        self.ax.axis('off')
        self.ax.imshow(img)
        plt.draw()

    def select_image(self, event):
        selected_image = self.image_files[self.index]
        self.selected_images.append(selected_image)
        shutil.copy(os.path.join(self.directory, selected_image), os.path.join(self.directory, 'selected'))
        self.next_image()

    def skip_image(self, event):
        self.next_image()

    def next_image(self):
        self.index += 1
        if self.index < len(self.image_files):
            self.ax.clear()
            self.show_image()
        else:
            with open('selected_for_augmentation.txt', 'w') as f:
                for image in self.selected_images:
                    f.write(f'{image}\n')
            plt.close()
            print('Done')
    
    def show(self):
        plt.show()

if __name__ == '__main__':
    selector = AugmentationSelector('augmentation_candidates')
    selector.show()
