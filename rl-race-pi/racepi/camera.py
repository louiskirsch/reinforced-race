from time import sleep

from numpy import ndarray
import picamera
from picamera import PiCamera
from picamera import array


class Camera:
    def __init__(self):
        self.camera = PiCamera()
        # Flips the image (Camera is mounted upside down)
        self.camera.vflip = True
        # Set the camera to black-and-white
        self.camera.color_effects = (128, 128)
        # The camera needs some time to calibrate the light sensitivity
        sleep(5)

    def get_picture(self, width: int, height: int) -> ndarray:
        size = (width, height)

        self.camera.capture("test.jpg", resize=size)

        with picamera.array.PiRGBArray(self.camera, size=size) as output:
            self.camera.capture(output, 'rgb', resize=size)
            array = output.array
            print('Captured {}x{} image with depth {}.'.format(
                array.shape[1], array.shape[0], array.shape[2]))

        grayscale = array[:, :, 0]

        print(grayscale)

        return grayscale