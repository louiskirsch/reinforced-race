from picamera import PiCamera
from time import sleep

imagePath = "image.jpg"

camera = PiCamera()
# Flips the image (Camera is mounted upside down)
camera.vflip = True
# Set the camera to black-and-white
camera.color_effects=(128,128)
# The camera needs some time to calibrate the light sensitivity
sleep(5)
camera.capture(imagePath,resize=(64,64))
