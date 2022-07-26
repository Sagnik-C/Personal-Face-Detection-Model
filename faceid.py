#Layout imports
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

#ux components imports
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

#Other kivy dependencies
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

#Other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

#App Layout
class camApp(App):

    def build(self):
        #Components
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1, .1))
        self.verification_label = Label(text="Verification uninitiated", size_hint=(1,.1))

        #Adding components to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        #Load tensorflow keras model
        self.model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist':L1Dist})

        #Capture webcam
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout
    
    def update(self, *args):
        #Reading Frames
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]

        #Flip image horizontally & convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture
    
    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (100,100))
        img = img/255.0
        return img
    
    def verify(self, *args):

        #initialize thresholds
        detection_threshold = 0.8
        verification_threshold = 0.6

        #Capture real-time feed
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)


        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            val_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
            inp_reshaped = np.expand_dims(input_img, axis=0)
            val_reshaped = np.expand_dims(val_img, axis=0)
            result = self.model.predict(tuple((inp_reshaped, val_reshaped)))
            results.append(result)
        #Validating with the thresholds
        detection = np.sum(np.array(results)>detection_threshold)
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        verified = verification > verification_threshold

        #Update Verification Label
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

        Logger.info(results)
        Logger.info(verification)

        
        return results, verified
        



if __name__=='__main__':
    camApp().run()