import rospy
import random
import pickle
from std_msgs.msg import String, Float64
from sensor_msgs.msg import Image
import numpy as np
from scipy.spatial.transform import Rotation as R

import pygame
from gtts import gTTS
import io

experiment_number = 0
experiment_subsection = 0

class AudioManager:
    def __init__(self, max_time=10):
        self.max_time = max_time

    def speak(self, text):
        tts = gTTS(text=text, lang='en', tld='us', slow=False)
        # Save the speech to a file-like object in memory (BytesIO)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        self.play(audio_fp)

        # Save the speech to a file
        # audio_fp.seek(0)
        # with open("output.mp3", "wb") as f:
        #     f.write(audio_fp.read())

    def play(self, audio_file):
        # Initialize the mixer
        pygame.mixer.init()

        # Load the audio file
        pygame.mixer.music.load(audio_file)

        # Play the audio
        pygame.mixer.music.play()

        # Wait until the music finishes playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(self.max_time)

        '''
        pygame.mixer.init()
        sound = pygame.mixer.Sound(audio_file)
        sound.play()
        pygame.time.wait(int(sound.get_length() * 1000))
        '''

class PPSLogger():
    def __init__(self, sub_type="Single"):
        self.tactile_pub = rospy.Publisher('/tower_command', String, queue_size=10)
        self.tactile_sub = rospy.Subscriber('/audio_sub', String, self.audio_callback, queue_size=10)
        self.moveit_pub = rospy.Publisher("/graspflow/move_to", String, queue_size=1)
        self.audio = AudioManager()
        
    def audio_callback(self, msg):
        if msg.data != 'capture' and 'grasp' not in msg.data:
            print("Received audio command: {}".format(msg.data))
            self.audio.speak(msg.data)

                
if __name__ == '__main__':
    try:
        rospy.init_node('command', anonymous=True)
        PPS_log = PPSLogger()
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass

# if __name__ == "__main__":
#     audio_manager = AudioManager()
#     audio_manager.speak("The task is finished, here is the item")
#     audio_manager.play("output.mp3")
    # audio_manager.record()
    # audio_manager.listen()