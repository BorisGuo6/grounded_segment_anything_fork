import argparse
import os
import copy

import numpy as np
import json
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from grasp_visualization import load_scene_filter, load_scene_global, get_segmented_pc, cue_world_point, get_segmented_pc_mask
# whisper
# import whisper
import os
import ast
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame, trimesh
from gtts import gTTS
import io
import speech_recognition as sr
import threading
# import rospy
# import ros_numpy


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
    # image_pil = image_path

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold,device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)
    

def speech_recognition(speech_file, model):
    # whisper
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(speech_file)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    speech_language = max(probs, key=probs.get)

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # print the recognized text
    speech_text = result.text
    return speech_text, speech_language

class AudioManager:
    def __init__(self, max_time=10):
        self.max_time = max_time

    def listen(self):
        # Initialize the recognizer
        recognizer = sr.Recognizer()

        # Use the microphone as the audio source
        with sr.Microphone() as source:
            print("Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=1)  # Reduce noise
            print("Listening...")
            audio = recognizer.listen(source)  # Capture the audio

            try:
                print("Recognizing...")
                # Use Google Web Speech API to recognize the speech
                text = recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text
            except sr.UnknownValueError:
                print("Sorry, I could not understand the audio.")
            except sr.RequestError:
                print("Sorry, the service is unavailable.")

        return None

    def record(self, file_name="my_recording.wav"):
        pass

    def speak(self, text):
        tts = gTTS(text=text, lang='en')
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

    def speak_async(self, text):
        # Run the speak method in a separate thread
        threading.Thread(target=self.speak, args=(text,), daemon=True).start()

if __name__ == "__main__":
    ## TODO upon activation, reads input from the microphone, and the NPY containing all the pointcloud
    ## identify from the RGB images, with the text prompt, the relevant area 
    ## and obtain a mask, using this, project out to a pointcloud and save it
    audio_manager = AudioManager()
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--item", type=str, help="path to config file", default='')
    parser.add_argument("--num_views", type=int, default=1)
    args = parser.parse_args()

    # cfg
    config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'  # change the path of the model config file
    grounded_checkpoint = 'groundingdino_swint_ogc.pth'  # change the path of the model
    sam_checkpoint = 'sam_vit_h_4b8939.pth'
    image_path = 'assets/demo1.jpg' ## REPLACE THIS
    output_dir = "outputs"
    box_threshold = 0.3
    text_threshold = 0.2
    iou_threshold = 0.3 
    device = 'cuda'
    secondary_mask = None

    _, global_pc, global_colours = load_scene_global(1, transforms="graspgen.txt")


    # load speech
    ## TODO REPLACE WITH LIVE INFERENCE

    # whisper_model = whisper.load_model("base")
    # speech_text, speech_language = speech_recognition(args.speech_file, whisper_model)
    # print(f"speech_text: {speech_text}")
    # print(f"speech_language: {speech_language}")

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)
    masks_all = []

    # initialize SAM
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    valid_masks = np.zeros(args.num_views)
    with open('/home/crslab/kelvin/GraspGen/item_prompt.txt', 'r') as f:
        obj_answer = ast.literal_eval(f.read())
        text_prompt = obj_answer['object'] ## REPLACE THIS
        target_secondary = obj_answer['target']

    if args.item != '':
        text_prompt = args.item

    
    for j in range(args.num_views):
        # visualize raw image
        image_pil, image = load_image("images/top_down_img.png".format(j))
        image_pil.save(os.path.join(output_dir, "raw_image_{}.jpg".format(j)))

        # run grounding dino model
        boxes_filt, scores, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, device=device
        )

        
        image = cv2.imread("images/top_down_img.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        # use NMS to handle overlapped boxes
        print(f"Before NMS: {boxes_filt.shape[0]} boxes")
        nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        print(f"After NMS: {boxes_filt.shape[0]} boxes")

        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

        try:
            masks, _, _ = predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(device),
                multimask_output = False,
            )
        except:
            print(f"unable to find the target object {text_prompt}")
            masks_all.append(None)
            continue

        if len(masks > 0):
            if scores[0] > 0.3:
                valid_masks[j] = 1
                masks_all.append(masks[0].cpu().numpy())
            else:
                masks_all.append(None)

        # draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)
        
        # plt.title(speech_text)
        plt.axis('off')
        plt.savefig(
            os.path.join(output_dir, "grounded_sam_whisper_output_{}.jpg".format(j)), 
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )


        save_mask_data(output_dir, masks, boxes_filt, pred_phrases)

        if target_secondary != None and target_secondary != "handover":
            image_pil, image = load_image("images/top_down_img.png".format(j))
            image_pil.save(os.path.join(output_dir, "raw_image_{}.jpg".format(j)))

            # run grounding dino model
            boxes_filt, scores, pred_phrases = get_grounding_output(
                model, image, target_secondary, box_threshold, text_threshold, device=device
            )

            
            image = cv2.imread("images/top_down_img.png")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)

            size = image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            # use NMS to handle overlapped boxes
            print(f"Before NMS: {boxes_filt.shape[0]} boxes")
            nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
            boxes_filt = boxes_filt[nms_idx]
            pred_phrases = [pred_phrases[idx] for idx in nms_idx]
            print(f"After NMS: {boxes_filt.shape[0]} boxes")

            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

            try:
                masks, _, _ = predictor.predict_torch(
                    point_coords = None,
                    point_labels = None,
                    boxes = transformed_boxes.to(device),
                    multimask_output = False,
                )
                if len(masks > 0):
                    if scores[0] > 0.4:
                        secondary_mask = masks[0].cpu().numpy()
                    else:
                        secondary_mask = None
            except:
                print(f"unable to find the target object {text_prompt}")
                secondary_mask = None

            
            if secondary_mask is None:
                print("No target found")
                audio_manager.speak(f"Sorry, I could not find a {target_secondary} to place the {text_prompt} into.")

            
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            for box, label in zip(boxes_filt, pred_phrases):
                show_box(box.numpy(), plt.gca(), label)
            
            # plt.title(speech_text)
            plt.axis('off')
            plt.savefig(
                os.path.join(output_dir, "masked_secondary_{}.jpg".format(j)), 
                bbox_inches="tight", dpi=300, pad_inches=0.0
            )

        break

    if 1 not in valid_masks:
        print("No valid masks found")
        ## SAY SOMETHING
        audio_manager.speak(f"Sorry, I could not find a {text_prompt}.")
        exit()

    
    # scene, all_pc, colours  = load_scene_filter(valid_masks, masks_all)
    scene, all_pc, colours = get_segmented_pc_mask(global_pc, global_colours, masks_all[0])
    cloud = trimesh.points.PointCloud(global_pc, colors=global_colours)
    # scene.add_geometry(cloud)
    # scene.show()

    # placement_coords = cue_world_point(200, 300)
    # print(placement_coords)

    np.save("/home/crslab/cehao/data/pc/pc_global.npy", global_pc)
    np.save("/home/crslab/cehao/data/pc/global_colours.npy", global_colours)
    np.save("/home/crslab/cehao/data/pc/pc_segmented.npy", all_pc)
    np.save("/home/crslab/cehao/data/pc/colours.npy", colours)

    ###############
    # Extra processing code to convert it into a compatible json file for GraspGen
    # Format: 
        #     >>> p.keys()
        # dict_keys(['object_info', 'grasp_info', 'scene_info'])
        # >>> len(p['object_info'])
        # 2
        # >>> type(p['object_info'])
        # <class 'dict'>
        # >>> p['object_info'].keys()
        # dict_keys(['pc', 'pc_color'])
        # >>> p['grasp_info'].keys()
        # dict_keys(['grasp_poses', 'grasp_conf'])
        # >>> p['scene_info'].keys()
        # dict_keys(['img_depth', 'img_color', 'pc_color'])
        # >>>

    ###############
    obj_map = {}
    obj_map['object_info'] = {}
    obj_map['object_info']['pc'] = all_pc.tolist()
    obj_map['object_info']['pc_color'] = colours.tolist()
    
    obj_map['scene_info'] = {}
    obj_map['scene_info']['pc_color'] =[global_pc.tolist()] 
    obj_map['scene_info']['img_color'] = global_colours.tolist()

    json.dump(obj_map, open("/home/crslab/kelvin/GraspGen/models/sample_data/actual_scene/file.json", 'w'))

    if target_secondary != None and target_secondary != "handover":
        _, secondary_pc, secondary_colour = get_segmented_pc_mask(global_pc, global_colours, secondary_mask)
        centroid = np.mean(secondary_pc, axis=0)
        centroid[2] += 0.15
        # scene.show()

        np.save("/home/crslab/cehao/data/grasp/secondary_tran.npy", centroid)

        cloud = trimesh.points.PointCloud(secondary_pc, colors=secondary_colour)
        scene.add_geometry(cloud)
    # scene.show()
    # if args.item != 'box':
    #     scene.show()
