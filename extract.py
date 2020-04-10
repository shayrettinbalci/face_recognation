import os
import cv2
from autocrop import Cropper
from PIL import Image


def set_photo(video_base):
    video_dir = os.listdir(video_base)
    for label in video_dir:
        for i, fn in enumerate(os.listdir(os.path.join(video_base, label))):
            print(f"start collecting faces from {label}'s data")
            cap = cv2.VideoCapture(os.path.join(video_base, label, fn))
            frame_count = 0
            while True:
                # read video frame
                ret, raw_img = cap.read()
                # process every 5 frames
                if frame_count % 5 == 0 and raw_img is not None:
                    h, w, _ = raw_img.shape
                    path = os.path.join(PHOTO_BASE, label)
                    if not os.path.exists(path):
                        os.mkdir(path)
                    cv2.imwrite(f'{PHOTO_BASE}{label}/{frame_count}.jpg', raw_img)

                frame_count += 1
                if frame_count == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    break


def set_faces(photo_base, face_base, cropper):
    photo_dir = os.listdir(photo_base)
    for label in photo_dir:
        for i, fn in enumerate(os.listdir(os.path.join(photo_base, label))):
            photos = os.path.join(photo_base, label, fn)
            try:
                cropped_array = cropper.crop(photos)
            except (AttributeError, TypeError):
                pass
            if cropped_array is not None:
                faces = Image.fromarray(cropped_array)
            path = os.path.join(face_base, label)
            if not os.path.exists(path):
                os.mkdir(path)
            faces.save(f'{face_base}{label}/{i}.jpg')


if __name__ == "__main__":
    VIDEO_BASE = 'video/'
    PHOTO_BASE = 'photo/'
    FACE_BASE = 'faces/'
    c = Cropper(width=240, height=240, face_percent=75)
    print("Creating photos from videos...")
    set_photo(VIDEO_BASE)
    print("Collecting faces from photos...")
    set_faces(PHOTO_BASE, FACE_BASE, c)
    print("Done!")
