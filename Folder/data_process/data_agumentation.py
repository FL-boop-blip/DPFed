import os
from PIL import Image, ImageOps
import random

# Define the dataset path and target number of images per class
dataset_path = '../Data/Raw/celeba/test/class_0'  # Path to the dataset folder
target_num_images = 2000 # Target number of images per class after augmentation

def augment_image(image):
    """Apply random augmentations to an image."""
    augmentations = [
        lambda x: x.rotate(random.randint(-30, 30)),  # Random rotation
        lambda x: ImageOps.mirror(x),  # Horizontal flip
        lambda x: ImageOps.crop(x, border=random.randint(0, 10)),  # Random crop
    ]
    augmentation = random.choice(augmentations)
    return augmentation(image)

# Group images by emotion
images_by_emotion = {}
for filename in os.listdir(dataset_path):
    if filename.endswith(('.jpg', '.png')):
        emotion = os.path.basename(dataset_path) # for rafdb
        # emotion = filename.split('_')[2].split('.')[0]
        if emotion not in images_by_emotion:
            images_by_emotion[emotion] = []
        images_by_emotion[emotion].append(filename)
        print(emotion)

# Augment images for each emotion
for emotion, images in images_by_emotion.items():
    num_images = len(images)
    print(num_images)
    
    if num_images < target_num_images:
        for i in range(target_num_images - num_images):
            original_image_path = os.path.join(dataset_path, images[i % num_images])
            with Image.open(original_image_path) as img:
                augmented_image = augment_image(img)
                # new_image_name = f'img_{num_images + i + 1}_{emotion}.jpg'
                new_image_name = f'img_{num_images + i + 1}.jpg' # for rafdb
                augmented_image.save(os.path.join(dataset_path, new_image_name))
        print(f"  → Added {target_num_images - num_images} augmented images.")
    elif num_images > target_num_images:
        imges_to_remove = images[target_num_images:]
        for img_name in imges_to_remove:
            os.remove(os.path.join(dataset_path, img_name))
        print(f"  → Removed {num_images - target_num_images} excess images.")

print("Image augmentation completed.")