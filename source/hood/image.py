import cv2, os

def process(image_path, target_size):
    
    frames = list()
    
    print(image_path)
    
    for image in os.listdir(image_path):
    
        image = cv2.imread(image_path + image, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (target_size[1], target_size[0]))
        image = image/255
        
        frames.append(image)
    #
    
    return frames
#