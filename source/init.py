from hood import handler

if __name__ == "__main__":
    
    #
    
    dataset_path_train = "/content/deepfake-detect/dataset/train/"
    dataset_path_validation = "/content/deepfake-detect/dataset/validation/"
    #dataset_path_validation = False
    
    #
    
    buffer_size = 20 # 0 whole dataset without reloads
    
    #
    
    input_image = (500, 500)
    seq_image = 10
    
    #
    
    output_labels = 1
    
    #
    
    
    loss_target = 0.05
    
    #
    
    save_path = "/content/"
    
    #
    
    handler = handler.Handler(dataset_path_train, dataset_path_validation, buffer_size, input_image, seq_image, output_labels, loss_target, save_path)
    handler.start()
    
    #
#