from hood import dataset, buffer, builder, image, label, log

class Handler():
    
    def __init__(self, dataset_train_path, dataset_validation_path, buffer_size, input_image, seq_image, output_labels, loss_target, save_path):
        
        self.dataset_train = dataset.Dataset(dataset_train_path)
        self.dataset_validation = dataset.Dataset(dataset_validation_path)
        
        #
        
        self.buffer_size = buffer_size
        
        #
        
        self.input_image = input_image
        self.seq_image = seq_image
        
        #
        
        self.output_labels = output_labels
        
        #
        
        self.model = builder.create((self.input_image[0], self.input_image[1], self.seq_image, 1), self.output_labels)
        
        #
        
        self.loss_target = loss_target
        
        #
        
        self.save_path = save_path
        
        #
        
        self.log = log.Log(self.save_path)
        
        #
        
        self.epoch = 1
        
        self.loss_train_current = 1.
        self.loss_train_last = 1.
        self.loss_train_record = 1.
        
        self.loss_validation_current = 1.
        self.loss_validation_last = 1.
        self.loss_validation_record = 1.
        
        #
        
        self.buffer = buffer.Dual(self.buffer_size, self.seq_image, self.input_image, self.output_labels)
    #
    
    def start(self):
        
        while True:
            
            #
            
            self.fit()
            
            #
            
            self.evaluate()
            
            #
            
            self.checkpoint()
            
            #
            
            self.write_log()
            
            #
            
            if self.is_done(): break
            
            #
            
            self.updates()
        #
    #

    def checkpoint(self):
        
        self.model.save(self.save_path + "model.keras")
        
        if(self.loss_validation_current < self.loss_validation_record):
            
            self.model.save(self.save_path + "model_evaluated.keras")
            
            print("record!")
        #
    #
    
    def write_log(self):
        
        self.log.line(self.epoch, round(self.loss_train_current, 4), round(self.loss_validation_current, 4))
    #
    
    def is_done(self):
        
        if(self.loss_validation_current <= self.loss_target): return True
        
        #
        
        return False
    #
    
    def updates(self):
        
        if(self.loss_validation_current > self.loss_validation_last):
            
            new_lr = self.model.optimizer.learning_rate * 0.88
            
            self.model.optimizer.learning_rate = new_lr
            
            print("new learning_rate: {:.6f}".format(new_lr.numpy()))
        #
        
        if(self.loss_train_current < self.loss_train_record): self.loss_train_record = self.loss_train_current
        if(self.loss_validation_current < self.loss_validation_record): self.loss_validation_record = self.loss_validation_current
        
        #
        
        self.loss_train_last = self.loss_train_current
        self.loss_validation_last = self.loss_validation_current
        
        #
        
        self.epoch += 1
    #
    
    def buffer_load(self, data, target_buffer):
        
        for i, item in enumerate(data):
            
            x = image.process(item[0], self.input_image)
            y = label.process(item[1])
            
            target_buffer.put(i, x, y)
        #
    #

    def fit(self):
        
        print("epoch#{}".format(self.epoch))
        
        #
        
        train_len = len(self.dataset_train.data)
        
        #
        
        losses = []
        
        for start in range(0, train_len, self.buffer_size):
            
            end = min(start + self.buffer_size, train_len)
            
            #
            
            print("step {} from {}".format([start, end], train_len))
            
            #
            
            self.buffer_load(self.dataset_train.data[start: end], self.buffer)
            
            #
            
            losses.append(self.model.fit(self.buffer.x[0: end - start], self.buffer.y[0: end - start], batch_size = 1, epochs = 1).history["loss"][0])
            
            #
        #
        
        self.loss_train_current = sum(losses)/len(losses)
        
        #
    #
    
    def evaluate(self):
        
        if len(self.dataset_validation.data) == 0: return
        
        #
        
        print("-------------------------------")
        
        #
        
        evaluate_len = len(self.dataset_validation.data)
        
        #
        
        losses = []
        
        for start in range(0, evaluate_len, self.buffer_size):
            
            end = min(start + self.buffer_size, evaluate_len)
            
            #
            
            print("step {} from {}".format([start, end], evaluate_len))
            
            #
            
            self.buffer_load(self.dataset_validation.data[start: end], self.buffer)
            
            #
            
            losses.append(self.model.evaluate(self.buffer.x[0: end - start], self.buffer.y[0: end - start], batch_size = 1))
            
            #
        #
        
        self.loss_validation_current = sum(losses)/len(losses)
        
        #
        
        print(f"validation {self.loss_validation_current}")
    #
#