class Log:

    def __init__(self, save_path):
        
        self.log_path = save_path + "log.txt"
        
        #
        
        self.title()
    #
    
    def title(self):
        
        with open(self.log_path, "w") as file:
            
            file.write("epoch| train  | valid  |\n")
            file.write("------------------------\n")
        #
    #
    
    def line(self, epoch, loss_train, loss_validation):
        
        with open(self.log_path, "a") as file:
            
            file.write("{}| {} | {} |\n".format(self.format_numeric(epoch, 5, "left"), self.format_numeric(loss_train, 6), self.format_numeric(loss_validation, 6)))
        #
    #
    
    def format_numeric(self, numeric, target_size, append_side = "right", tale = " "):
        
        if(len(str(numeric)) >= target_size): return str(numeric)[0:target_size]
        
        #
        
        space = "".join([tale for _ in range(target_size - len(str(numeric)))])
        
        if(append_side == "right"):
            return str(numeric) + space
        #
        if(append_side == "left"):
            return space + str(numeric)
        #
    #
#