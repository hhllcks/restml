import retrain

class restml_model:
    def __init__(self, root_path):
        self.root_path = root_path
        self.createModel(500)
    
    def createModel(self, training_steps):
        # call retrain script with the following parameters
        # bottleneck dir = root_path/bottlenecks
        # model dir = root_path/inception
        # training steps = training_steps
        # output graph = root_path/model/graph.pb
        # output labels = root_path/model/labels.txt
        # image_dir = root_path/data
        rt = retrain.inceptionRetrainer(image_dir = self.root_path + "/data",
                 output_graph = self.root_path + "/model/graph.pb",
                 output_labels = self.root_path + "/model/labels.txt",
                 summaries_dir = self.root_path+ "/retrain_logs",
                 model_dir = self.root_path + "/inception",
                 bottleneck_dir = self.root_path + "/bottlenecks",
                 how_many_training_steps=training_steps
            )
        
if __name__ == "__main__":
    mdl = restml_model("/tmp")