from ImageRetraining import InceptionRetrainer
import operator
import tensorflow as tf
from model import RESTmlModel

class MdlImageRetraining(RESTmlModel):
    
    def getType(self):
        return 'tf_img_retrain'
    
    def checkModeldata(self):
        bReturn = True
        if not "root_path" in self._modeldata:
            print("root_path missing in modeldata. Please provide a root path")
            bReturn = False
        else:
            self.root_path = self._modeldata["root_path"]
            
        if not "steps" in self._modeldata:
            print("steps missing in modeldata. Please provide the amount of steps!")
            bReturn = False
        else:
            self.steps = self._modeldata["steps"]
            
        return bReturn
    
    def predict(self, image_path):
        
        if self._check == False:
            print("Please fix modeldata first")
            return
        
        labels_path = self.root_path + "/model/labels.txt"
        graph_path = self.root_path + "/model/graph.pb"
        results = []
        
        # Read in the image_data
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        
        # Loads label file, strips off carriage return
        label_lines = [line.rstrip() for line in tf.gfile.GFile(labels_path)]
        
        # Unpersists graph from file
        with tf.gfile.FastGFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        
        with tf.Session() as sess:
            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            
            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                results.append((human_string, score))
                
            # Sort by percentage descending
            results_sorted = sorted(results, key=operator.itemgetter(1)) 
            results_sorted.reverse()
            return results_sorted
    
    def fit(self):
        # call retrain script with the following parameters
        # bottleneck dir = root_path/bottlenecks
        # model dir = root_path/inception
        # training steps = training_steps
        # output graph = root_path/model/graph.pb
        # output labels = root_path/model/labels.txt
        # image_dir = root_path/data    
                
        if self._check == False:
            print("Please fix modeldata first")
            return
        
        rt = InceptionRetrainer(image_dir = self.root_path + "/data",
                 output_graph = self.root_path + "/model/graph.pb",
                 output_labels = self.root_path + "/model/labels.txt",
                 summaries_dir = self.root_path+ "/retrain_logs",
                 model_dir = self.root_path + "/inception",
                 bottleneck_dir = self.root_path + "/bottlenecks",
                 how_many_training_steps=self.steps
            )
        rt.retrain()

    
