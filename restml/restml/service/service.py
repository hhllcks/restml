import connexion
import yaml
import restml

def predict(**kwargs):
    print(kwargs) 

class RESTmlService:
    def __init__(self,mdl, doFit):
        # here we will get important system information
        self.mdl = mdl      # the model
        self.doFit = doFit
    
    def predict(self):
        print("huhu")
    
    def createPredictFunction(self):
        def newFunc(**kwargs):
            for param in kwargs: 
                print(param)
        return newFunc
        
    def createService(self):
        # fitting the model if necessary
        if self.doFit == True:
            self.mdl.fit()
        
        # creating the API based on the modeldata
        self.modeldata = self.mdl.getModeldata()
        if not "description" in self.modeldata:
            self.modeldata["description"] = "No description given"
          
        restml.restml.service.service.predict = self.createPredictFunction();
              
        readYAMLFile = open('./restml/service/swagger/restml_template_file.yaml', 'r')
        yamlObject = yaml.load(readYAMLFile) 
        yamlObject["paths"]["/predict"]["post"]["parameters"] = self.mdl.getPredictParameters()
        
        writeYAMLFile = open('./restml/service/swagger/restml_template_file_with_params.yaml', 'w')
        yaml.dump(yamlObject,writeYAMLFile)  
                    
        app = connexion.App(__name__, 9090, specification_dir='swagger/')
        app.add_api('restml_template_file_with_params.yaml', 
                    arguments={
                        'title': self.mdl.getType(),
                        'description': self.modeldata['description']
                        })
        app.run()