import restml
import os.path
from werkzeug.utils import secure_filename
#from restml.model.ImageRetraining_Model import MdlImageRetraining

modeldata = {"root_path":"/Users/hendrikhilleckes/data/flower_photos","steps":20}

mdl = restml.model.ImageRetraining_Model.MdlImageRetraining(modeldata)

def coolPredictionFunction(imgpath):
    # check if file exists and is .jpg
    if imgpath.endswith('.jpg') & os.path.isfile(imgpath):
        return mdl.predict(imgpath)
    else:
        return ("Please provide a valid .jpg file",400)
    
def coolPredictionFunctionFile(img):
    # save img in tmp folder
    filename = secure_filename(img.filename)
    if filename.endswith('.jpg'):
        target_filename = os.path.join("/tmp/", filename)
        img.save(target_filename)
        return mdl.predict(target_filename)
    else:
        return ("Please provide a valid .jpg file",400)
    
def myPredictFunction(): pass
    
service = restml.service.service.RESTmlService(mdl, False)

service.createService()
