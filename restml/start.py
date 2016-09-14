import restml
#from restml.model.ImageRetraining_Model import MdlImageRetraining

modeldata = {"root_path":"/Users/hendrikhilleckes/data/flower_photos","steps":20}

# create model class
mdl = restml.model.ImageRetraining_Model.MdlImageRetraining(modeldata)
# Fit model
#mdl.fit()

# Predict
results = mdl.predict("/Users/hendrikhilleckes/data/rose.jpg")

print(results)

print(mdl.getType())
print(mdl.getModeldata())