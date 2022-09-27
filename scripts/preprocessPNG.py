from PIL import Image
import sys
sys.path.append("..")
from classification import Classification

output_dict = {'Adialer.C': 0,'Agent.FYI': 1,'Allaple.A': 2,'Allaple.L': 3,'Alueron.gen!J': 4,'Autorun.K': 5,
 'C2LOP.P': 6,'C2LOP.gen!g': 7,'Dialplatform.B': 8,'Dontovo.A': 9, 'Fakerean': 10, 'Instantaccess': 11, 
 'Lolyda.AA1': 12, 'Lolyda.AA2': 13, 'Lolyda.AA3': 14, 'Lolyda.AT': 15, 'Malex.gen!J': 16,'Obfuscator.AD': 17,
 'Rbot!gen': 18,'Skintrim.N': 19, 'Swizzor.gen!E': 20,'Swizzor.gen!I': 21, 'VB.AT': 22,'Wintrim.BX': 23,'Yuner.A': 24}

model = None
def get_key(val,out):
    for key, value in out.items():
         if val == value:
             return key
    return "key doesn't exist"

def preprocessImage(path):
 	image = cv2.imread(path)
 	image = cv2.resize(image,(64,64))
 	image = img_to_array(image)
 	image = np.expand_dims(image,axis=0)
 	return image
classfication = Classification()
def Bayes_ResNet50_predict(image):
	image = Image.open(image)
	class_name=classfication.detect_image(image)
	print(class_name)
	return class_name