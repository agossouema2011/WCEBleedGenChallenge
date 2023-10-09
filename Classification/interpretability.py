# This is the CAMs interpretability code
import torch

from model import build_model

batch_size = 20

import os
from pathlib import Path

from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image

from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask

#DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

checkpoint = torch.load(os.path.join('outputs', 'model.pth'))
    
# Load the model.
model = build_model(
                    num_classes=len(CLASS_NAMES)
                    ).to(DEVICE)
    
model.load_state_dict(checkpoint['model_state_dict']) # load the saved model


# Set your CAM extractor
from torchcam.methods import SmoothGradCAMpp
cam_extractor = SmoothGradCAMpp(model)


# Get your own image input here
image_path = "input/Test/TestDataset2/A0461.png"
img = read_image(image_path)
# Preprocess it for your chosen model
input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

#with SmoothGradCAMpp(model) as cam_extractor:
# Preprocess your data and feed it to the model
out = model(input_tensor.unsqueeze(0))
# Retrieve the CAM by passing the class index and the model output
activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

# Resize the CAM and overlay it
result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
# Display it
    
the_name=Path(image_path).stem # get the name of the file
plt.axis('off')
plt.title(label=the_name,
             fontsize=20, 
              color="green")
plt.imshow(result);plt.axis('off'); plt.tight_layout(); plt.show()
the_name="CAM_TestDataset_"+the_name
plt.savefig('outputs/'+the_name+'.png') # save it
 
