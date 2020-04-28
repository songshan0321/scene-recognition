import torch
import torchvision
from PIL import Image
from torch.autograd import Variable as V
from torchvision import transforms as trn

centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
# model = torchvision.models.alexnet(pretrained=True).cuda()
img = Image.open('test_image/0.jpg')
input_img = V(centre_crop(img).unsqueeze(0))

model_file = 'alexnet_places365.pth.tar'
model = torchvision.models.__dict__['alexnet'](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)

input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(model, input_img, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)