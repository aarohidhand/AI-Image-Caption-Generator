from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

image = Image.open("egpics/spizzy.jpg")

text = 'a photograph of'   ## input text for conditional caption
inputs = processor(image, text, return_tensors='pt')

outputs = model.generate(**inputs)
caption = processor.decode(outputs[0],skip_special_tokens=True)

print ('Generated Caption:', caption)
