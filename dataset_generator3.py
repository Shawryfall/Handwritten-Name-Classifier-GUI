import openai
import os
from PIL import Image
from io import BytesIO
import requests
import time

# OpenAI API key
openai.api_key = ''

# Lists of names and fonts
names = ["Emily", "Michael", "Sophia", "Jacob"]
fonts = ["Arial", "Times New Roman"]

# Target size for images
target_size = (128, 128)

# Function to generate image
def generate_image(prompt, output_dir, name, font):
    img_filename = f"trial3_{name}_{font}.png"
    img_path = os.path.join(output_dir, img_filename)
    
    if os.path.exists(img_path):
        print(f"Skipping image generation for {img_filename} (already exists)")
        return
    
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    
    time.sleep(12)  # Add a delay of 12 seconds between each request
    
    image_url = response['data'][0]['url']
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    
    # Resize image to target size and convert to RGB
    img = img.resize(target_size)
    img = img.convert('RGB')
    
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the image
    img.save(img_path)
    print(f"Saved image: {img_path}")

# Generate prompts and images
output_dir = "generated_images"
for name in names:
    # Create a subdirectory for each name
    name_dir = os.path.join(output_dir, name)
    for font in fonts:
        prompt = f"The name '{name}' written in {font} font."
        generate_image(prompt, name_dir, name, font)