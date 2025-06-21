import requests
import base64
from PIL import Image
from io import BytesIO

api_url = "http://localhost:5000/generate_and_upscale"
prompt = "A vibrant coral reef with colorful fish and sunlight streaming through the water, highly detailed, underwater photography."

payload = {"prompt": prompt}
headers = {"Content-Type": "application/json"}

print(f"Sending request for prompt: {prompt}")
response = requests.post(api_url, json=payload, headers=headers)

if response.status_code == 200:
    result = response.json()
    if "image_base64" in result:
        img_data = base64.b64decode(result["image_base64"])
        image = Image.open(BytesIO(img_data))
        output_path = "generated_upscaled_api_image.png"
        image.save(output_path)
        print(f"Image successfully generated and saved to {output_path}")
    else:
        print("Error: 'image_base64' not found in response.")
        print(result)
else:
    print(f"API Error: {response.status_code}")
    print(response.json())
