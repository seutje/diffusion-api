import argparse
import requests
import base64
from PIL import Image
from io import BytesIO

parser = argparse.ArgumentParser(description="Call the diffusion API")
parser.add_argument("prompt", help="Text prompt for image generation")
parser.add_argument("--seed", type=int, help="Optional seed for generation")
parser.add_argument("--url", default="http://localhost:5000/generate_and_upscale", help="API URL")

args = parser.parse_args()

payload = {"prompt": args.prompt}
if args.seed is not None:
    payload["seed"] = args.seed

headers = {"Content-Type": "application/json"}

print(f"Sending request for prompt: {args.prompt} with seed: {payload.get('seed', 'random')}")
response = requests.post(args.url, json=payload, headers=headers)

if response.status_code == 200:
    result = response.json()
    if "image_base64" in result:
        img_data = base64.b64decode(result["image_base64"])
        image = Image.open(BytesIO(img_data))
        output_path = "generated_upscaled_api_image.png"
        image.save(output_path)
        print(f"Image successfully generated and saved to {output_path}")
        if "seed" in result:
            print(f"Seed used: {result['seed']}")
    else:
        print("Error: 'image_base64' not found in response.")
        print(result)
else:
    print(f"API Error: {response.status_code}")
    print(response.json())
