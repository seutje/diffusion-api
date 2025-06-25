import os
import io
import base64
import random
import hashlib
from flask import Flask, request, jsonify
from PIL import Image

# --- Initial library check and installation ---
# This block attempts to install necessary libraries if they are not found.
# In a production environment, it's highly recommended to install dependencies
# using pip and requirements.txt BEFORE running the application.
try:
    from diffusers import DiffusionPipeline
    from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
    from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
    import torch
    import accelerate  # Required for CPU offload
    # Flask is explicitly checked here
    import flask
except ImportError:
    print("Required libraries not found. Installing 'flask', 'diffusers', 'torch', 'accelerate', 'einops', 'timm', and 'transformers'...")
    install_command = "pip install Flask diffusers torch accelerate einops timm transformers"
    print(f"Executing: {install_command}")
    os.system(install_command)
    try:
        # Re-import after installation attempt
        from diffusers import DiffusionPipeline
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
        import torch
        import accelerate
        import flask
        print("Libraries installed and imported successfully.")
    except ImportError as e:
        print(f"\nFailed to import all necessary libraries even after attempting installation: {e}")
        print("Please ensure your Python environment is compatible with required packages and check installation guides.")
        print("Please verify that all dependencies are compatible with your system.")
        exit(1)
# --- End library check ---

app = Flask(__name__)

# Global pipeline variable
omnigen_pipeline = None
models_loaded = False
# Directory to store generated images
IMAGES_DIR = "generated_images"
os.makedirs(IMAGES_DIR, exist_ok=True)

def load_models():
    """Load the OmniGen2 model once at startup."""
    global omnigen_pipeline, models_loaded

    try:
        print("Loading OmniGen2 model components...")

        model_path = "OmniGen2/OmniGen2"
        dtype = torch.bfloat16

        omnigen_pipeline = OmniGen2Pipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        omnigen_pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        omnigen_pipeline.enable_model_cpu_offload()
        print("OmniGen2 model loaded successfully.")

        models_loaded = True
        print("Model initialized and ready for requests.")

    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please check your environment, driver setup, and ensure sufficient memory.")
        models_loaded = False

# --- API Endpoints ---

@app.route('/generate', methods=['POST'])
def generate_image():
    """
    API endpoint to generate an image based on a text prompt using the OmniGen2 model.
    Expects a JSON body with a 'prompt' key.
    Returns a base64 encoded image string.
    """
    if not models_loaded:
        return jsonify({"error": "Models are not yet loaded or failed to load. Please check server logs."}), 503

    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Invalid request: 'prompt' field is required in JSON body."}), 400

    prompt = data['prompt']
    seed = data.get('seed')
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        print(f"No seed provided. Using random seed: {seed}")
    else:
        try:
            seed = int(seed)
        except (ValueError, TypeError):
            return jsonify({"error": "Seed must be an integer."}), 400
        print(f"Using provided seed: {seed}")
    print(f"Received request for base generation with prompt: '{prompt}' and seed: {seed}")

    # Check if an image with the same prompt and seed already exists
    key = f"{prompt}|{seed}"
    image_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    image_path = os.path.join(IMAGES_DIR, f"{image_hash}.png")
    if os.path.exists(image_path):
        print(f"Returning cached image for prompt '{prompt}' and seed {seed}")
        with open(image_path, "rb") as f:
            img_str = base64.b64encode(f.read()).decode("utf-8")
        return jsonify({
            "image_base64": img_str,
            "message": "Image retrieved from cache!",
            "seed": seed
        })

    try:
        # Generate the image
        print(f"Generating image for prompt: '{prompt}' with seed: {seed}")
        generated_image = omnigen_pipeline(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=50,
            max_sequence_length=1024,
            generator=torch.Generator("cpu").manual_seed(seed)
        ).images[0]
        print("Image generated.")

        # Save image to disk
        generated_image.save(image_path)

        # Convert PIL Image to Bytes and then to Base64
        buffered = io.BytesIO()
        generated_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({
            "image_base64": img_str,
            "message": "Image generated successfully!",
            "seed": seed
        })

    except Exception as e:
        print(f"Error during image generation: {e}")
        return jsonify({"error": f"Internal server error during image generation: {e}"}), 500

@app.route('/generate_and_upscale', methods=['POST'])
def generate_and_upscale_image():
    """
    API endpoint maintained for backward compatibility. Uses the OmniGen2 model to
    generate a 1024x1024 image. No separate upscaling step is performed.
    """
    if not models_loaded:
        return jsonify({"error": "Models are not yet loaded or failed to load. Please check server logs."}), 503

    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Invalid request: 'prompt' field is required in JSON body."}), 400

    prompt = data['prompt']
    seed = data.get('seed')
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        print(f"No seed provided. Using random seed: {seed}")
    else:
        try:
            seed = int(seed)
        except (ValueError, TypeError):
            return jsonify({"error": "Seed must be an integer."}), 400
        print(f"Using provided seed: {seed}")
    print(f"Received request for generation with prompt: '{prompt}' and seed: {seed}")

    # Check cache before generating
    key = f"{prompt}|{seed}"
    image_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    image_path = os.path.join(IMAGES_DIR, f"{image_hash}.png")
    if os.path.exists(image_path):
        print(f"Returning cached image for prompt '{prompt}' and seed {seed}")
        with open(image_path, "rb") as f:
            img_str = base64.b64encode(f.read()).decode("utf-8")
        return jsonify({
            "image_base64": img_str,
            "message": "Image retrieved from cache!",
            "seed": seed
        })

    try:
        # Generate the image with OmniGen2
        print(f"Generating image for prompt: '{prompt}' with seed: {seed}")
        upscaled_image = omnigen_pipeline(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=50,
            max_sequence_length=1024,
            generator=torch.Generator("cpu").manual_seed(seed)
        ).images[0]
        print("Image generated.")

        # Save image to disk
        upscaled_image.save(image_path)

        # Convert PIL Image to Bytes and then to Base64
        buffered = io.BytesIO()
        upscaled_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({
            "image_base64": img_str,
            "message": "Image generated successfully!",
            "seed": seed
        })

    except Exception as e:
        print(f"Error during image generation: {e}")
        return jsonify({"error": f"Internal server error during image processing: {e}"}), 500

@app.route('/')
def health_check():
    """Basic health check endpoint."""
    status = "loaded" if models_loaded else "loading/failed"
    return jsonify({"status": "API is running", "models": status})

if __name__ == '__main__':
    print("Starting Flask application...")
    print("Loading models in the background. This may take a while depending on your system.")
    load_models() # Load models when the app starts
    
    # Run the Flask app
    # host='0.0.0.0' makes the server accessible from outside the WSL2 container (e.g., from Windows host)
    app.run(host='0.0.0.0', port=5000)

