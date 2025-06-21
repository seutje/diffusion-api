import os
import io
import base64
from flask import Flask, request, jsonify
from PIL import Image

# --- Initial library check and installation ---
# This block attempts to install necessary libraries if they are not found.
# In a production environment, it's highly recommended to install dependencies
# using pip and requirements.txt BEFORE running the application.
try:
    from diffusers import StableDiffusionPipeline, StableDiffusionUpscalePipeline
    import torch
    import accelerate # Required for CPU offload
    import bitsandbytes # Required for 4-bit and 8-bit quantization
    # Flask is explicitly checked here
    import flask
except ImportError:
    print("Required libraries not found. Installing 'flask', 'diffusers', 'torch', 'sentencepiece', 'accelerate', and 'bitsandbytes'...")
    install_command = "pip install Flask diffusers torch sentencepiece accelerate bitsandbytes"
    print(f"Executing: {install_command}")
    os.system(install_command)
    try:
        # Re-import after installation attempt
        from diffusers import StableDiffusionPipeline, StableDiffusionUpscalePipeline
        import torch
        import accelerate
        import bitsandbytes
        import flask
        print("Libraries installed and imported successfully.")
    except ImportError as e:
        print(f"\nFailed to import all necessary libraries even after attempting installation: {e}")
        print("Please ensure your Python environment is compatible with required packages and check installation guides.")
        print("For bitsandbytes on Windows/WSL, you might need specific versions or pre-compiled wheels.")
        exit(1)
# --- End library check ---

app = Flask(__name__)

# Global variables for the pipelines
base_pipeline = None
upscale_pipeline = None
models_loaded = False
device = "cpu"
model_dtype = torch.float32

def load_models():
    """
    Loads the Stable Diffusion base and upscaling models.
    This function is called once when the Flask app starts.
    """
    global base_pipeline, upscale_pipeline, models_loaded, device, model_dtype

    if torch.cuda.is_available():
        device = "cuda"
        print("CUDA (NVIDIA GPU) found. Using GPU for accelerated inference.")
        # Check for Ampere or newer for bfloat16 (RTX 4070 is compute capability 8.6)
        if torch.cuda.get_device_capability()[0] >= 8:
            model_dtype = torch.bfloat16
            print("Using torch.bfloat16 for GPU (Ampere+ detected).")
        else:
            print("Using torch.float32 for GPU.")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        print("MPS (Apple Silicon GPU) found. Using GPU for accelerated inference.")
        model_dtype = torch.float32 # MPS typically uses float32
    else:
        print("No GPU found. Falling back to CPU. This will be significantly slower.")

    try:
        # --- Load Base Stable Diffusion Pipeline (for 512x512 generation) ---
        print(f"Attempting to load Stable Diffusion v1.5 base model with dtype={model_dtype}...")
        base_pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=model_dtype
        )
        if device != "cpu":
            base_pipeline.to(device)
        # Enable sequential CPU offload. This is crucial for memory management.
        base_pipeline.enable_sequential_cpu_offload()
        print(f"Base model loaded successfully on {device}.")

        # --- Load Upscaling Pipeline ---
        print(f"Attempting to load Stable Diffusion x4 Upscaler model with 8-bit quantization and CPU offload...")
        upscale_pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler",
            torch_dtype=model_dtype, # Use consistent dtype
            load_in_8bit=True # Upscaler is also large, using 8-bit quantization
        )
        if device != "cpu":
            upscale_pipeline.to(device)
        # Enable sequential CPU offload for the upscaler as well.
        upscale_pipeline.enable_sequential_cpu_offload()
        print(f"Upscaling model loaded successfully on {device}.")

        models_loaded = True
        print("All models initialized and ready for requests.")

    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please check your environment, driver setup, and ensure sufficient memory.")
        models_loaded = False

# --- API Endpoints ---

@app.route('/generate', methods=['POST'])
def generate_image():
    """
    API endpoint to generate a 512x512 image based on a text prompt.
    Does NOT perform an upscale step.
    Expects a JSON body with a 'prompt' key.
    Returns a base64 encoded image string.
    """
    if not models_loaded:
        return jsonify({"error": "Models are not yet loaded or failed to load. Please check server logs."}), 503

    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Invalid request: 'prompt' field is required in JSON body."}), 400

    prompt = data['prompt']
    print(f"Received request for base generation with prompt: '{prompt}'")

    try:
        # Generate the 512x512 image
        print(f"Generating 512x512 image for prompt: '{prompt}'")
        image_512 = base_pipeline(
            prompt=prompt,
            num_inference_steps=50,
            height=512,
            width=512,
        ).images[0]
        print("512x512 image generated.")

        # Convert PIL Image to Bytes and then to Base64
        buffered = io.BytesIO()
        image_512.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({"image_base64": img_str, "message": "512x512 image generated successfully!"})

    except Exception as e:
        print(f"Error during image generation: {e}")
        return jsonify({"error": f"Internal server error during image generation: {e}"}), 500

@app.route('/generate_and_upscale', methods=['POST'])
def generate_and_upscale_image():
    """
    API endpoint to generate and upscale an image based on a text prompt.
    Expects a JSON body with a 'prompt' key.
    Returns a base64 encoded image string.
    """
    if not models_loaded:
        return jsonify({"error": "Models are not yet loaded or failed to load. Please check server logs."}), 503

    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Invalid request: 'prompt' field is required in JSON body."}), 400

    prompt = data['prompt']
    print(f"Received request for generation and upscale with prompt: '{prompt}'")

    try:
        # Generate the initial 512x512 image
        print(f"Generating initial 512x512 image for prompt: '{prompt}'")
        image_512 = base_pipeline(
            prompt=prompt,
            num_inference_steps=50,
            height=512,
            width=512,
        ).images[0]
        print("Initial 512x512 image generated.")

        # Upscale the image
        print(f"Upscaling image for prompt: '{prompt}'")
        upscaled_image = upscale_pipeline(
            prompt=prompt, # Keep prompt for upscaler context if it uses it
            image=image_512, # Pass the generated 512x512 image
        ).images[0]
        print("Image upscaled.")

        # Convert PIL Image to Bytes and then to Base64
        buffered = io.BytesIO()
        upscaled_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({"image_base64": img_str, "message": "Image generated and upscaled successfully!"})

    except Exception as e:
        print(f"Error during image generation or upscaling: {e}")
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

