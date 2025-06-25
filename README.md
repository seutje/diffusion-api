# Diffusion API

This repository provides a simple Flask API for generating images using the **HiDream-I1-Fast** diffusion model. The `/generate` and `/generate_and_upscale` endpoints accept a JSON body containing a `prompt` and an optional `seed`.

If running with PyTorch 2.0 or later, the UNet is compiled with `torch.compile` when the API starts for faster inference.

Generated images are cached on disk under `generated_images/`. When a request is made with the same prompt and seed combination, the API returns the cached image instead of re-running the model.

Run the API with:

```bash
python run_api.py
```

To load the model from a custom location, set the `HIDREAM_REPO` environment
variable to a local directory or alternate Hugging Face repository before
starting the server.

Use the helper script to call the API:

```bash
python call_api.py "an astronaut riding a horse" --seed 42
```
