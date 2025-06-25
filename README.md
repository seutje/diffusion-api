# Diffusion API

This repository provides a simple Flask API for generating images using the **HiDream-I1-Fast** diffusion model. The `/generate` and `/generate_and_upscale` endpoints accept a JSON body containing a `prompt` and an optional `seed`.

Generated images are cached on disk under `generated_images/`. When a request is made with the same prompt and seed combination, the API returns the cached image instead of re-running the model.

Run the API with:

```bash
python run_api.py
```

Use the helper script to call the API:

```bash
python call_api.py "an astronaut riding a horse" --seed 42
```
