import os
import dotenv
import torch
from diffusers import DiffusionPipeline

# import dotenv
dotenv.load_dotenv()

# Run an image generation model locally with Hugging Face diffusers - no API key needed.
# The model is downloaded once from Hugging Face and cached (~/.cache/huggingface/hub).
# FLUX.1-schnell is a high quality model distilled to produce an image in about 4 steps.
# Note: it is a ~30GB download and needs a lot of memory (32GB+ recommended).
model = os.environ.get('LOCAL_IMAGE_MODEL', 'black-forest-labs/FLUX.1-schnell')

# Use the Apple Silicon GPU (mps), an NVIDIA GPU (cuda), or fall back to the CPU
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(device)

# FLUX is trained in bfloat16, which halves memory use on a GPU; the CPU needs full precision
dtype = torch.float32 if device == "cpu" else torch.bfloat16

print(dtype)

# FLUX.1-schnell is a gated model: accept its license at
# https://huggingface.co/black-forest-labs/FLUX.1-schnell, then put your Hugging Face
# access token in .env as HUGGING_FACE_API_KEY (or run `hf auth login`)
token = os.environ.get('HUGGING_FACE_API_KEY')

pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=dtype, token=token)
pipe = pipe.to(device)

try:
    # Create an image from the prompt
    # FLUX.1-schnell needs only ~4 steps and no classifier-free guidance
    # return_dict=False makes the pipeline return a plain tuple: (list_of_images,)
    images = pipe(
        prompt='Bunny riding on a horse, holding a lollipop, on a foggy meadow where it grows daffodils. It says "hello Muen"',    # Enter your prompt text here
        num_inference_steps=4,
        guidance_scale=0.0,
        max_sequence_length=256,  # schnell supports prompts up to 256 tokens
        width=512,
        height=512,
        return_dict=False,
    )[0]
    image = images[0]

    # Set the directory for the stored image
    image_dir = os.path.join(os.curdir, 'images')

    # If the directory doesn't exist, create it
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)

    # Initialize the image path (note the filetype should be png)
    image_path = os.path.join(image_dir, 'generated-image.png')

    # The pipeline returns a PIL image, so we can save it directly
    image.save(image_path)
    print(f"Image saved to {image_path}")

    # Display the image in the default image viewer
    image.show()

except Exception as err:
    print(f"Image generation failed: {err}")
