import os
import replicate
import requests

# Initial parameters
image_output_name = "model_result.png"

# Check API token is loaded
api_token = os.getenv("REPLICATE_API_TOKEN")
if not api_token:
    raise ValueError("The REPLICATE_API_TOKEN environment variable is not set. "
                     "Log in replicate.com and request an API_TOKEN if you don't have one. ")
else:
    print("API TOKEN_LOADED -> " + api_token)


input = {
    "image": open("./test.png", "rb"),
    "prompt": "masterpiece, best quality, highres, <lora:more_details:0.5> <lora:SDXLrender_v2.0:1>",
    "negative_prompt": "(worst quality, low quality, normal quality:2) JuggernautNegative-neg",
    "scale_factor": 2,
    "dynamic": 1.1,
    "creativity": 0.1,
    "resemblance": 2.9,
    "tiling_width": 112,
    "tiling_height": 144,
    "num_inference_steps": 25,
    "seed": 342,
    "sharpen": 0

}
output = replicate.run(
    "philz1337x/clarity-upscaler:f11a4727f8f995d2795079196ebda1bcbc641938e032154f46488fc3e760eb79",
    input=input
)

output_url = output[0]
print("Result image hosted: " + output_url)

# Download image in local
response = requests.get(output_url)
    # Check if the request is successful (code 200)
if response.status_code == 200:
    with open(image_output_name, "wb") as f:
        f.write(response.content)
    print("Image downloaded and saved as " + image_output_name)
else:
    print(f"Failed to download the image. Status code: {response.status_code}")