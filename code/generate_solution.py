import json
import time

solution = {
    "product_code": "product_1",
    "metadata": {
        "prompt": "A photorealistic sks-brand soda can"
    },
    "results": [
        {
            "inpainted_image": "inpainted_image.jpg",
            "background_image": "background.jpg",
            "mask_image": "mask.png",
            "content_score": 0.78,
            "quality_score": 0.81,
            "volume_score": 0.75,
            "running_time": 10.5
        }
    ]
}

with open("./solution/provisional/solution.json", "w") as f:
    json.dump(solution, f, indent=4)

print("Solution JSON generated!")
