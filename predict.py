from typing import List
from cog import BasePredictor, Input, Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
import torch
from qwen_vl_utils import process_vision_info

class Predictor(BasePredictor):
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load the Qwen2-VL-2B-Instruct model.
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto"
        )
        self.model.to(self.device)
        # Load the processor and tokenizer.
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    def predict(
        self,
        image: Path = Input(
            description="Local image file (for example: file:///path/to/image.jpg). If omitted, only text is used.",
            default=None
        ),
        prompt: str = Input(
            description="Text prompt to send to the model.",
            default="Describe this image."
        )
    ) -> List[str]:
        # Prepare the chat messages for the Qwen2-VL chat interface.
        messages = []
        if image is not None:
            messages.append({
                "type": "image",
                "image": f"file://{str(image)}"
            })
        if prompt:
            messages.append({
                "type": "text",
                "text": prompt
            })
        chat_messages = [{"role": "user", "content": messages}]

        # Use the processor to prepare the text input.
        text_input = self.processor.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
        # Process the image (and video, if any) inputs.
        image_inputs, video_inputs = process_vision_info(chat_messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        # Run inference.
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        # Remove the prompt tokens from the generated output.
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

