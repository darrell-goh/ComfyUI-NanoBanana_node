import requests
import json
import time
import base64
import io
import os
import numpy as np
import torch
import tiktoken
from PIL import Image
import hashlib
from dotenv import load_dotenv
from .chat_manager import ChatSessionManager

# Load environment variables from .env file
load_dotenv()

class NanoBananaNode:
    """
    A node for interacting with Vertex AI's generateContent API.
    Supports text and images as input.
    Returns three outputs:
      1) "Output": the text response from the LLM
      2) "image": an image tensor if generated
      3) "Stats": a string detailing tokens per second, input tokens, and output tokens
    """

    models_cache = None
    last_fetch_time = 0
    cache_duration = 3600  # Cache duration in seconds (1 hour)

    def __init__(self):
        self.chat_manager = ChatSessionManager()

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input specification for this node.
        Includes optional inputs for image data.
        """
        return {
            "required": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant."
                }),
                "user_message_box": ("STRING", {
                    "multiline": True,
                    "default": "Hello, how are you?"
                }),
                "model": (cls.fetch_nano_banana_models(),),
                "image_generation": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "round": 1,
                }),
                "chat_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "user_message_input": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING",)
    RETURN_NAMES = ("Output", "image", "Stats")

    FUNCTION = "generate_response"
    CATEGORY = "LLM"

    @classmethod
    def fetch_nano_banana_models(cls):
        """
        Fetches a list of Gemini model IDs for Vertex AI, caching them.
        Note: This requires proper authentication and project/location configuration.
        """
        current_time = time.time()
        if (cls.models_cache is None) or (current_time - cls.last_fetch_time > cls.cache_duration):
            # Get model list from environment variable or use defaults
            models_env = os.getenv("VERTEX_AI_MODELS", "gemini-3-pro-image-preview,gemini-2.5-flash-image")
            model_list = [m.strip() for m in models_env.split(",") if m.strip()]
            cls.models_cache = sorted(model_list)
            cls.last_fetch_time = current_time
        return cls.models_cache if cls.models_cache else ["gemini-3-pro-image-preview"] # Ensure it's never empty

    def validate_temperature(self, temperature):
        """
        Validates and converts temperature value to float within acceptable range.
        """
        try:
            temp = float(temperature)
            return max(0.0, min(2.0, temp))  # Clamp between 0.0 and 2.0
        except (ValueError, TypeError):
            return 1.0  # Return default if conversion fails



    def generate_response(self, system_prompt, user_message_box, model,
                         temperature, chat_mode, image_generation=False, 
                         user_message_input=None, **kwargs):
        """
        Sends a completion request to the Vertex AI generateContent endpoint.
        Handles text and optional image inputs.

        Returns three outputs:
          (1) Output: the LLM's text response
          (2) image: an image tensor if the response contains an image, else empty tensor
          (3) Stats: a string with tokens per second, prompt tokens, completion tokens
        """
        # Create empty placeholder image
        placeholder_image = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        
        # Get API key from environment variable
        api_key = os.getenv("VERTEX_AI_API_KEY", "")
        if not api_key:
             return ("Error: VERTEX_AI_API_KEY not found in environment variables.", placeholder_image, "Stats N/A")

        # Get endpoint configuration from environment variables
        endpoint_base = "https://aiplatform.googleapis.com/v1"
        use_simple_endpoint = os.getenv("VERTEX_AI_USE_SIMPLE_ENDPOINT", "false").lower() == "true"
        
        if use_simple_endpoint:
            if not os.getenv("VERTEX_AI_ENDPOINT"):
                return ("Error: VERTEX_AI_ENDPOINT must be set when using simple endpoint format.", placeholder_image, "Stats N/A")
            # Simple endpoint format for Vertex AI: {endpoint}/google/{model}:generateContent
            endpoint_base = os.getenv("VERTEX_AI_ENDPOINT")
            url = f"{endpoint_base}/google/{model}:generateContent"
        else:
            # Standard Vertex AI format (for direct API access)
            project = os.getenv("VERTEX_AI_PROJECT", "project")
            location = os.getenv("VERTEX_AI_LOCATION", "location")
            url = f"{endpoint_base}/projects/{project}/locations/{location}/publishers/*/models/{model}:generateContent"
        
        headers = {
            "api-key": f"{api_key}",
            "Content-Type": "application/json",
        }

        # Validate and convert temperature
        validated_temp = self.validate_temperature(temperature)

        # Decide whether to use user_message_input or user_message_box
        user_text = user_message_input if user_message_input is not None and user_message_input.strip() else user_message_box

        # Initialize session_path
        session_path = None
        
        # Handle chat mode
        if chat_mode:
            # Get or create a chat session
            session_path, messages = self.chat_manager.get_or_create_session(user_text, system_prompt)
            
            # Check if we need to update the system prompt (for existing sessions)
            if messages and messages[0]["role"] == "system" and messages[0]["content"] != system_prompt:
                # Update system prompt if it has changed
                messages[0]["content"] = system_prompt
        else:
            # Non-chat mode: Build the messages array, starting with a system prompt.
            messages = [
                {"role": "system", "content": system_prompt},
            ]

        # --- Build the user message content ---
        user_content_blocks = []

        # 1. Add Text part (always present)
        user_content_blocks.append({
            "type": "text",
            "text": user_text
        })

        # 2. Add Image parts (optional) - support multiple images from kwargs
        # Process all image_N inputs from kwargs
        image_keys = sorted([k for k in kwargs.keys() if k.startswith('image_')], 
                           key=lambda x: int(x.split('_')[1]))
        
        for image_key in image_keys:
            if kwargs[image_key] is not None:
                try:
                    img_str = self.image_to_base64(kwargs[image_key])
                    user_content_blocks.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_str}"
                        }
                    })
                except Exception as e:
                    print(f"Error processing {image_key}: {e}")
                    return (f"Error processing {image_key}: {e}", placeholder_image, "Stats N/A")

        # Determine message format based on content type
        # Use simple string format for text-only requests to ensure compatibility
        # Use structured format only when we have multimodal content
        has_multimodal_content = len(user_content_blocks) > 1 or any(block.get("type") != "text" for block in user_content_blocks)
        
        if has_multimodal_content:
            # Use structured format for multimodal content
            new_user_message = {
                "role": "user",
                "content": user_content_blocks
            }
        else:
            # Use simple string format for text-only requests
            new_user_message = {
                "role": "user",
                "content": user_text
            }
        
        if chat_mode:
            # In chat mode, append to existing conversation (but don't save yet - wait for response)
            messages.append(new_user_message)
        else:
            # In non-chat mode, messages array already has system prompt, just append user message
            messages.append(new_user_message)

        # --- Construct the final payload ---
        # Convert messages to Vertex AI contents format
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            if msg["role"] == "system":
                # System messages become systemInstruction in Vertex AI
                continue
            parts = []
            if isinstance(msg["content"], str):
                parts.append({"text": msg["content"]})
            elif isinstance(msg["content"], list):
                for block in msg["content"]:
                    if block.get("type") == "text":
                        parts.append({"text": block.get("text", "")})
                    elif block.get("type") == "image_url":
                        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": block["image_url"]["url"].split(",")[1]}})
            contents.append({"role": role, "parts": parts})
        
        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": validated_temp
            }
        }
        
        # Add system instruction if present
        for msg in messages:
            if msg["role"] == "system":
                data["systemInstruction"] = {
                    "parts": [{"text": msg["content"]}]
                }
                break
        
        # Only add modalities parameter if explicitly requested by user
        # This prevents "Multi-modal output is not supported" errors on text-only models
        if image_generation:
            data["generationConfig"]["candidateCount"] = 1
            print(f"Image generation enabled by user setting")

        # --- Pre-calculate text input tokens (rough estimate) ---
        # Note: Actual token count depends on the model and includes image data.
        # Rely on the API response for accurate usage stats.
        text_token_estimate = 0
        try:
            text_token_estimate = self.count_tokens(system_prompt, model) + self.count_tokens(user_text, model)
        except Exception as e:
            print(f"Warning: Token counting failed - {e}")


        # --- Make API Call and Process Response ---
        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            end_time = time.time()

            result = response.json()

            # --- Extract results and calculate stats ---
            # Vertex AI returns "candidates" instead of "choices"
            if not result.get("candidates") or not result["candidates"][0].get("content"):
                 raise ValueError("Invalid response format from API: 'candidates' or 'content' missing.")

            # Parse response for text and image content
            candidate = result["candidates"][0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            
            text_output = ""
            image_tensor = placeholder_image
            
            # Extract text and image from parts
            for part in parts:
                if "text" in part:
                    text_output += part["text"]
                elif "inlineData" in part:
                    # Handle inline image data
                    inline_data = part["inlineData"]
                    if inline_data.get("mimeType", "").startswith("image"):
                        base64_str = inline_data.get("data", "")
                        try:
                            image_tensor = self.base64_to_image(base64_str)
                            print(f"Successfully decoded image from API response")
                        except Exception as e:
                            print(f"Error decoding image: {e}")

            # Vertex AI returns usage metadata differently
            usage_metadata = result.get("usageMetadata", {})
            prompt_tokens = usage_metadata.get("promptTokenCount", text_token_estimate)
            completion_tokens = usage_metadata.get("candidatesTokenCount", 0)
            if completion_tokens == 0 and text_output: # Estimate completion tokens if API doesn't provide them
                 try:
                     completion_tokens = self.count_tokens(text_output, model)
                 except Exception as e:
                     print(f"Warning: Completion token counting failed - {e}")

            # Vertex AI doesn't provide response_ms, use client-side timing
            response_ms = None


            # Calculate tokens per second (TPS)
            tps = 0
            elapsed_time = end_time - start_time
            if response_ms is not None:
                server_elapsed_time = response_ms / 1000.0
                if server_elapsed_time > 0:
                    tps = completion_tokens / server_elapsed_time
            elif elapsed_time > 0:
                # Use client-side timing as fallback, less accurate due to network latency
                 tps = completion_tokens / elapsed_time
                 # Optional: apply a heuristic correction factor if needed, but server time is better
                 # correction_factor = 1.28 # Example factor, might need tuning
                 # tps *= correction_factor

            stats_text = (
                f"TPS: {tps:.2f}, "
                f"Prompt Tokens: {prompt_tokens}, "
                f"Completion Tokens: {completion_tokens}, "
                f"Temp: {validated_temp:.1f}, "
                f"Model: {model}"
            )

            # Save conversation in chat mode
            if chat_mode and session_path:
                # Append assistant's response to the conversation
                assistant_message = {
                    "role": "assistant",
                    "content": text_output
                }
                messages.append(assistant_message)
                
                # Save the updated conversation
                self.chat_manager.save_conversation(session_path, messages)

            return (text_output, image_tensor, stats_text)

        except requests.exceptions.RequestException as e:
            error_message = f"API Request Error: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json() # Try to get JSON error detail
                    error_message += f" | Details: {error_detail}"
                except json.JSONDecodeError:
                    error_message += f" | Status: {e.response.status_code} | Response: {e.response.text[:200]}" # Show raw text if not JSON
            else:
                 error_message += " (Network or connection issue)" # Generic network error

            return (error_message, placeholder_image, "Stats N/A due to error")
        except Exception as e: # Catch other potential errors (e.g., JSON parsing, value errors)
             return (f"Node Error: {str(e)}", placeholder_image, "Stats N/A due to error")

    @staticmethod
    def image_to_base64(image):
        """
        Converts a ComfyUI IMAGE (torch.Tensor, BHWC, float 0-1)
        into a base64-encoded PNG string.
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError("Input 'image' is not a torch.Tensor")

        # Remove batch dimension if present
        if image.ndim == 4:
            if image.shape[0] != 1:
                 print(f"Warning: Image batch size is {image.shape[0]}, using only the first image.")
            image = image.squeeze(0) # Shape HWC

        if image.ndim != 3:
             raise ValueError(f"Unexpected image dimensions: {image.shape}. Expected HWC.")

        # Convert float tensor (0-1) to numpy array (0-255, uint8)
        image_np = image.cpu().numpy()
        if image_np.dtype != np.uint8:
             if image_np.min() < 0 or image_np.max() > 1:
                  print("Warning: Image tensor values outside [0, 1] range. Clamping.")
                  image_np = np.clip(image_np, 0, 1)
             image_np = (image_np * 255).astype(np.uint8)

        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image_np, 'RGB') # Assuming RGB, adjust if needed

        # Save PIL Image to a bytes buffer as PNG
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")

        # Encode the bytes buffer to base64 string
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    @staticmethod
    def base64_to_image(base64_str: str) -> torch.Tensor:
        """
        Converts a base64 image string to a ComfyUI image tensor
        Returns tensor in [1, H, W, 3] format with values in [0, 1]
        """
        try:
            # Decode base64 string to image
            img_data = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(img_data))
            img = img.convert("RGB")

            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # Add batch dimension: [1, H, W, 3]
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            print(f"Successfully converted base64 to image tensor: {img_tensor.shape}")
            return img_tensor
            
        except Exception as e:
            print(f"Error in base64_to_image: {e}")
            # Return a small placeholder image instead of failing
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

    @staticmethod
    def count_tokens(text, model):
        """
        Count tokens for a given text using tiktoken.
        Uses model-specific encodings where possible, falls back to cl100k_base.
        Handles potential errors during encoding.
        """
        if not text or not isinstance(text, str):
            return 0

        # Strip any model modifiers like :floor, :nitro, :online
        base_model = model.split(':')[0] if ':' in model else model

        # Simplified mapping, cl100k_base is common for many recent models
        encoding_name = "cl100k_base"
        try:
            # List known models/prefixes that definitely use cl100k_base
            # Add others if known, but cl100k_base is a safe default for many
            cl100k_models = [
                "openai/gpt-4", "openai/gpt-3.5", "openai/gpt-4o",
                "anthropic/claude",
                "google/gemini",
                "meta-llama/llama-2", "meta-llama/llama-3",
                "mistralai/mistral", "mistralai/mixtral",
            ]
            # Check if the base_model or its prefix matches known cl100k models
            is_cl100k = any(base_model.startswith(prefix) for prefix in cl100k_models)

            if is_cl100k:
                 encoding_name = "cl100k_base"
            # else: # Add logic for other encodings if needed, e.g., p50k_base for older models
            #    pass # Stick with cl100k_base as default for now

            encoding = tiktoken.get_encoding(encoding_name)
            token_count = len(encoding.encode(text, disallowed_special=())) # Allow special tokens
            return token_count

        except Exception as e:
            print(f"Warning: Tiktoken error for model '{model}' (base: '{base_model}', encoding: '{encoding_name}'): {e}. Falling back to estimation.")
            # Fallback: Estimate tokens based on characters (rough approximation)
            # Average ~4 chars per token is a common heuristic
            return max(1, round(len(text) / 4))


    @classmethod
    def IS_CHANGED(cls, system_prompt, user_message_box, model,
                   temperature, chat_mode, image_generation=False, 
                   user_message_input=None, **kwargs):
        """
        Check if any input that affects the output has changed.
        Includes hashing image data.
        """
        # Hash image data if present - handle multiple images from kwargs
        image_hashes = []
        image_keys = sorted([k for k in kwargs.keys() if k.startswith('image_')], 
                           key=lambda x: int(x.split('_')[1]))
        
        for image_key in image_keys:
            if kwargs[image_key] is not None:
                image = kwargs[image_key]
                if isinstance(image, torch.Tensor):
                    try:
                        hasher = hashlib.sha256()
                        hasher.update(image.cpu().numpy().tobytes())
                        image_hashes.append(hasher.hexdigest())
                    except Exception as e:
                        print(f"Warning: Could not hash {image_key} data for IS_CHANGED: {e}")
                        image_hashes.append(f"{image_key}_hashing_error")
                else:
                    image_hashes.append(None)

        # Ensure temperature is consistently represented (e.g., as float)
        try:
            temp_float = float(temperature) if isinstance(temperature, (str, int, float)) else 1.0
            temp_float = max(0.0, min(2.0, temp_float))
        except (ValueError, TypeError):
            temp_float = 1.0


        # Combine all relevant inputs into a tuple for comparison
        # Use primitive types where possible for reliable hashing/comparison
        return (system_prompt, user_message_box, model,
                temp_float, chat_mode, image_generation, 
                tuple(image_hashes), user_message_input)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "NanoBananaNode": NanoBananaNode
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaNode": "Nano Banana (Pro) Node"
}
