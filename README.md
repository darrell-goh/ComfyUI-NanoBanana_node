# ComfyUI Nano Banana Node

A custom node for ComfyUI that provides seamless integration with Google Vertex AI REST API format, featuring dynamic image inputs and chat mode functionality.

This project is derived from the design and structural patterns of the [ComfyUI-Openrouter_node](https://github.com/gabe-init/ComfyUI-Openrouter_node) project by **@gabe-init**, from which elements such as dynamic image input handling and general node architecture principles were adapted. The original work is licensed under the MIT Licence, and its licence is included in this repository in accordance with its terms.

This project is an independent implementation focused solely on Nano Banana and Vertex AI integration and does not provide OpenRouter functionality.

## Features

- **Dynamic Image Inputs** - Automatically adds new image input slots as you connect images (up to 10)
- **Image Generation Support** - Generate images with Gemini/Nano Banana models through Vertex API
- **Chat Mode** - Maintain conversation context across multiple messages with automatic session management
- **Multi-Image Support** - Send multiple images in a single request to supported Gemini models
- **Flexible Configuration** - Use environment variables for easy endpoint and model configuration

## Installation

1. Clone this repository into your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/gabe-init/ComfyUI-Openrouter_node
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Copy `.env.template` to `.env` and configure:
```bash
cp .env.template .env
```

4. Edit `.env` with your configuration:

   **For own proxy (simple endpoint):**
   ```env
   VERTEX_AI_API_KEY=your_api_key_here
   VERTEX_AI_USE_SIMPLE_ENDPOINT=true
   VERTEX_AI_ENDPOINT=https://your-nano-banana-endpoint.com
   VERTEX_AI_MODELS=gemini-2.5-flash-image,gemini-3-pro-image-preview
   ```

   **For direct Vertex AI access (standard endpoint):**
   ```env
   VERTEX_AI_API_KEY=your_vertex_ai_api_key
   VERTEX_AI_USE_SIMPLE_ENDPOINT=false
   VERTEX_AI_PROJECT=your-gcp-project-id
   VERTEX_AI_LOCATION=your-endpoint-location
   VERTEX_AI_MODELS=gemini-2.5-flash-image,gemini-3-pro-image-preview
   ```

5. Restart ComfyUI

## Usage

The Nano Banana node provides a simple interface to interact with Gemini models through the Vertex AI proxy service.

### Inputs

#### Required Inputs:

- **system_prompt**: The system prompt that sets the behavior of the LLM
- **user_message_box**: The user message to send to the model
- **model**: The Gemini model to use (configured via environment variables)
- **image_generation**: Enable image generation for supported models
- **temperature**: Controls the randomness of output (0.0 to 2.0)
- **chat_mode**: Enable conversation mode to maintain context across messages

#### Optional Inputs:

- **image_1** through **image_10**: Dynamic image inputs that automatically appear as you connect images
- **user_message_input**: Alternative input for the user message, useful for connecting to other nodes

### Outputs:

- **Output**: The text response from the model
- **image**: An image tensor if the response contains a generated image
- **Stats**: Token usage statistics (TPS, prompt tokens, completion tokens, temperature, model)

## Examples

### Image Generation

1. Set a system prompt
2. Enter a generation prompt (e.g., "Generate a beautiful sunset over mountains")
3. Enable the "image_generation" option
4. Select an image-capable model (e.g., "gemini-2.5-flash-image")
5. Run the workflow
6. The generated image will appear in the "image" output

### Image Edit

1. Connect an image output from another node to the "image_1" input
2. Set a system prompt describing the edit style
3. Enter an edit prompt (e.g., "Change the sky to purple and add clouds")
4. Enable the "image_generation" option
5. Select an image-capable model
6. Run the workflow
7. The edited image will appear in the "image" output

### Image Combine

1. Connect multiple image outputs to "image_1", "image_2", etc.
2. Set a system prompt describing how to combine the images
3. Enter a combination prompt (e.g., "Merge these images into a single cohesive composition")
4. Enable the "image_generation" option
5. Select an image-capable model
6. Run the workflow
7. The combined image will appear in the "image" output

### Chat Mode

Chat Mode allows you to maintain conversation context across multiple messages:

1. **Enable Chat Mode**: Toggle "chat_mode" to True
2. **Automatic Session Management**: 
   - Sessions are created automatically when you start a conversation
   - Continue the same session if sending messages within 1 hour
   - After 1 hour of inactivity, a new session is created
3. **Session Storage**: Conversations are stored in a `chats` folder with timestamps

#### Managing Chat Sessions:

Use the included `manage_chats.py` utility:

```bash
# List all chat sessions
python manage_chats.py list

# View a specific session
python manage_chats.py view session_name

# Export a session to different formats (json, txt, md)
python manage_chats.py export session_name -f md -o output.md

# Clean up old sessions
python manage_chats.py clean -d 30
```

## Configuration

### Environment Variables

- `VERTEX_AI_API_KEY`: Your Vertex API key (required)
- `VERTEX_AI_USE_SIMPLE_ENDPOINT`: Set to `true` for own proxy
- `VERTEX_AI_ENDPOINT`: Your Vertex AI endpoint URL (only activated for simple endpoint)
- `VERTEX_AI_MODELS`: Comma-separated list of available models
- `VERTEX_AI_PROJECT`: Google Cloud project ID (only for direct Vertex AI access)
- `VERTEX_AI_LOCATION`: Google Cloud region (only for direct Vertex AI access)

### Supported Models

Configure your available models in the `.env` file. Common Gemini models include:
- `gemini-2.5-flash-image` - Fast image generation and understanding
- `gemini-3-pro-image-preview` - Advanced image capabilities
- Other Gemini models as supported by your Vertex AI instance

## Troubleshooting

- **Connection errors**: Verify your `VERTEX_AI_ENDPOINT` and `VERTEX_AI_API_KEY` in `.env`
- **Model not available**: Check that the model is included in `VERTEX_AI_MODELS`
- **Image generation not working**: Ensure you've enabled `image_generation` and are using a compatible model
- **Chat mode issues**: Check that the `chats` folder has write permissions

## License

MIT License - See LICENSE file for details

## Credits

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - The amazing ComfyUI framework
- @gabe-init – Original OpenRouter node design that inspired the dynamic input and node structure