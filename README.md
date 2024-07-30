# python-clip-ml-image-description-generator
Python 3 API using OpenAI's CLIP model, generate top 5 image text descriptions

## Usage

1. Install dependencies from `requirements.txt` e.g. `pip install -r requirements.txt`
2. Start the FastAPI server with `fastapi dev main.py`, `dev` indicated running in development mode
3. By default server is exposed on port 8000 e.g. `curl 'http://127.0.0.1:8000'`, there's a Hello World test endpoint an optional name parameter to test 'curl 'http://127.0.0.1:8000/test?name=Lorena''
4. To access docs use `http://127.0.0.1:8000/docs`
5. To generate description an example curl is provided here: 
```
curl \
  -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/lorena-selfie.png" \
  http://localhost:8000/description
```

## Want to learn more?

- CLIP Huggable API Docs
- [Python example tutorial](https://crumbly.medium.com/clipdirect-5da430087ea)
- [Python example linking images with text](https://towardsdatascience.com/linking-images-and-text-with-openai-clip-abb4bdf5dbd2)