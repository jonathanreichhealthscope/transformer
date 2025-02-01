import requests
import os

def download_file(url, filename):
    print(f"Downloading {filename}...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded {filename}")
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")

def main():
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

    # Base URLs for GPT-2 files
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main"
    
    # Files to download
    files = {
        "gpt2.vocab": f"{base_url}/encoder.json",
        "gpt2.merges": f"{base_url}/vocab.bpe",
        "gpt2.vocab.json": f"{base_url}/encoder.json",
        "gpt2.merges.json": f"{base_url}/vocab.bpe"
    }

    # Download each file
    for filename, url in files.items():
        download_file(url, filename)

if __name__ == "__main__":
    main() 