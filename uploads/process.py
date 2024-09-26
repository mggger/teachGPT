import re
import base64
import requests
import markdown
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

class MarkdownImageCaptionReplacer:
    def __init__(self, config):
        self.config = config
        self.session = requests.Session()

    def get_image_description(self, image_url):
        # Download the image
        response = self.session.get(image_url)
        if response.status_code != 200:
            return "[Failed to load image]"

        image_data = response.content

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config['openai_api_key']}"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Briefly describe this picture within 30 words"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = self.session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return "Failed to get image description."

    def process_markdown(self, markdown_content):
        # Convert Markdown to HTML
        html_content = markdown.markdown(markdown_content)

        # Process HTML content
        soup = BeautifulSoup(html_content, 'html.parser')
        img_tags = soup.find_all('img')

        image_urls = [img['src'] for img in img_tags if img.get('src') and img['src'].startswith(self.config['image_domain'])]
        for url in image_urls:
            description = self.get_image_description(url)
            for img in soup.find_all('img', src=url):
                img.replace_with(f"![{description}]({url})")

        # Convert processed HTML back to Markdown-like format
        processed_content = str(soup)
        # Remove HTML tags except for line breaks
        processed_content = re.sub(r'<(?!br\s*/|p>).*?>', '', processed_content)
        # Replace <br/> with newlines
        processed_content = re.sub(r'<br\s*/?>', '\n', processed_content)
        # Replace <p> and </p> with newlines
        processed_content = re.sub(r'</?p>', '\n\n', processed_content)
        # Remove any extra newlines
        processed_content = re.sub(r'\n{3,}', '\n\n', processed_content)

        return processed_content.strip()

    def replace_image_captions(self, input_text):
        # Find all occurrences of ![](image_domain/...)
        pattern = rf'!\[\]\(({re.escape(self.config["image_domain"])}[^\)]+)\)'
        image_urls = re.findall(pattern, input_text)

        # Process images in parallel
        with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
            future_to_url = {executor.submit(self.get_image_description, url): url for url in image_urls}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    description = future.result()
                    input_text = input_text.replace(f'![]({url})', f'![{description}]({url})')
                except Exception as exc:
                    print(f'{url} generated an exception: {exc}')

        return input_text

    def remove_empty_headers(self, text):
        # Remove lines that only contain '#' characters and optional spaces
        lines = text.split('\n')
        non_empty_lines = []
        for line in lines:
            if not re.match(r'^\s*#+\s*$', line):
                non_empty_lines.append(line)
        return '\n'.join(non_empty_lines)

    def split_into_chunks(self, text):
        # Split the text into chunks based on headers
        chunks = []
        current_title = "None"
        current_content = []
        lines = text.split('\n')

        for line in lines:
            if line.strip().startswith('#'):
                if current_content:
                    chunks.append((current_title, '\n'.join(current_content)))
                current_title = line.strip()
                current_content = []
            else:
                current_content.append(line)

        # Add the last chunk if it exists
        if current_title and current_content:
            chunks.append((current_title, '\n'.join(current_content)))

        # Filter out chunks with empty content
        chunks = [(title.replace("#", "").strip(), content) for title, content in chunks if content.strip()]

        return chunks


def create_safe_filename(original_filename, title):
    # Remove file extension and convert to lowercase
    base_name = os.path.splitext(original_filename)[0].lower()

    # Convert title to lowercase and replace non-alphanumeric characters with underscore
    safe_title = re.sub(r'[^\w\-_\. ]', '_', title.lower())

    # Replace spaces with underscore
    safe_title = safe_title.replace(' ', '_')

    # Combine base_name and safe_title
    full_name = f"{base_name}_{safe_title}"

    # Replace multiple consecutive underscores with a single underscore
    full_name = re.sub(r'_{2,}', '_', full_name)

    return f"{full_name}.txt"

def main():
    config = {
        "openai_api_key": "xx",
        "image_domain": "https://lblc.cc",
        "max_workers": 10
    }

    replacer = MarkdownImageCaptionReplacer(config)

    files = ['lecture_1_monetary_policy_part_1.md', 'lecture_2_monetary_policy_part_2.md', 'lecture_3_fiscal_policy.md',
             'lecture_4_exchange_rates.md', 'lecture_5__balanca_of_payments.md']

    for file in files:
        with open(file, 'r') as f:
            content = f.read()

        content_without_empty_headers = replacer.remove_empty_headers(content)
        chunks = replacer.split_into_chunks(content_without_empty_headers)

    # Process the content
        with ThreadPoolExecutor(max_workers=config['max_workers']) as executor:
            futures = []
            for title, content in chunks:
                future = executor.submit(replacer.process_markdown, content)
                futures.append((title, future))

            processed_chunks = []
            for title, future in futures:
                try:
                    processed_content = future.result()
                    processed_chunks.append((title, processed_content))
                except Exception as exc:
                    print(f'Processing generated an exception: {exc}')
                    processed_chunks.append((title, content))  # Keep original if processing fails

            for title, content in processed_chunks:
                # Create a safe filename
                filename = create_safe_filename(file, title)
                filepath = os.path.join('input', filename)

                with open(filepath, 'w') as f:
                    f.write(f"{file} - {title}\n\n")
                    f.write(content)

if __name__ == "__main__":
    main()