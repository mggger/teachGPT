import base64
import json
import os
import re
from collections import defaultdict

import boto3
import requests
import unstructured_client
from botocore.config import Config as BotoConfig
from unstructured_client.models import operations, shared
import logging


class PDFProcessor:
    def __init__(self, config):
        self.config = config
        self.unstructured_client = unstructured_client.UnstructuredClient(
            api_key_auth=config['unstructured_api_key'],
            server_url=config['unstructured_api_endpoint'],
        )
        self.r2_client = boto3.client(
            's3',
            endpoint_url=config['r2_endpoint_url'],
            aws_access_key_id=config['r2_access_key'],
            aws_secret_access_key=config['r2_secret_key'],
            config=BotoConfig(signature_version='s3v4'),
            region_name='auto'
        )

        self.image_count = {}  # To keep track of image count per page
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def run(self, pdf_path):
        json_elements = self.extract_pdf(pdf_path)
        return self.process_content(json_elements, pdf_path)

    def extract_pdf(self, filename):
        with open(filename, "rb") as f:
            data = f.read()
        req = operations.PartitionRequest(
            partition_parameters=shared.PartitionParameters(
                files=shared.Files(
                    content=data,
                    file_name=filename,
                ),
                strategy=shared.Strategy.HI_RES,
                languages=['eng'],
                split_pdf_page=True,
                split_pdf_allow_failed=True,
                split_pdf_concurrency_level=15,
                extract_image_block_types=["Image", "Table"]
            ))

        res = self.unstructured_client.general.partition(request=req)
        element_dicts = [element for element in res.elements]
        return json.dumps(element_dicts, indent=2)

    @staticmethod
    def html_table_to_markdown(html):
        rows = re.findall(r'<tr.*?>(.*?)</tr>', html, re.DOTALL)
        csv_rows = []
        for row in rows:
            cells = re.findall(r'<t[hd].*?>(.*?)</t[hd]>', row, re.DOTALL)
            # Replace newlines and commas in cell content
            cleaned_cells = [cell.strip().replace('\n', ' ').replace(',', ';') for cell in cells]
            csv_rows.append(','.join(cleaned_cells))

        # Combine all rows into a single string
        csv_content = '\n'.join(csv_rows)

        output = "The CSV table is as follows:\n\n"
        output += csv_content + "\n"
        return output

    def upload_to_r2(self, image_data, filename):
        self.r2_client.put_object(Bucket=self.config['r2_bucket_name'], Key=filename, Body=image_data)
        url = f"{self.config['r2_public_url']}/{filename}"
        return url

    def sanitize_filename(self, filename):
        # Convert to lowercase and replace non-alphanumeric characters with underscore
        return re.sub(r'[^a-z0-9]+', '_', filename.lower())

    def get_image_filename(self, pdf_filename, page_number):
        base_filename = os.path.splitext(pdf_filename)[0]
        sanitized_filename = self.sanitize_filename(base_filename)

        # Increment image count for this page
        if (sanitized_filename, page_number) not in self.image_count:
            self.image_count[(sanitized_filename, page_number)] = 1
        else:
            self.image_count[(sanitized_filename, page_number)] += 1

        image_number = self.image_count[(sanitized_filename, page_number)]

        return f"{sanitized_filename}_{page_number}_{image_number}.png"

    def get_image_description(self, image_data):
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
                            "text": "Briefly describe this picture in 20 words or less."
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

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return "Failed to get image description."

    def process_content(self, json_elements, pdf_path):
        grouped_data = defaultdict(list)
        for item in json.loads(json_elements):
            metadata = item['metadata']
            page_number = metadata['page_number']
            grouped_data[page_number].append(item)

        all_pages = []
        filename = os.path.basename(pdf_path)

        for page_number in sorted(grouped_data.keys()):
            self.logger.info(f"Processing page {page_number} of {filename}")
            page_content = []
            title = f"{filename}_{page_number}"

            for item in grouped_data[page_number]:
                if item['type'] == 'NarrativeText':
                    page_content.append(item['text'])
                elif item['type'] == 'Image':
                    self.logger.info(f"Processing image on page {page_number}")
                    img_data = base64.b64decode(item['metadata']['image_base64'])
                    img_filename = self.get_image_filename(filename, page_number)
                    img_url = self.upload_to_r2(img_data, img_filename)
                    img_description = self.get_image_description(img_data)
                    page_content.append(f"{img_description}(link: {img_url})")
                    self.logger.info(f"Image processed and uploaded: {img_url}")
                elif item['type'] == 'Table':
                    self.logger.info(f"Processing table on page {page_number}")
                    markdown_table = self.html_table_to_markdown(item['metadata']['text_as_html'])
                    page_content.append(markdown_table)
                    self.logger.info("Table processed and converted to Markdown")

            all_pages.append({
                "id": title,
                "title": filename,
                "text": '\n'.join(page_content).strip()
            })
            self.logger.info(f"Completed processing page {page_number}")

        self.logger.info(f"Finished processing all pages of {filename}")
        return all_pages