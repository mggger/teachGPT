from pathlib import Path
import pandas as pd
from graphrag.index.utils import gen_md5_hash

class DB:
    """
    A database class for managing document data using Parquet files.
    """

    def __init__(self):
        """
        Initialize the DB instance.

        :param workspace: The path to the workspace directory
        """
        self.dataset_path = Path("dataset.parquet")
        self._init()

    def _init(self):
        """
        Initialize the database file if it doesn't exist.
        """
        if not self.dataset_path.exists():
            empty_df = pd.DataFrame(columns=["id", "title", "text"])
            empty_df.to_parquet(self.dataset_path)

    def upsert_data(self, doc_data: dict):
        """
        Insert or update a single document in the database.

        :param doc_data: A dictionary containing 'text', 'title', and optionally 'id' of the document
        :return: The ID of the inserted/updated document
        """
        df = self.load_data()
        if 'id' in doc_data and doc_data['id'] is not None:
            id = doc_data['id']
        else:
            id = gen_md5_hash(doc_data, ['text', 'title'])

        new_row = {
            "id": id,
            "text": doc_data['text'],
            "title": doc_data['title']
        }
        if id in df['id'].values:
            df.loc[df['id'] == id] = new_row
        else:
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_parquet(self.dataset_path, index=False)
        return id

    def batch_upsert_data(self, doc_data_list: list[dict]):
        """
        Insert or update multiple documents in the database.

        :param doc_data_list: A list of dictionaries, each containing 'text', 'title', and optionally 'id' of a document
        :return: A list of IDs of the inserted/updated documents
        """
        df = self.load_data()
        new_rows = []
        for doc_data in doc_data_list:
            if 'id' in doc_data and doc_data['id'] is not None:
                id = doc_data['id']
            else:
                id = gen_md5_hash(doc_data, ['text', 'title'])

            new_row = {
                "id": id,
                "text": doc_data['text'],
                "title": doc_data['title']
            }
            new_rows.append(new_row)

        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df]).drop_duplicates(subset='id', keep='last').reset_index(drop=True)
        df.to_parquet(self.dataset_path)
        return [row['id'] for row in new_rows]

    def load_data(self):
        """
        Load the entire dataset from the Parquet file.

        :return: A pandas DataFrame containing all the data
        """
        return pd.read_parquet(self.dataset_path)

    def delete_data(self, ids: list[str]):
        """
        Delete documents from the database based on their IDs.

        :param ids: A list of document IDs to be deleted
        """
        df = self.load_data()
        df = df[~df['id'].isin(ids)]
        df.to_parquet(self.dataset_path)

    def delete_data_by_title(self, title: str):
        """
        Delete all documents from the database that match the given title.

        :param title: The title of the documents to be deleted
        :return: The number of documents deleted
        """
        df = self.load_data()
        initial_count = len(df)
        df = df[df['title'] != title]
        df.to_parquet(self.dataset_path)
        return initial_count - len(df)

    def get_data(self, id: str):
        """
        Retrieve a single document from the database by its ID.

        :param id: The ID of the document to retrieve
        :return: A dictionary containing the document data, or None if not found
        """
        df = self.load_data()
        return df[df['id'] == id].to_dict('records')[0] if not df[df['id'] == id].empty else None

    def get_all_titles(self):
        """
        Retrieve all unique titles from the database.

        :return: A list of all unique titles in the database
        """
        df = self.load_data()
        return df['title'].unique().tolist()