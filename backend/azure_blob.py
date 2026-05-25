from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

import os

load_dotenv()

connection_string = os.getenv(
    "AZURE_STORAGE_CONNECTION"
)

blob_service_client = (
    BlobServiceClient
    .from_connection_string(
        connection_string
    )
)

CONTAINER_NAME = "retailfiles"


def upload_to_blob(file_path):

    blob_name = os.path.basename(
        file_path
    )

    blob_client = (
        blob_service_client
        .get_blob_client(

            container=CONTAINER_NAME,

            blob=blob_name
        )
    )

    with open(
        file_path,
        "rb"
    ) as data:

        blob_client.upload_blob(

            data,

            overwrite=True
        )

    print(
        f"{blob_name} uploaded successfully"
    )