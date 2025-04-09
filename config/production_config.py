from pydantic import BaseSettings
from typing import Optional
import boto3
from botocore.exceptions import ClientError

class ProductionConfig(BaseSettings):
    """AWS production database configuration"""
    host: str
    port: int
    database: str
    user: str
    password: str
    region: str

    class Config:
        env_prefix = "AWS_DB_"
        env_file = ".env"

    @classmethod
    def get_aws_secret(cls, secret_name: str) -> str:
        """Retrieve secret from AWS Secrets Manager"""
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=cls().region
        )

        try:
            get_secret_value_response = client.get_secret_value(
                SecretId=secret_name
            )
            return get_secret_value_response['SecretString']
        except ClientError as e:
            raise Exception(f"Error retrieving secret: {str(e)}") 