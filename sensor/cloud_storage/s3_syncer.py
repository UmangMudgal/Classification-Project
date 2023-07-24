import os 


class S3Sync:
    def sync_folder_to_s3(self, folder, aws_bucket_url):
        """
        Description: This will push the desired folder in the local to S3 Bucket
        :param: folder: Artifact Folder path
                aws_bucket_url: AWS S3 Bucket URL
        """
        command = f"aws s3 sync {folder} {aws_bucket_url}"
        os.system(command=command)

    def sync_folder_from_s3(self, folder, aws_bucket_url):
        """
        Description: This will pull the desired folder in the local to S3 Bucket
        :param: folder: Folder path to save the folder from S3 Bucket
                aws_bucket_url: AWS S3 Bucket URL
        """
        command = f"aws s3 sync {aws_bucket_url} {folder}"
        os.system(command=command)