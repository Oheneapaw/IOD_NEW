#!/usr/bin/env python
# coding: utf-8

# <div>
# <img src=https://www.institutedata.com/wp-content/uploads/2019/10/iod_h_tp_primary_c.svg width="300">
# </div>

# # Lab 10.1: Introduction to AWS SageMaker and Managed Services
# 
# 

# 
# ### Introduction
# In this lab you will be introduced to some Amazon Web Services. Notably, you will see how to work with Jupyter notebooks in a SageMaker session, make use of cloud storage through an S3 bucket and explore several demos related to image recognition and text analysis.
# 
# Note that this notebook will need to be loaded into an AWS SageMaker instance under a `conda_python3` environment in order to be run successfully.

# ### 1. Connecting to AWS

# a) Create a free AWS account via https://aws.amazon.com/free.

# b) Spend a few moments familiarising yourself with this page noting the extensive list of AWS services within categories such as Compute, Machine Learning, Storage and Analytics. All of these are documented at https://docs.aws.amazon.com/.

# ### 2. Creating an Amazon SageMaker notebook instance

# a) From the console select "Amazon SageMaker". From the left select "Notebook instances". https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/notebook-instances

# b) Click the orange button "Create notebook instance" and enter a Notebook instance name of your choice.

# c) Keep the notebook instance type as the default.

# d) Under "Permissions and encryption" - IAM role, select "create a new IAM role" if you do not have one already. This is a set of permissions to make AWS service requests. More information about IAM (Identity & Access Management) can be found at https://aws.amazon.com/iam/faqs/.

# e) All other options may be left at their default settings. Note that there is also the option to clone a public git repository to this notebook instance but we will not be doing this here. Finally select "Create notebook instance" at the bottom of the page.

# f) It usually takes a few minutes for the instance to be ready as the status changes from "Pending" to "InService". Then you may click on "Open Jupyter" and you will be taken to a familiar jupyter dashboard view. You may browse through a number of SageMaker example notebooks by selecting the "SageMaker Examples" tab. Any of these can either be previewed or used.

# ### 3. Using SageMaker and S3

# Upload this notebook into your AWS SageMaker instance under a `conda_python3` environment.

# #### Import libraries

# In[3]:


#!pip install boto3


# In[4]:


import pandas as pd
from skimage import io
from io import BytesIO
import urllib.request
from matplotlib import pyplot as plt
import boto3 #AWS API
from zipfile import ZipFile


# The boto3 library allows access to AWS APIs via wrapper functions.

# Clients provide low-level access to Amazon Web Services. Resources are higher level with more object-oriented readable syntax providing less functionality than clients. Examples of resources include or S3 buckets (storage) or EC2 instances (computing).

# In[5]:


s3 = boto3.client('s3')
s3_resource = boto3.resource('s3')


# Running the cell below should indicate the region of your SageMaker instance.

# In[6]:


my_region = boto3.session.Session().region_name
print("Success - my SageMaker instance is in the " + my_region + " region.")


# Next we create an S3 bucket (container) into which we will place data. Files within buckets are accessed by keys. Hence note that while dir1/f1 and dir1/f2 are keys referencing two files, f1 and f2 are not interpreted as files contained within dir1.

# In[7]:


my_bucket_name = 'charlie-lab-10-1' # <--- CHANGE THIS TO A UNIQUE NAME FOR YOUR BUCKET (no uppercase letters)


# In[8]:


try:
    if  my_region == 'us-east-1':
        s3_resource.create_bucket(Bucket=my_bucket_name)
    else:
        s3_resource.create_bucket(Bucket=my_bucket_name, CreateBucketConfiguration={ 'LocationConstraint': my_region })
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error: ',e)


# Verify that a bucket was created also by viewing the S3 console at https://console.aws.amazon.com/s3.

# **Exercise**: Fill in the missing code to list your S3 buckets.

# In[9]:


response = s3.list_buckets()
print(response)

# Output the bucket names
print('Existing buckets:')
for bucket in response['Buckets']:
    print(f'  {bucket["Name"]}')


# The following code allows one to list the contents of a bucket along with file sizes.

# In[10]:


# Function from https://github.com/aws-samples/aws-open-data-analytics-notebooks/blob/master/exploring-data/README.md

def list_bucket_contents(bucket, match='', size_mb=0):
    bucket_resource = s3_resource.Bucket(bucket)
    total_size_gb = 0
    total_files = 0
    match_size_gb = 0
    match_files = 0
    for key in bucket_resource.objects.all():
        key_size_mb = key.size/1024/1024
        total_size_gb += key_size_mb
        total_files += 1
        list_check = False
        if not match:
            list_check = True
        elif match in key.key:
            list_check = True
        if list_check and not size_mb:
            match_files += 1
            match_size_gb += key_size_mb
            print(f'{key.key} ({key_size_mb:3.0f}MB)')
        elif list_check and key_size_mb <= size_mb:
            match_files += 1
            match_size_gb += key_size_mb
            print(f'{key.key} ({key_size_mb:3.0f}MB)')

    if match:
        print(f'Matched file size is {match_size_gb/1024:3.1f}GB with {match_files} files')

    print(f'Bucket {bucket} total size is {total_size_gb/1024:3.1f}GB with {total_files} files')


# We will investigate the contents of a bucket from the Registry of Open Data on AWS. A listing of these datasets is at https://registry.opendata.aws/. In particular we shall make use of the COCO image dataset (Common Objects in Context), used for object detection and segmentation. From https://registry.opendata.aws/fast-ai-coco/ we see that the bucket name is `fast-ai-coco`.

# In[11]:


list_bucket_contents(bucket='fast-ai-coco')


# Next, we copy the `val2017.zip` file (containing 5000 images) to the bucket you created above via the following function.

# In[12]:


def copy_to_bucket(origin_bucket_name, destination_bucket_name, file_name):
    copy_source = {
        'Bucket': origin_bucket_name,
        'Key': file_name
    }
    s3_resource.Object(destination_bucket_name, file_name).copy(copy_source)


# **Exercise**: Use the above function to copy val2017.zip to your bucket. Then use the list_bucket_contents function to verify that the 'val2017.zip' was successfully transferred. Alternatively, use the S3 console.

# In[13]:


# Define your S3 bucket name 
my_bucket_name = 'charlie-lab-10-1'

# Copy the val2017.zip file from the public 'fast-ai-coco' bucket to your own S3 bucket
copy_to_bucket('fast-ai-coco','charlie-lab-10-1' , 'val2017.zip')


# In[14]:


# List the contents of your S3 bucket to verify that the 'val2017.zip' file was copied
list_bucket_contents(bucket=my_bucket_name)


# S3 is a storage location that does not allow computation. The following code allows us to extract selected files.

# In[15]:


zip_obj = s3_resource.Object(bucket_name= my_bucket_name, key='val2017.zip')
buffer = BytesIO(zip_obj.get()["Body"].read())
z = ZipFile(buffer)


# Here is a list of the files contained in the zip file.

# In[16]:


for filename in z.namelist():
    print(filename)


# We select a sample of these image files for transfer to our SageMaker instance.

# In[17]:


list_of_files = ['007108', '008021', '023781', '024021', '052412', '060855', '062808', '071938', '104424', '108495',
              '138115', '170474', '246308', '280891', '324258', '325483', '336232', '402765', '494913']


# In[18]:


for filename in z.namelist():
    if filename[-10:-4] in list_of_files:
        file_info = z.getinfo(filename)
        s3_resource.meta.client.upload_fileobj(
            z.open(filename),
            Bucket=my_bucket_name,
            Key=f'{filename}'
        )


# **Exercise**: Verify that contents of your S3 bucket have been modified.

# In[19]:


list_bucket_contents(bucket='charlie-lab-10-1')


# Next we copy the jpg files from your S3 bucket to the local instance for viewing.

# In[20]:


for f in list_of_files:
    new_file_name =  f + '.jpg'
    bucket_file_name = 'val2017/000000' + new_file_name
    s3.download_file(my_bucket_name, bucket_file_name, new_file_name)


# Similarly the ``upload_file`` method can be used to upload local files to the bucket.

# **Exercise**: Use the imshow and imread methods to plot some or all the images.

# In[21]:


from skimage import io
import matplotlib.pyplot as plt

# Loop through the list of image file names
for f in list_of_files:
    # Construct the local file name (assuming images were downloaded as <image_name>.jpg)
    new_file_name = f + '.jpg'
    
    # Read the image using skimage's imread
    image = io.imread(new_file_name)
    
    # Plot the image using matplotlib's imshow
    plt.imshow(image)
    plt.title(f'Image: {f}')  # Set the title to the image file name
    plt.axis('off')  # Turn off the axis for a cleaner display
    plt.show()


# The following command can be used to save the files in one archive file. This makes it more convenient to download the images at once. Alternatively one can download the entire (5000-file) archive at https://s3.amazonaws.com/fast-ai-coco/val2017.zip.

# In[22]:


#save file to local machine
get_ipython().system('tar chvfz photos.tar.gz *.jpg')


# Then you can download this file to your machine. Using the command prompt or Mac terminal run `tar -xvzf photos.tar.gz -C photos` to extract the files from this archive.

# ### Working with files from your local machine

# As with your Jupyter notebooks that run locally on your machine, SageMaker notebooks can also work with files from the internet or your local machine.

# **Exercise**: Upload a file from your DATA folder used in this course. Display the first five rows to verify that it was loaded successfully.

# In[ ]:





# In[ ]:





# ### 4. Amazon Rekognition via the Console

# Go the Amazon Rekognition console page by searching for Rekogntion, or go directly to https://console.aws.amazon.com/rekognition/home?region=us-east-1#/. From there click on "Try Demo" to see a series of image recognition capabilities ranging from "Object and Scene Detection" to "Text in Image". In addition to the samples there you may wish to try either your own examples or the following files that we downloaded above:

# - for Object and Scene Detection: 007108, 008021, 023781, 024021, 052412, 060855, 062808, 071938, 246308, 280891, 336232, 494913
# - for Facial Analysis: 104424, 108495, 138115
# - for Celebrity Recognition: 170474
# - for Face Comparison: 324258, 325483
# - for Text in Image: 402765

# Performing these actions via the API rather than the console requires additional access, such as an Amazon Free Tier account (which requires a credit card).

# ### 5. Amazon Textract via the Console

# Go through the following 10-minute demo to understand the features and capabilities of Amazon Textract to extract text from documents.
# 
# https://aws.amazon.com/getting-started/hands-on/extract-text-with-amazon-textract/
# 
# In addition to the same data provided, you may wish to use the files `amzn_stock.pdf` and `sample_form.JPG` from the DATA folder.

# ### 6. Amazon Comprehend via the Console

# Go through the following 10-minute demo to understand the features and capabilities of Amazon Comprehend for text analysis.
# 
# https://aws.amazon.com/getting-started/hands-on/analyze-sentiment-comprehend/

# ### 7. Closing your notebook instance and deleting your S3 bucket

# Go through the following 10-minute demo to understand the features and capabilities of Amazon Comprehend for text analysis.
# 
# https://aws.amazon.com/getting-started/hands-on/analyze-sentiment-comprehend/

# Run the following cell to delete your bucket.

# In[ ]:


bucket_to_delete = s3_resource.Bucket(my_bucket_name)
bucket_to_delete.objects.all().delete()


# Verify that the bucket has been emptied by navigating to https://console.aws.amazon.com/s3/. You may then delete the bucket by selecting it in the S3 console page and clicking "Delete".

# Download this notebook to your local machine when you are done. Then stop your notebook instance by visiting https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/notebook-instances, selecting the instance and then choosing "Stop" under the "Actions" menu.

# 
# 
# ---
# 
# 
# 
# ---
# 
# 
# 
# > > > > > > > > > © 2024 Institute of Data
# 
# 
# ---
# 
# 
# 
# ---
# 
# 
# 
# 
