import os
import argparse

parser = argparse.ArgumentParser(description="""Kaggle Dataset Connector""")
parser.add_argument('--kaggle_dataset_name', action='store', dest='kaggle_dataset_name', required=True, help="""--- The name of the dataset ---""")

parser.add_argument('--target_path', action='store', dest='target_path', default='/cnvrg/output', help="""--- The path to save the dataset files to ---""")

parser.add_argument('--cnvrg_dataset', action='store', dest='cnvrg_dataset', required=False, default='None', help="""--- the name of the cnvrg dataset to store in ---""")

parser.add_argument('--file_name', action='store', dest='file_name', required=False, default='None', help="""--- If a single file is needed then this is the name of the file ---""")

parser.add_argument('--project_dir', action='store', dest='project_dir', help="""--- For inner use of cnvrg.io ---""")

parser.add_argument('--output_dir', action='store', dest='output_dir', help="""--- For inner use of cnvrg.io ---""")

parser.add_argument('--kaggle_username', action='store', dest='kaggle_username',required=False,  help="""--- For inner use of cnvrg.io ---""")

parser.add_argument('--kaggle_key', action='store', dest='kaggle_key', required=False, help="""--- For inner use of cnvrg.io ---""")

args = parser.parse_args()
dataset_name = args.kaggle_dataset_name
dataset_path = args.target_path
file_name = args.file_name
cnvrg_dataset = args.cnvrg_dataset

kaggle_username = args.kaggle_username
kaggle_key =  args.kaggle_key
os.environ["KAGGLE_KEY"]=kaggle_key
os.environ["KAGGLE_USERNAME"]=kaggle_username

download_command = f'kaggle datasets download {dataset_name} --unzip'

if dataset_path:
    download_command +=  f' -p {dataset_path}'
if file_name.lower() != 'none':
    download_command += f' -f {file_name}'

print(f'Downloading dataset {dataset_name} to {dataset_path}')
os.system(download_command)

if cnvrg_dataset.lower() != 'none':
    from cnvrgv2 import Cnvrg
    cnvrg = Cnvrg()
    ds = cnvrg.datasets.get(cnvrg_dataset)
    try:
        ds.reload()
    except:
        print('The provided Dataset was not found')
    print(f'Creating a new dataset named {cnvrg_dataset}')
    ds = cnvrg.datasets.create(name=cnvrg_dataset)
    print('Uploading files to Cnvrg dataset')
    ds.put_files(paths=[dataset_path])
    
