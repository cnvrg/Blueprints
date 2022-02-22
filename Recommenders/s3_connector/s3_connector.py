import requests
import argparse

parser = argparse.ArgumentParser(description="""Preprocessor""")
parser.add_argument('-u', '--url', action='store', dest='url',
                    default='url', required=True, help="""url""")
args = parser.parse_args()
url = args.url

req = requests.get(url)
url_content = req.content
csv_file = open('/cnvrg/user_items_data.csv', 'wb')

csv_file.write(url_content)
csv_file.close()