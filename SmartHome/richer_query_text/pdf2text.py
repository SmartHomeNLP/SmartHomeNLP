from tika import parser
import pandas as pd
import re

privacy_query = []
security_query = []

pdf_path = r"G:\SmartHome\richer_query_text\1-s2.0-S0167404815001017-main.pdf"
raw = parser.from_file(pdf_path)
privacy_content = re.sub('\n+', ' ', raw['content'])
privacy_content = re.sub('R E F E R E N C E S.*', '', privacy_content)

privacy_query.append(privacy_content)

pdf_path = r"G:\SmartHome\richer_query_text\06187862.pdf"
raw = parser.from_file(pdf_path)
security_content = re.sub('\n+', ' ', raw['content'])
security_content = re.sub('REFERENCES.*', '', security_content)

security_query.append(security_content)

pdf_path = r"G:\SmartHome\richer_query_text\25148715.pdf"
raw = parser.from_file(pdf_path)
privacy_content = re.sub('\n+', ' ', raw['content'])
privacy_content = privacy_content[1870:]
privacy_content = re.sub(r'References.*', '', privacy_content)

privacy_query.append(privacy_content)

pdf_path = r"G:\SmartHome\richer_query_text\j.1745-6606.2006.00070.x (1).pdf"
raw = parser.from_file(pdf_path)
privacy_content = re.sub('\n+', ' ', raw['content'])
privacy_content = re.sub(r'REFERENCES.*', '', privacy_content)

privacy_query.append(privacy_content)

privacy_query = pd.DataFrame({'text': privacy_query})
privacy_query.to_csv('./richer_query_text/privacy_query.csv')

security_query = pd.DataFrame({'text': security_query})
security_query.to_csv('./richer_query_text/security_query.csv')