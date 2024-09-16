import os
import csv
import fnmatch
import pandas as pd
from tqdm import tqdm
from xml.etree.ElementTree import parse
import argparse
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
LABEL_ID_DICT = {}
ROOT_PATH = '/content'

def label_id_parser():
    labels = ['barricade', 'bicycle', 'bus', 'car', 'carrier', 'cat', 'dog', 'motorcycle', 'movable_signage', 'person', 'scooter', 'stroller', 'truck', 'wheelchair',
          'bench', 'bollard', 'chair', 'fire_hydrant', 'kiosk', 'parking_meter', 'pole', 'power_controller', 'potted_plant', 'stop', 'table', 'traffic_light_controller', 'traffic_light', 'traffic_sign', 'tree_trunk']
    for i in range(len(labels)):
        LABEL_ID_DICT[labels[i]] = i
        
def label2id(x):
    return LABEL_ID_DICT[x]

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
  
def get_info(bbox_folder,file_name, root_path, save_path):
    field = ["folder_name", "image_id", "width", "height", "label", "occluded", "xtl", "ytl", "xbr", "ybr"]

    tree = parse(os.path.join(ROOT_PATH,'data',bbox_folder, file_name))
    root = tree.getroot()

    images = root.findall("image")
    rows = []
    for img in images:
        boxes = img.findall("box")
        for box in boxes:
            row = [file_name.replace('.xml', ''), img.get("name"), img.get("width"), img.get("height"), box.get("label"), 
                   box.get("occluded"), box.get("xtl"), box.get("ytl"), box.get("xbr"), box.get("ybr")]
            rows.append(row)
    df = pd.DataFrame(rows, columns = field)
    df.to_csv(save_path + '/' + file_name.replace('xml', 'csv'), index=False)

def xml2csv(source):
    src_path = os.path.join(source) # /content/data/Bbox_01/..
    makedirs(ROOT_PATH+'/'+'annotations_csv')
    dest_path = os.path.join(ROOT_PATH,'annotations_csv')
    
    for folder in tqdm(os.listdir(src_path), desc = 'xml parsing...'):
        unzip_check = False
        for file_ in os.listdir(os.path.join(src_path,folder)):
            if file_.split('.')[-1] == 'xml':
                try:
                    get_info(folder, file_, src_path, dest_path)
                except Exception as e:
                    print(file_, e)
            
          
def csv2txt(dest):
    csv_path = os.path.join(ROOT_PATH,'annotations_csv')
    makedirs(dest)
    txt_path = os.path.join(dest)
    label_id_parser()
    
    for csv_file in tqdm(os.listdir(csv_path)):
        df = pd.read_csv(os.path.join(csv_path,csv_file))

        image_id = df['image_id'].unique()

        for id in image_id:
            file_name = id.replace('jpg','txt')
  
            label_df = df[df['image_id'] == id][['label','xtl','ytl','xbr','ybr','width','height']]
            label_df.loc[:,('label')] = label_df['label'].apply(label2id)
  
            label_df.loc[:,('center_x')] = round((label_df['xtl'] + label_df['xbr']) / (2 * label_df['width']), 6)
            label_df.loc[:,('ceenter_y')] = round((label_df['ytl'] + label_df['ybr']) / (2 * label_df['height']), 6)
            label_df.loc[:,('width_norm')] = round((label_df['xbr'] - label_df['xtl']) / label_df['width'] , 6)
            label_df.loc[:,('height_norm')] = round((label_df['ybr'] - label_df['ytl']) / label_df['height'] , 6)
  
            label_df.drop(labels = ['xtl','ytl','xbr','ybr','width','height'], axis = 1,inplace = True)
  
            label_df.to_csv(os.path.join(txt_path,file_name), sep=' ', header = None, index = False)
  
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',  help='source folder name')
    parser.add_argument('--dest',  help='destination folder name')
    parser.add_argument('--root',  default='/content')

    args = parser.parse_args()

    xml2csv(args.source) # xml파일에서 데이터 추출 후 csv 파일 생성
    csv2txt(args.dest)