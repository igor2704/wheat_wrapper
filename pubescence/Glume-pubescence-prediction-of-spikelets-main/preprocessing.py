from config import PROJECT_DIR, DATA_DIR, PATTERNS, HOLDOUT_BASE_DIR, SEGMENTATION_MODEL_DIR
import numpy as np
import pandas as pd
import cv2
import re
import os
import time
from matplotlib import pyplot as plt
import yaml
import random as rand

class DataPreprocessor():
    def __init__(self, raw_labels_data_path=f'{PROJECT_DIR}/raw_labels_data'):
        self.__raw_labels_data_path = raw_labels_data_path
        self.labels = self.process_raw_annotation()
        self.subspecies_stats = {}
        self.species_stats = {}
    
    def process_raw_annotation(self):
        # 1. preprocess raw_labels annotation
        raw_data = []
        with open(f'{self.__raw_labels_data_path}', 'r') as inp:
            raw_data = inp.read().split(';')
            for i, text in enumerate(raw_data):
                raw_data[i] = text.split('\n')
                filter_obj = filter(lambda x: x!='', raw_data[i])
                raw_data[i] = list(filter_obj)
            raw_data.pop(0)

        # 2. create dataframe and write data
        df = pd.DataFrame({'Номер_каталога': [], 'Вид' : [], 'Опушение' : [],
                          'Вегетация' : [], 'Подвиды' : []})

        ploid = 'Диплоиды'
        species = ''
        for data_list in raw_data:
            if not data_list:
                continue

            # x == species in that case or ploidness
            if len(data_list) == 1:
                species = re.search(PATTERNS['species'], data_list[0])
                if not species:
                    ploid = re.search(PATTERNS['ploid'], data_list[0]).group()
                else:
                    species = species.group()

            # otherways we consider specific species of wheat
            else:
                catalog_num = data_list[0]
                vegetation_list = []
                subspecies_list = []

                pubescence = 0
                if data_list[1] == '+':
                    pubescence = 1

                # regex patterns to distinguish vegetation for order numbers
                for i in range(2, len(data_list)):
                    vegetation = re.search(PATTERNS['vegetation'], data_list[i])
                    if vegetation:
                        vegetation_list.append(vegetation.group())
                    else:
                        # it's not vegetation though
                        subspecies = re.search(PATTERNS['subspecies'], data_list[i])
                        if subspecies:
                            subspecies_list.append(subspecies.group())

                # add data into dataframe
                df = df.append(pd.DataFrame({'Номер_каталога': [catalog_num],
                       'Вид':[species],
                       'Опушение':[str(pubescence)],
                       'Вегетация':[';'.join(vegetation_list)],
                       'Подвиды':[';'.join(subspecies_list)]
                      }))

        df.to_csv(f'{PROJECT_DIR}/labels.csv', index=False)
        return df
    
    # Go through file system and calculate general num of images
    def __img_type_stats(self, img_type):
        df = self.labels.copy()
        species = {'Вид': [],
                   'Плоидность': [],
                   'Опушение': [],
                   'Тип_изображений': [],
                   'Подвиды': [],
                   'Опушенные': [],
                   'Неопушенные': []}
        
        for dirpath, dirnames, filenames in os.walk(DATA_DIR):
            if re.search(img_type, dirpath):
                spec = re.search(PATTERNS['species'], dirpath).group()
                ploid = re.search(PATTERNS['ploid'], dirpath).group()
                pubesc = 0
                    
                pubesc_stats = [0, 0]
                subspec_name = re.search(PATTERNS['subspecies'], dirpath)
                if subspec_name:
                    subspec_name = subspec_name.group()
                else:
                    continue
                    
                for fn in filenames:
                    if re.search(PATTERNS[img_type], fn):
                        vegetation = re.search(PATTERNS['vegetation'], fn)
                        tmp1 = df.loc[df['Вид'] == spec]
                        tmp2 = tmp1.loc[tmp1['Подвиды'].str.contains(subspec_name)]
                        tmp3 = pd.DataFrame()
                        if vegetation:
                            vegetation = vegetation.group()
                            tmp3 = tmp2.loc[tmp2['Вегетация'].str.contains(vegetation)]
                        else:
                            vegetation = ''
                            
                        if tmp2.empty:
                            break
                        
                        # probably because russian X and american X
                        # problem of human's labour...
                        if tmp3.empty and 'X' in vegetation:
                            vegetation = vegetation.replace('X', 'Х')
                            tmp3 = tmp2.loc[tmp2['Вегетация'].str.contains(vegetation)]
                        elif tmp3.empty and 'Х' in vegetation:
                            vegetation = vegetation.replace('Х', 'X')
                            tmp3 = tmp2.loc[tmp2['Вегетация'].str.contains(vegetation)]
                        elif not tmp2.empty: # not all images are represented with vegetation
                            if len(tmp2) > 1:
                                tmp2 = tmp2[0:1]
                                
                            pubesc = int(tmp2['Опушение'].item())
                            pubesc_stats[pubesc] += 1
                            continue
                            
                        if len(tmp3) > 1:
                            tmp3 = tmp3[0:1]
                        pubesc = int(tmp3['Опушение'].item())
                        pubesc_stats[pubesc] += 1
                
                species['Вид'].append(spec)
                species['Плоидность'].append(ploid)
                species['Тип_изображений'].append(img_type)
                species['Подвиды'].append(subspec_name)
                species['Опушенные'].append(pubesc_stats[1])
                species['Неопушенные'].append(pubesc_stats[0])
                species['Опушение'].append(pubesc)
        return pd.DataFrame(species)
        
    def dataset_stats(self):
        pin_stats = self.__img_type_stats('pin')
        table_stats = self.__img_type_stats('table')
        df = pd.concat([pin_stats, table_stats], axis=0)
        df['Опушенные'] = df['Опушенные'].astype(np.int16)
        df['Неопушенные'] = df['Неопушенные'].astype(np.int16)
        df['Всего'] = df['Опушенные'] + df['Неопушенные']
        
        self.subspecies_stats = df.copy()
        tmp = df[df['Всего'] > 0][['Вид', 'Опушенные', 'Неопушенные', 'Всего']]
        self.species_stats = tmp.groupby('Вид').sum().sort_values(by='Всего')
        return df
    
        
    # Used to extract species for training
    def create_annotation_file(self, split_species_dict=None, ratios=[0.8, 0.1, 0.1], filename='example_data_split'):
        df = self.labels.copy()
        
        def get_label(img_path):
            label = None
            # get name of subspecies and species
            subspecies = re.search(PATTERNS['subspecies'], img_path).group()
            species = re.search(PATTERNS['species'], img_path).group()
            vegetation = re.search(PATTERNS['vegetation'], img_path)

            # get the label according to species and subspecies
            tmp1 = df.loc[df['Вид'] == species]
            tmp2 = tmp1.loc[tmp1['Подвиды'].str.contains(subspecies)]
            tmp3 = pd.DataFrame()
            if vegetation:
                vegetation = vegetation.group()
                tmp3 = tmp2.loc[tmp2['Вегетация'].str.contains(vegetation)]
            else:
                vegetation = ''

            if tmp2.empty:
                return None

            # probably because russian X and american X
            # problem of human's labour...
            if tmp3.empty and 'X' in vegetation:
                vegetation = vegetation.replace('X', 'Х')
                tmp3 = tmp2.loc[tmp2['Вегетация'].str.contains(vegetation)]
            elif tmp3.empty and 'Х' in vegetation:
                vegetation = vegetation.replace('Х', 'X')
                tmp3 = tmp2.loc[tmp2['Вегетация'].str.contains(vegetation)]
            elif not tmp2.empty: # not all images are represented with vegetation
                if len(tmp2) > 1:
                    tmp2 = tmp2[0:1]
                label = int(tmp2['Опушение'].item())
            
            if not tmp3.empty:
                if len(tmp3) > 1:
                    tmp3 = tmp3[0:1]
                label = int(tmp3['Опушение'].item())
            return label

        def get_sample(img_path):
            species = re.search(PATTERNS['species'], img_path).group()
            for key, value in split_species_dict.items():
                if species in value:
                    return key
            return None

        split_dict = {'Path': [], 'Label': [], 'Group': []}
        index_to_group = {
            0: 'train',
            1: 'valid',
            2: 'test'
        }
        count = 0
            
        for dirpath, dirnames, filenames in os.walk(DATA_DIR):
            for fn in filenames:
                img_name = re.search(PATTERNS['pin'], fn)
                if not img_name:
                    img_name = re.search(PATTERNS['table'], fn)

                # if filename doesn't satisfy any of patterns above
                if not img_name:
                    continue

                img_name = img_name.group()
                img_path = f'{dirpath}/{img_name}'
                label = get_label(img_path)

                # if label wasn't defined for that instance
                if label is None:
                    continue

                split_dict['Path'].append(f'{dirpath}/{fn}')
                split_dict['Label'].append(label)
                if split_species_dict:
                    split_dict['Group'].append(get_sample(dirpath))
                else:
                    group_index = rand.choices([0, 1, 2], weights=ratios)[0]
                    group = index_to_group[group_index]
                    split_dict['Group'].append(group)
                count += 1
                    

        # save data and species summary
        split_dict = pd.DataFrame(split_dict)
    
        # save csv and yaml file about split
        if split_species_dict:
            with open(f'{PROJECT_DIR}/splits/{filename}.yaml', 'w') as out:
                yaml.dump(split_species_dict)
            
        split_dict.to_csv(f'{PROJECT_DIR}/splits/{filename}.csv', sep=';', index=False)
        
    def split_species(self, split_name):
        # there are several known splits. Firstly process them.
        if not os.path.isfile(f'{PROJECT_DIR}/labels.csv'):
            raise FileNotFoundError(f"Labels file hasn't been generated yet")
            
        if split_name in os.listdirs(f'{PROJECT_DIR}/splits/'):
            print(f'{split_name} is already created in {PROJECT_DIR}/splits/')
        else:
            self.form_optimal_random_split()
            
    def form_optimal_random_split(self, species_data=None, 
                                  epsilon=0.025, delta=0.1, limit=1000, verbose=True, plot=False):
        def choose_random(source):
            rand_group = rand.randint(0,2)
            rand_idx = rand.randint(0, len(source)-1)
            return (rand_idx, rand_group)

        def update_ratio(new_species, group_idx):
            ratios[group_idx][0] += species_data.loc[new_species][2]/sum_samples
            # number of sample with/without pubescence
            ratios[group_idx][1] += species_data.loc[new_species][1]
            ratios[group_idx][2] += species_data.loc[new_species][0]

        # groups[0] is supposed to be the largest, so we predefine
        # some species manually to make alforithm run faster
        def predefine_run():
            predefined = ['T. aethiopicum', 'T. durum', 'T. timopheevii']
            groups = [predefined, [], []]
            ratios = [[0, 0, 0], [0, 0,  0], [0, 0, 0]]

            # fill ratio with right nubers
            for s in predefined:
                ratios[0][0] += species_data.loc[s][2]/sum_samples
                ratios[0][1] += species_data.loc[s][1]
                ratios[0][2] += species_data.loc[s][0]

            species = list(species_data.index)
            for x in predefined:
                species.remove(x)
            return (species, groups, ratios)
        
        # check dict
        if not species_data:
            species_data = self.species_stats.copy()
            
        # let's compute overall number of samples:
        sum_samples = species_data['Всего'].sum()
            
        # general statistics
        start = time.time()
        num_iterations = 0
        max_diff = 100
        best_ratios = []
        best_choice = []
        statistics = []
        failures = 0

        species, groups, ratios = [], [], []

        while failures < limit and max_diff > delta:
            failures += 1
            num_iterations += 1
            # define data for single experiment
            species, groups, ratios = predefine_run()

            while species:
                # if all validation and test has 0.08 - 0.13 of overall 
                # - add all in train and stop
                if 0.1 - delta < ratios[1][0] < 0.1 + delta and 0.1 - delta < ratios[2][0] < 0.1 + delta:
                    for s in species:
                        groups[0].append(s)
                        update_ratio(s, 0)
                    break

                rand_idx, rand_group = choose_random(species)
                # if validation or test
                # we need to ensure, that perc_{i} < 0.13
                if rand_group == 1 or rand_group == 2:
                    tmp_ratio = ratios[rand_group][0] + species_data.loc[species[rand_idx]][2]/sum_samples
                    if tmp_ratio < 0.1 + delta:
                        groups[rand_group].append(species[rand_idx])
                        update_ratio(species[rand_idx], rand_group)
                        species.pop(rand_idx)
                    # else - new iteration
                else:
                    groups[rand_group].append(species[rand_idx])
                    update_ratio(species[rand_idx], rand_group)
                    species.pop(rand_idx)

            # now we need to check coefficients
            k1, k2, k3 = [r[2]/r[1] if r[1]!=0 else -1 for r in ratios]
            diff = max(abs(k1-k2), abs(k2-k3), abs(k1-k3))
            if diff < max_diff:
                max_diff = diff
                failures = 0
                best_choice = groups.copy()
                best_ratios = ratios.copy()
                if verbose:
                    print(f'The choice now is better: {max_diff}')
            statistics.append(max_diff)

        if verbose:
            print(f'Final choice: {max_diff}')
            print('Time consumed', time.time()-start)
        
        # now we may plot graph of monte carlo method
        if plot:
            x = np.arange(0, num_iterations)
            plt.plot(x, statistics, color='red')
            plt.xlabel('Number of iterations')
            plt.ylabel('Error')
            plt.savefig('Convergence_of_random_algorithm.png')
            plt.close()
        return (best_choice, best_ratios)
        
    
def get_bbox(raw_img, mask, dilation=False):       
    # acquire 3-channel mask for spike only
    tmp = np.moveaxis(mask, -1, 0)
    spike_body_mask = tmp[1]
    
    # applying dilation to expand boundaries of mask
    if dilation:
        kernel = np.ones((11, 11), np.uint8)
        spike_body_mask = cv2.dilate(spike_body_mask, kernel, iterations=1)
    tmp[0] = spike_body_mask
    tmp[2] = spike_body_mask
    tmp[1] = spike_body_mask
    tmp = np.moveaxis(tmp, [0, 1, 2], [2, 0, 1])
    
    # product of mask and image
    img = cv2.bitwise_and(raw_img, tmp)
            
    # Make crop and unsqueeze to square binary-power form
    original = img.copy()
    gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
    # Find contours, obtain bounding box, extract and save ROI
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # List with dimensions of contours
    areas = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        areas.append((x, y, w, h, w*h))
                    
    areas = sorted(areas, key=lambda tup: tup[4], reverse=True)
    # we need second largest bounding box, first is the overall image
    x,y,w,h,_ = areas[1]
                    
    # get rectangle with corner x,y and the approproate w,h
    cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)
    img = original[y:y+h, x:x+w] 
    return img           


def make_all_bboxes(overwrite=False, dilation=False):
    for dirpath, dirnames, filenames in os.walk(DATA_DIR):
        for fn in filenames:
            # Get data of ploidness, img_name, img_type
            ploid = re.search(PATTERNS['ploid'], dirpath)
            ploid = ploid.group() if ploid else None
            species = re.search(PATTERNS['species'], dirpath)
            species = species.group() if species else None
            img_type = 'pin'
            img_name = re.search(PATTERNS['pin'], fn)
            if not img_name:
                img_type = 'table'
                img_name = re.search(PATTERNS['table'], fn)
            if img_name:
                img_name = img_name.group()
            
            # now we process our acquired data
            if not overwrite and os.path.exists((f'''{save_dir}/{ploid}/{species}/{img_type}
                                                /{img_name}''').replace('.jpg','.png').replace('.JPG','.png')):
                return
    
    # Create directory with species
    if not os.path.exists(f'{save_dir}/{ploid}/{species}/{img_type}'):
        species = species.replace(' ', '\ ')
        os.system(f'mkdir {save_dir}/{ploid}')
        os.system(f'mkdir {save_dir}/{ploid}/{species}')
        os.system(f'mkdir {save_dir}/{ploid}/{species}/{img_type}')
        
    # Open files, mask and get product o mask and image
    raw_img = cv2.imread(dirpath + '/' + img_name)
    nm = (f'{mask_dir}/{ploid}/{img_type}/{img_name}').replace('.jpg','.png').replace('.JPG','.png')
    mask = cv2.imread(nm)
    
    if mask is None:
        raise RuntimeError(f'There is no mask for image: {dirpath}/{img_name}\n')
    if mask.shape != raw_img.shape:
        raise RuntimeError(f'Dimension problem: {dirpath}/{img_name}\n')
                           
    # run function to extract bbox
    img = get_bbox(raw_img, mask, dilation=dilation)
     
    # save bounding box
    species = species.replace('\ ', ' ')
    cv2.imwrite((f'{save_dir}/{ploid}/{species}/{img_type}/{img_name}').replace('.jpg', '.png').replace('.JPG', '.png'), img)
    
            
def make_all_holdout_bboxes(overwrite=False, dilation=False):
    save_path = f'{HOLDOUT_BASE_DIR}spikelets/'
    inp_path = f'/data/cv_project/spikedroid/опушение/'
    mask_path = f'{HOLDOUT_BASE_DIR}masks/'
    pubesc_patt = r'опушением'

    with open(f'{HOLDOUT_BASE_DIR}holdout_annotation', 'w') as out:
        for dirpath, dirnames, filenames in os.walk(inp_path):
            for fn in filenames:
                # define target
                target = 1 if re.search(pubesc_patt, dirpath) else 0
                
                # read image, mask and get pure image
                raw_img = cv2.imread(f'{dirpath}/{fn}')
                mask = []
                tmp_path = f'{mask_path}неопушенные/{fn}'.replace('jpg', 'png')
                if target == 1:
                    tmp_path = f'{mask_path}опушенные/{fn}'.replace('jpg', 'png')
                    mask = cv2.imread(tmp_path)
                else:
                    mask = cv2.imread(tmp_path)
                    
                if mask is None:
                    raise RuntimeError(f'There is no mask for image: {dirpath}/{img_name}\n')
                if mask.shape != raw_img.shape:
                    raise RuntimeError(f'Dimension problem: {dirpath}/{img_name}\n')
                
                img = get_bbox(raw_img, mask, dilation=dilation)
                
                cv2.imwrite(f'{save_path}{fn}', img)
                out.write(f'{save_path}{fn};{target};test\n')
                
                
def get_mask(input_path, save_mask=False):
    init_dir = os.getcwd()
    inp = input_path.replace(' ', '\\ ')
    out = f'{os.getcwd()}'.replace(' ', '\\ ')
    os.chdir('/home/rostepifanov/import/bin/segmentation/')
    SEG_SCRIPT = f'./infer -bone efficientnet-b2 -mn model_efficientnet-b2.bin --cuda --verbose -bs 32 -ip {inp} -op {out}/'
    mask = []
    try:
        os.system(SEG_SCRIPT)
        img_patt = r'[-\w \d \s _{}]*.(jpg|png|PNG|JPG)'
        mask_name = re.search(img_patt, input_path).group().replace('jpg','png').replace('JPG', 'png')
        mask = cv2.imread(f'{out}/{mask_name}')
        if not save_mask:
            os.remove(f'{out}/{mask_name}')
    except FileNotFoundError:
        print("Image wasn't found or save dir doesn't exist. Check you image path, please!")
    os.chdir(init_dir)
    return mask
    
    
def get_dataset_masks():
    init_dir = os.getcwd()
    os.chdir(SEGMENTATION_MODEL_DIR)
    
    SEG_SCRIPT = './infer -bone efficientnet-b2 -mn model_efficientnet-b2.bin --cuda --verbose -bs 32 -ip {img_path} -op {out_path}'
    for dirpath, dirnames, filenames in os.walk('/data/cv_project/spikedroid/ploid_classification/'):
        for fn in filenames:
            img_name = re.search(PATTERNS['pin'], fn)
            img_type = 'pin'
            if not img_name:
                img_name = re.search(PATTERNS['table'], fn)
                img_type = 'table'
            if not img_name:
                img_type = None
                continue
                
            img_name = img_name.group()
            ploid = re.search(PATTERNS['ploid'], dirpath).group(1)
            os.system(SEG_SCRIPT.format(img_path = f'{dirpath}/{img_name}'.replace(' ', '\ '), 
                            out_path=f'{PROJECT_DIR}/masks/{ploid}/pin'))
    os.chdir(SEGMENTATION_MODEL_DIR)
    
    
def get_holdout_masks():
    init_dir = os.getcwd()
    os.chdir(SEGMENTATION_MODEL_DIR)
    SEG_SCRIPT = './infer -bone efficientnet-b2 -mn model_efficientnet-b2.bin --cuda --verbose -bs 32 -ip {img_path} -op {out_path}'

    for dirpath, dirname, filenames in os.walk(f'{HOLDOUT_BASE_DIR}/опушение'):
        pubescened = re.search(r'с\sопушением', dirpath)
        if not pubescened:
            pubescened = 'неопушенные'
        else:
            pubescened = 'опушенные'
            
        for fn in filenames:
            img_path = f'{dirpath}/{fn}'.replace(' ', '\ ')
            out_path = f'{HOLDOUT_BASE_DIR}/masks/{pubescened}'.replace(' ', '\ ')
            if not os.path.exists(f'{HOLDOUT_BASE_DIR}/masks/{pubescened}'):
                os.mkdir(f'{HOLDOUT_BASE_DIR}/masks/{pubescened}')
            os.system(SEG_SCRIPT.format(img_path=img_path, out_path=out_path))
    os.chdir(SEGMENTATION_MODEL_DIR)