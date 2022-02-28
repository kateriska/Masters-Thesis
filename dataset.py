import glob
import os
import cv2
import numpy as np
import shutil
import random

class Dataset:
    def __init__(self):
        super().__init__()
        self.dataset_path = os.path.join('dataset', 'train', '*')
        self.preprocessed_dataset_path = os.path.join('dataset', 'train_preprocessed', '*')
        self.preprocessed_dataset_val_path = os.path.join('dataset', 'val_preprocessed', '*')
        self.preprocessed_dataset_test_path = os.path.join('dataset', 'test_preprocessed', '*')

    def preprocess_dataset(self):
        for file in glob.glob(os.path.abspath(self.preprocessed_dataset_path)):
            os.remove(file)

        for file in glob.glob(self.dataset_path):
            file_substr = file.split('/')[-1]
            extension = os.path.splitext(file)[1][1:]

            if extension == 'xml':
                shutil.copyfile(file, os.path.join('dataset', 'train_preprocessed', file_substr))
                continue

            img = cv2.imread(file, 0)
            img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)

            ret, tresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # noise removal
            kernel = np.ones((51,51), np.uint8)
            opening = cv2.morphologyEx(tresh_img, cv2.MORPH_OPEN,kernel) # use morphological operations

            cv2.Canny(opening, 100, 200)
            result_tresh = cv2.add(tresh_img, opening)
            result_orig = cv2.add(img, opening) # add mask with input image

            cv2.imwrite(os.path.join('dataset', 'train_preprocessed', file_substr),result_orig)

        self.split_dataset()

    # split dataset into test and train folder and store in train or test folder also their xml annotations
    def split_dataset(self):
        #if len(os.listdir(os.path.abspath('./dataset/train_preprocessed/')) ) == 0:
        #    print("Directory is empty")
        #else:
        print(os.path.abspath(self.preprocessed_dataset_test_path))
        for file in glob.glob(os.path.abspath(self.preprocessed_dataset_val_path)):
            os.remove(file)

        for file in glob.glob(os.path.abspath(self.preprocessed_dataset_test_path)):
            os.remove(file)

        test_images_names = []

        '''
        test_dataset_names = ['dys_SG127_1', 'healthy_SG69_3', 'healthy_SG13_2', 'SG43_3', 'healthy_SG49_5', 'SG13_3', 'atopic_eczema_FP_00027', 'healthy_SG48_2', 'SG20_4', 'healthy_SG5_2', 'healthy_SG75_4', 'atopic_eczema_FP_00317', 'healthy_SG17_3', 'SG4_4', 'SG17_1', 'healthy_SG11_1', 'dys_SG128_2', 'atopic_eczema_FP_00010', 'SG31_4', 'healthy_SG65_1', 'healthy_SG72_2', 'healthy_SG56_3', 'healthy_SG58_1', 'healthy_SG57_2', 'dys_SG110_1', 'dys_SG119_3', 'SG30_5', 'atopic_eczema_FP_00185', 'atopic_eczema_FP_00350', 'SG36_3', 'SG15_5', 'healthy_SG24_2', 'SG7_4', 'SG2_1', 'atopic_eczema_FP_00008', 'healthy_SG15_1', 'healthy_SG56_2', 'healthy_SG27_2', 'dys_SG11_1', 'dys_SG132_5', 'atopic_eczema_FP_00374', 'healthy_SG47_3', 'dys_SG108_4', 'dys_SG101_2', 'SG39_1', 'dys_SG114_3', 'atopic_eczema_FP_00034', 'healthy_SG43_1', 'SG22_3', 'healthy_SG37_4', 'healthy_SG7_5', 'dys_SG133_1', 'SG14_4', 'atopic_eczema_FP_00032', 'SG45_5', 'atopic_eczema_FP_00196', 'healthy_SG37_2', 'healthy_SG24_3', 'dys_SG124_2', 'healthy_SG27_1', 'SG16_4', 'SG34_1', 'dys_SG10_1', 'SG27_2', 'healthy_SG78_5', 'dys_SG103_5', 'SG11_2', 'healthy_SG35_3', 'healthy_SG3_1', 'healthy_SG41_1', 'atopic_eczema_FP_00205', 'healthy_SG33_3', 'atopic_eczema_FP_00160', 'healthy_SG9_4', 'dys_SG106_2', 'healthy_SG73_5', 'dys_SG125_3', 'atopic_eczema_FP_00605', 'SG36_1', 'atopic_eczema_FP_00472', 'healthy_SG58_2', 'healthy_SG34_5', 'dys_SG110_2', 'atopic_eczema_FP_00028', 'healthy_SG1_1', 'SG23_2', 'healthy_SG19_2', 'healthy_SG68_2', 'healthy_SG56_1', 'atopic_eczema_FP_00453', 'dys_SG122_1', 'healthy_SG53_1', 'healthy_SG64_2', 'healthy_SG20_2', 'healthy_SG61_3', 'healthy_SG48_3', 'SG33_4', 'SG37_4', 'atopic_eczema_FP_00188', 'dys_SG12_5', 'healthy_SG3_5', 'healthy_SG49_4', 'SG33_3', 'dys_SG1002_1', 'SG42_4', 'SG44_2', 'healthy_SG42_4', 'atopic_eczema_FP_00039', 'dys_SG130_2', 'SG45_3', 'atopic_eczema_FP_00380', 'SG32_3', 'healthy_SG23_1', 'atopic_eczema_FP_00340', 'atopic_eczema_FP_00322', 'dys_SG118_3', 'SG35_3', 'healthy_SG41_3', 'SG22_1', 'healthy_SG77_5', 'healthy_SG34_2', 'healthy_SG5_3', 'healthy_SG76_2', 'healthy_SG55_3', 'healthy_SG45_1', 'dys_SG122_3', 'healthy_SG73_4', 'healthy_SG77_1', 'dys_SG107_3', 'atopic_eczema_FP_00487', 'SG36_5', 'SG29_3', 'dys_SG111_3', 'SG21_4', 'healthy_SG55_5', 'healthy_SG28_5', 'SG20_3', 'SG38_2', 'healthy_SG36_2', 'SG10_5', 'atopic_eczema_FP_00193', 'dys_SG121_3', 'healthy_SG7_1', 'healthy_SG41_5', 'SG43_2', 'atopic_eczema_FP_00318', 'dys_SG132_3', 'healthy_SG20_3', 'atopic_eczema_FP_00183', 'healthy_SG38_2', 'dys_SG107_2', 'dys_SG12_3', 'SG13_5', 'healthy_SG25_1', 'atopic_eczema_FP_00360', 'SG27_3', 'healthy_SG21_1', 'atopic_eczema_FP_00142', 'healthy_SG58_3', 'healthy_SG14_2', 'healthy_SG5_5', 'atopic_eczema_FP_00376', 'dys_SG109_3', 'atopic_eczema_FP_00037', 'healthy_SG27_3', 'dys_SG10_5', 'atopic_eczema_FP_00477', 'SG3_5', 'SG31_1', 'dys_SG1000_5', 'healthy_SG43_3', 'atopic_eczema_FP_00162', 'dys_SG101_1', 'healthy_SG40_4', 'healthy_SG4_1', 'SG24_3', 'healthy_SG8_2', 'atopic_eczema_FP_00481', 'atopic_eczema_FP_00156', 'dys_SG111_1', 'dys_SG115_3', 'SG31_3', 'healthy_SG72_1', 'healthy_SG6_3', 'healthy_SG74_2', 'atopic_eczema_FP_00044', 'atopic_eczema_FP_00203', 'dys_SG104_3', 'atopic_eczema_FP_00192', 'healthy_SG14_3', 'dys_SG100_2', 'atopic_eczema_FP_00363', 'dys_SG110_4', 'atopic_eczema_FP_00369', 'healthy_SG79_2', 'dys_SG106_5', 'atopic_eczema_FP_00486', 'healthy_SG16_5', 'healthy_SG46_5', 'healthy_SG54_3', 'atopic_eczema_FP_00493', 'healthy_SG10_2', 'SG1_2', 'SG44_4', 'dys_SG108_5', 'atopic_eczema_FP_00324', 'dys_SG109_5', 'healthy_SG3_3', 'healthy_SG4_5', 'SG18_2', 'healthy_SG40_1', 'SG41_1', 'dys_SG117_4', 'SG5_4', 'SG34_2', 'atopic_eczema_FP_00146']
        '''
        '''
        All imgs stats:

        Real verruca: 42
        Real dysh: 89
        Real psor: 62
        Real atopic: 286
        Real NIST database (without obvious disease): 956

        Sfinge dys: 200
        Sfinge verruca: 225
        Sfinge atopic: 500
        Sfinge healthy: 400
        '''

        atopic_real = []
        dysh_real = []
        psor_real = []
        verruca_real = []

        dysh_generated = []
        verruca_generated = []
        atopic_generated = []
        healthy_generated = []
        psor_generated = []

        nist_real = []

        for file in glob.glob(self.preprocessed_dataset_path):
            file_substr = file.split('/')[-1]
            extension = os.path.splitext(file)[1][1:]

            if extension == 'xml':
                continue

            test_image_name = file_substr.rsplit('.', 1)[0]

            if all(x in test_image_name for x in ["atopic", "_FP_"]):
                atopic_real.append(test_image_name)
            elif all(x in test_image_name for x in ["dys", "_FP_"]):
                dysh_real.append(test_image_name)
            elif all(x in test_image_name for x in ["psor", "_FP_"]):
                psor_real.append(test_image_name)
            elif all(x in test_image_name for x in ["verruca", "_FP_"]):
                verruca_real.append(test_image_name)
            elif "dys_SG" in test_image_name:
                dysh_generated.append(test_image_name)
            elif "atopic_eczema_SG" in test_image_name:
                atopic_generated.append(test_image_name)
            elif "healthy_SG" in test_image_name:
                healthy_generated.append(test_image_name)
            elif "nist_" in test_image_name:
                nist_real.append(test_image_name)
            elif "PsoriasisDamagedImg-SG" in test_image_name:
                psor_generated.append(test_image_name)
            elif "SG" in test_image_name:
                verruca_generated.append(test_image_name)



        # split real samples from each class 60 : 20 : 20 (train : val : test)
        print(len(atopic_real))
        atopic_real_train, atopic_real_val, atopic_real_test = self.split_train_val_test(atopic_real, 0.6, 0.8)
        print(len(atopic_real_train))
        print(len(atopic_real_val))
        print(len(atopic_real_test))
        dysh_real_train, dysh_real_val, dysh_real_test = self.split_train_val_test(dysh_real, 0.6, 0.8)
        psor_real_train, psor_real_val, psor_real_test = self.split_train_val_test(psor_real, 0.6, 0.8)
        verruca_real_train, verruca_real_val, verruca_real_test = self.split_train_val_test(verruca_real, 0.6, 0.8)

        # split generated fingerprints in ratio 80 : 10 : 10
        dysh_generated_train, dysh_generated_val, dysh_generated_test = self.split_train_val_test(dysh_generated, 0.8, 0.9)
        verruca_generated_train, verruca_generated_val, verruca_generated_test = self.split_train_val_test(verruca_generated, 0.8, 0.9)
        atopic_generated_train, atopic_generated_val, atopic_generated_test = self.split_train_val_test(atopic_generated, 0.8, 0.9)
        healthy_generated_train, healthy_generated_val, healthy_generated_test = self.split_train_val_test(healthy_generated, 0.8, 0.9)
        psor_generated_train, psor_generated_val, psor_generated_test = self.split_train_val_test(psor_generated, 0.8, 0.9)

        nist_real_train, nist_real_val, nist_real_test = self.split_train_val_test(nist_real, 0.8, 0.9)


        val_images_names = []
        test_images_names = []
        for file in glob.glob(self.preprocessed_dataset_path):
            file_substr = file.split('/')[-1]
            extension = os.path.splitext(file)[1][1:]

            if extension == 'xml':
                continue

            #generated_random_number = random.random()

            test_image_name = file_substr.rsplit('.', 1)[0]
            #if (generated_random_number >= 0.8):
            #if test_image_name in test_dataset_names:
            if test_image_name in atopic_real_val or test_image_name in dysh_real_val or test_image_name in psor_real_val or test_image_name in verruca_real_val or test_image_name in dysh_generated_val or test_image_name in verruca_generated_val or test_image_name in atopic_generated_val or test_image_name in healthy_generated_val or test_image_name in nist_real_val or test_image_name in psor_generated_val:
                shutil.move(file, os.path.join('dataset', 'val_preprocessed', file_substr))
                shutil.move(os.path.join('dataset', 'train_preprocessed', test_image_name + ".xml"), os.path.join('dataset', 'val_preprocessed', test_image_name + ".xml"))
                #test_image_name = file_substr.rsplit('.', 1)[0]
                val_images_names.append(test_image_name)
            elif test_image_name in atopic_real_test or test_image_name in dysh_real_test or test_image_name in psor_real_test or test_image_name in verruca_real_test or test_image_name in dysh_generated_test or test_image_name in verruca_generated_test or test_image_name in atopic_generated_test or test_image_name in healthy_generated_test or test_image_name in nist_real_test or test_image_name in psor_generated_test:
                shutil.move(file, os.path.join('dataset', 'test_preprocessed', file_substr))
                shutil.move(os.path.join('dataset', 'train_preprocessed', test_image_name + ".xml"), os.path.join('dataset', 'test_preprocessed', test_image_name + ".xml"))

                test_images_names.append(test_image_name)
        print("Val images:")
        print(val_images_names)
        print("Test images:")
        print(test_images_names)


        #for file in glob.glob(self.preprocessed_dataset_path):
        #file_substr = file.split('/')[-1]
        #test_xml_name = file_substr.rsplit('.', 1)[0]

        #if test_xml_name in test_images_names:
            #shutil.move(file, os.path.join('dataset', 'test_preprocessed', file_substr))


    def split_train_val_test(self, samples, split_val_value, split_test_value):
        random.shuffle(samples)

        train, val, test = np.split(samples, [int(len(samples)* split_val_value), int(len(samples)*split_test_value)])
        return train, val, test
