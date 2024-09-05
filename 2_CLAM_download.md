## 流程
### 1.下载tcga数据
下载tcga数据，应用tcga_download_gpt.py脚本 并行下载
切换工作目录

    cd  .\cpath\tcga_download\

    python .\tcga_download_gpt.py -m .\input\gdc_manifest.2024-09-05.txt -s .\output\ -w 6

### 2.patch切割
切换工作目录

    cd ..\CLAM-master\

    python create_patches_fp.py `
    --source ../tcga_download/output `
    --save_dir ./RESULTS_DIRECTORY  `
    --patch_size 256 `
    --preset tcga.csv `
    --seg `
    --patch `
    --stitch 

### 3.Feature Extraction
先通过下面脚本输出Step2.csv文件后，再进行操作

    import os 
    import pandas as pd 

    df = pd.read_csv('./RESULTS_DIRECTORY/process_list_autogen.csv') # 这个是上一步生成的文件
    ids1 = [i[:-4] for i in df.slide_id]
    ids2 = [i[:-3] for i in os.listdir('./RESULTS_DIRECTORY/patches/')]
    df['slide_id'] = ids1
    ids = df['slide_id'].isin(ids2)
    sum(ids)
    df.loc[ids].to_csv('RESULTS_DIRECTORY/Step_2.csv',index=False)

执行chatgpt改良的特征提取脚本

    $env:CUDA_VISIBLE_DEVICES="0"
    python extract_features_fp_gpt.py `
    --data_h5_dir ./RESULTS_DIRECTORY/ `
    --data_slide_dir ../tcga_download/output/ `
    --csv_path ./RESULTS_DIRECTORY/Step_2.csv `
    --feat_dir ./FEATURES_DIRECTORY `
    --batch_size 512 `
    --slide_ext .svs `
    --num_workers 2 

### 4.训练模型

    python main.py `
    --drop_out 0.25 `
    --early_stopping `
    --lr 2e-4 `
    --k 10 `
    --label_frac 0.75 `
    --exp_code task_2_tumor_subtyping_CLAM_50 `
    --weighted_sample `
    --bag_loss ce `
    --inst_loss svm `
    --task task_2_tumor_subtyping `
    --model_type clam_sb `
    --log_data `
    --data_root_dir /FEATURES_DIRECTORY `
    --subtyping

    $env:CUDA_VISIBLE_DEVICES="0"
    python main.py `
    --drop_out 0.25 `
    --early_stopping `
    --lr 2e-4 `
    --k 10 `
    --exp_code task_2_tumor_subtyping_CLAM_50 `
    --weighted_sample `
    --bag_loss ce `
    --inst_loss svm `
    --task task_2_tumor_subtyping `
    --model_type clam_sb `
    --log_data `
    --subtyping `
    --data_root_dir /tumor_subtyping_resnet_features`
    --embed_dim 1024


## 报错记录

### 1.PIL.Image.DecompressionBombError: Image size (5593464576 pixels) exceeds limit of 1866240000 pixels, could be decompression bomb DOS attack.
解决方案： 

    from PIL import Image

    # 调整最大像素数限制
    Image.MAX_IMAGE_PIXELS =   6593464576 # 或者设置为一个更大的数值

### 2.Support for input that cannot be coerced to a numeric array was deprecated in SciPy 1.9.0 and removed in SciPy 1.11.0. Please consider np.unique.
解决方案：dataset_generic.py文件

    # 定义计算众数的函数
    def custom_mode(arr):
        values, counts = np.unique(arr, return_counts=True)
        max_count_index = np.argmax(counts)
        return values[max_count_index]

调成class中的patient_data_prep函数

	def patient_data_prep(self, patient_voting='max'):
		patients = np.unique(np.array(self.slide_data['case_id']))  # Get unique patients
		patient_labels = []
		
		for p in patients:
			locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
			assert len(locations) > 0
			label = self.slide_data['label'][locations].values
			
			if patient_voting == 'max':
				label = label.max()  # Get patient label (MIL convention)
			elif patient_voting == 'maj':
				label = custom_mode(label)  # Use custom mode function without indexing
			else:
				raise NotImplementedError
			
			patient_labels.append(label)
		
		self.patient_data = {'case_id': patients, 'label': np.array(patient_labels)}
		



