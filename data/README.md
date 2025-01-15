# Datasets

## Chest X-ray collection from Indiana University

* 7470 normalized images (14.19 GB)
* indiana_projections.csv(289.4 kB)
* indiana_reports.csv(1.68 MB) 

### References
https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university     


## MIMIC-CXR-JPG - chest radiographs with structured labels

The MIMIC Chest X-ray JPG (MIMIC-CXR-JPG) Database v2.0.0 is a large publicly available dataset of chest radiographs in JPG format with structured labels derived from free-text radiology reports. The MIMIC-CXR-JPG dataset is wholly derived from MIMIC-CXR, providing JPG format files derived from the DICOM images and structured labels derived from the free-text reports. The aim of MIMIC-CXR-JPG is to provide a convenient processed version of MIMIC-CXR, as well as to provide a standard reference for data splits and image labels. The dataset contains 377,110 JPG format images and structured labels derived from the 227,827 free-text radiology reports associated with these images. The dataset is de-identified to satisfy the US Health Insurance Portability and Accountability Act of 1996 (HIPAA) Safe Harbor requirements. Protected health information (PHI) has been removed. The dataset is intended to support a wide body of research in medicine including image understanding, natural language processing, and decision support.

### Requesting access to MIMIC data
* [ ] Physionet account that takes 8 days to be approved
* [ ] Complete required training for [`CITI Data or Specimens Only Research`](https://physionet.org/about/citi-course/) which includes 14 Modules with respective quizes. Completion might vary but could take 3 to 6 hours. **Note:** After completing the training, you will receive a notification within approximately 24 hours granting you access to the datasets.
    * SBE Refresher 1 - Defining Research with Human Subjects (ID 15029)	 
    * SBE Refresher 1 - Privacy and Confidentiality (ID 15035)	
    * SBE Refresher 1 - Assessing Risk (ID 15034)	
    * SBE Refresher 1 - Research with Children (ID 15036)	
    * SBE Refresher 1 - International Research (ID 15028)	
    * Instructions (ID 764)	
    * Biomed Refresher 2 - History and Ethical Principles (ID 511)	
    * Biomed Refresher 2 - Regulations and Process (ID 512)	
    * Biomed Refresher 2 - SBR Methodologies in Biomedical Research (ID 515)	
    * Biomed Refresher 2 - Genetics Research (ID 518)	
    * Biomed Refresher 2 - Records-Based Research (ID 516)	
    * Biomed Refresher 2 - Populations in Research Requiring Additional Considerations and/or Protections (ID 519)	
    * Biomed Refresher 2 - HIPAA and Human Subjects Research (ID 526)	
    * Biomed Refresher 2 - Conflicts of Interest in Research Involving Human Subjects (ID 17545)	
* [ ] Sign Data Use Agreement - MIMIC-CXR-JPG - chest radiographs with structured labels v2.1.0
https://physionet.org/sign-dua/mimic-cxr-jpg/2.1.0/


### Donwload MIMIC-CXR-JPG
The following command downloads only the first 10 records:
```
head -n 10 IMAGE_FILENAMES | wget -r -N -c -np --user YOUR_PHYSIONET_USERNAME --ask-password https://physionet.org/files/mimic-cxr-jpg/2.1.0/
```

* mimic-cxr-reports.tar.gz (136M) mimic-cxr-reports.tar.gz - for convenience, all free-text reports have been compressed in a single archive file
From https://physionet.org/content/mimic-cxr/2.1.0/

### Forum for questions
* https://github.com/MIT-LCP/mimic-code/discussions
* https://github.com/MIT-LCP/mimic-code/issues 


### References 

Johnson, Alistair, Lungren, Matthew, Peng, Yifan, Lu, Zhiyong, Mark, Roger, Berkowitz, Seth, and Steven Horng. "MIMIC-CXR-JPG - chest radiographs with structured labels" (version 2.1.0). PhysioNet (2024). https://doi.org/10.13026/jsn5-t979. 
https://scholar.google.com/scholar?cites=17057590153839370425&as_sdt=2005&sciodt=0,5&hl=en

Johnson, A., M. Lungren, Y. Peng, Z. Lu, R. Mark, S. Berkowitz, and S. Horng. MIMIC-CXR-JPG-chest radiographs with structured labels (version 2.0. 0). PhysioNet (2019).
https://scholar.google.com/scholar?cites=14964042261104640233&as_sdt=2005&sciodt=0,5&hl=en


