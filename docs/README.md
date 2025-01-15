# Docs

## Install [uv](https://github.com/astral-sh/uv): "An extremely fast Python package manager".
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Create venv
```
cd ~/cdi-hub/tutorials/automatic-medical-image-reporting
uv venv --python 3.12 # Create a virtual environment at .venv.
```

## Acvivate venv
```
source .venv/bin/activate #To activate the virtual environment:
deactivate
```

## Install python package deps
```
uv pip install --editable ".[test, learning]" # Install the package in editable mode with test and learning dependencies
uv pip install ."[learning]" # Install learning dependencies
```

## Clean up any existing installation and reinstall:
```
uv pip uninstall ammir
uv pip install -e ".[test,learning]"
```


## References

<details>
  <summary>Click to see references</summary>

* [[RATCHET-2021]](https://arxiv.org/abs/2107.02104),
* [[BioBERT-2020]](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506)
* Train and evaluate Medical Transformer for Chest X-ray Diagnosis and Reporting [7-9]. 
* Implement evaluation tools for image captioning, including BLEU, ROUGE-L, CIDEr, METEOR, SPICE scores [10]. 
* Developoing a python-based interface using either [Streamlit](https://streamlit.io/) for a web-based solution or a simple command-line interface with [Click](https://click.palletsprojects.com/en/8.1.x/) or another suitable tool.

1. Guo, Li, Anas M. Tahir, Dong Zhang, Z. Jane Wang, and Rabab K. Ward. "Automatic Medical Report Generation: Methods and Applications." APSIPA Transactions on Signal and Information Processing 13, no. 1 (2024). [DOI](10.1561/116.20240044) [arxiv](https://arxiv.org/abs/2408.13988)

2. Hou, Benjamin, Georgios Kaissis, Ronald M. Summers, and Bernhard Kainz. "Ratchet: Medical transformer for chest x-ray diagnosis and reporting." In Medical Image Computing and Computer Assisted Intervention–MICCAI 2021: 24th International Conference, Strasbourg, France, September 27–October 1, 2021, Proceedings, Part VII 24, pp. 293-303. Springer International Publishing, 2021.  [arxiv](https://arxiv.org/pdf/2107.02104) [google-citations](https://scholar.google.com/scholar?cites=6324608147072853701&as_sdt=2005&sciodt=0,5&hl=en)

3. Ramesh, Vignav, Nathan A. Chi, and Pranav Rajpurkar. "Improving radiology report generation systems by removing hallucinated references to non-existent priors." In Machine Learning for Health, pp. 456-473. PMLR, 2022. [arxiv](https://arxiv.org/abs/2210.06340) [github-repo](https://github.com/rajpurkarlab/CXR-ReDonE) [google-scholar](https://scholar.google.com/scholar?cites=4808802074430489275&as_sdt=2005&sciodt=0,5&hl=en)

4. https://physionet.org/content/mimic-cxr-jpg/2.1.0/   

4.1. https://github.com/filipepcampos/mimic-cxr-jpg-loader

4.2. "Training a Convolutional Neural Network to Classify Chest X-rays" https://github.com/MIT-LCP/2019-hst-953/blob/master/tutorials/mimic-cxr/mimic-cxr-train.ipynb

4.3. "Predict plueral effusion in chest x-rays using the MIMIC-CXR-JPG dataset" https://github.com/dalton-macs/pleural-effusion-cnn/tree/main/data

5. https://physionet.org/content/mimic-cxr/2.1.0/     

5.1 https://mimic.mit.edu/docs/iv/modules/cxr/ 

6. https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university     

6.1 https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university/code

7.  https://github.com/omar-mohamed/X-Ray-Report-Generation/    

8. https://doi.org/10.1016/j.imu.2021.100557    

9. https://github.com/farrell236/RATCHET   

10. https://github.com/Aldenhovel/bleu-rouge-meteor-cider-spice-eval4imagecaption 

</details>



