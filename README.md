# cs598project

## Dataset Construction:
1. Download this [file](https://physionet.org/content/mimic-cxr-jpg/2.1.0/mimic-cxr-2.1.0-test-set-labeled.csv) and rename to ```labels.csv```
2. Download this [file](https://physionet.org/content/mimic-cxr/2.1.0/cxr-study-list.csv.gz) and rename to ```paths.csv```
3. Run ```python3 preprocess/get_data_paths.py``` - creates a text file with all of the urls for each study and study id.
4. Run ```python3 preprocess/build_dataset.py``` - creates a csv file with labels associated with the radiology report

## Encoding
1. Run ```python3 preprocess/tokenize-dataset.py``` - creates a json with the encoded impressions

## Model
1. Run ```python3 train/load_train.py``` - creates the model and saves as "my_visualchexbert_model.pth"

## Citation
1. [Link](https://arxiv.org/pdf/2102.11467) to the research paper
2. [Link](https://github.com/stanfordmlgroup/VisualCheXbert) to the the GitHub repository of the research paper

```
@article{jain2021visualchexbert,
  title={VisualCheXbert: Addressing the Discrepancy Between Radiology Report Labels and Image Labels},
  author={Jain, Saahil and Smit, Akshay and Truong, Steven QH and Nguyen, Chanh DT and Huynh, Minh-Thanh and Jain, Mudit and Young, Victoria A and Ng, Andrew Y and Lungren, Matthew P and Rajpurkar, Pranav},
  journal={arXiv preprint arXiv:2102.11467},
  year={2021}
}
```