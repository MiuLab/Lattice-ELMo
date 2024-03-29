# LatticeLM: Learning Spoken Language Representations with Neural Lattice Language Modeling
[Paper](https://www.aclweb.org/anthology/2020.acl-main.347/)
| [Slides](https://www.csie.ntu.edu.tw/~yvchen/doc/ACL20_LatticeLM_slide.pdf)

Source code for our ACL 2020 paper *Learning Spoken Language Representations with Neural Lattice Language Modeling*

## Requirements
* Python >= 3.6
* Install the required Python packages with `pip3 install -r requirements.txt`

## Dataset
Unfortunately, we are not allowed to redistribute ATIS, SWDA and MRDA.
We provide a transcribed and processed dataset of the SNIPS NLU benchmark, where the audio files were generated with a TTS system, for training and evaluation.

#### Create a custom dataset
If you would like to create your custom dataset with lattices generated by your own Kaldi ASR models, please refer to the Preprocessing section [here](https://github.com/MiuLab/Lattice-SLU/blob/master/README.md)

## How to run
The training configs are located in [models](models).

### Steps
To train baseline models using ASR 1-best output with or without ELMo embeddings:

```
# For static word embeddings
python3 main.py ../models/snips_tts/asr/1

# For pre-trained ELMo embeddings
python3 main.py ../models/snips_tts/asr/2
```

To evaluate a classifier
```
python3 main.py --test --best_valid {classifier_model_dir}
```

To train a baseline LatticeLSTM classifier using ASR lattice
```
python3 main.py ../models/snips_tts/lattice/1
```

To train a LatticeLSTM classifier with pretrained ELMo embeddings
```
python3 main.py ../models/snips_tts/lattice/2
```

To fine-tune ELMo with lattices
```
python3 main_lm.py ../models/snips_tts/lattice/lm/1
```

To train a LatticeLSTM classifier with fine-tuned ELMo embeddings, you might want to modify the checkpoint number in the config.
```
python3 main.py ../models/snips_tts/lattice/3
```

## Reference
Please cite the following paper

    @inproceedings{huang-chen-2020-learning,
        title = "Learning Spoken Language Representations with Neural Lattice Language Modeling",
        author = "Huang, Chao-Wei  and
          Chen, Yun-Nung",
        booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
        month = jul,
        year = "2020",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/2020.acl-main.347",
        pages = "3764--3769",
        abstract = "Pre-trained language models have achieved huge improvement on many NLP tasks. However, these methods are usually designed for written text, so they do not consider the properties of spoken language. Therefore, this paper aims at generalizing the idea of language model pre-training to lattices generated by recognition systems. We propose a framework that trains neural lattice language models to provide contextualized representations for spoken language understanding tasks. The proposed two-stage pre-training approach reduces the demands of speech data and has better efficiency. Experiments on intent detection and dialogue act recognition datasets demonstrate that our proposed method consistently outperforms strong baselines when evaluated on spoken inputs. The code is available at https://github.com/MiuLab/Lattice-ELMo.",
    }
