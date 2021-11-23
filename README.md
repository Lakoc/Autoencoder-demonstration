# Autoencoder demonstration

The program aims to demonstrate autoencoder learning and model's feature representation in low dimensional space. The
project is a part of [Soft computing course](https://www.fit.vut.cz/study/course/14810/.en) on the Faculty of
Information Technology Brno University of Technology.

Program is written in pure [numpy](https://numpy.org/) and [matplotlib](https://matplotlib.org/). Graphical user
interface was designed in [PySimpleGUI](https://pysimplegui.readthedocs.io/en/latest/). Full project specification and
documentation could be found [here](/documentation/Documentation.pdf).

The project is licensed under Apache License 2.0.
![Alt Text](documentation/demonstration.gif)

## Autoencoder

An autoencoder is a type of artificial neural network used to learn efficient codings of unlabeled data. An autoencoder
has two main parts: an encoder that maps the input into the code, and a decoder that maps the code to a reconstruction
of the input.
![](https://upload.wikimedia.org/wikipedia/commons/3/37/Autoencoder_schema.png)
*<div align="center"> The image was adopted from [Wikipedia](https://en.wikipedia.org/wiki/Autoencoder). </div>*

## Installation

To use this, simply download [anaconda distribution](https://www.anaconda.com/) and create new environment.

```bash
conda env create -f environment.yml
```

Activate enviroment and export pythonpath.

```bash
conda activate SFC
export PYTHONPATH=$PWD:$PYTHONPATH
```

And run gui application by next command.

```bash
python3 main.py
```

### Hope you  like the program. Cheers :)

![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)
