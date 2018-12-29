# Haiku Generation with Deep Learning _(Work in Progress)_

Read a full description of the project [here](https://towardsdatascience.com/generating-haiku-with-deep-learning-dbf5d18b4246).

This project is an attempt to use deep learning to generate haikus _that conform
to the 5-7-5 syllable pattern_. Much previous research into generating haikus
doesn't enforce syllable counts[[1]](https://www.cs.bgu.ac.il/~yoavg/publications/calc09haiku.pdf)[[2]](https://neuro.cs.ut.ee/wp-content/uploads/2018/02/poetry.pdf), largely because modern English haikus often
don't strictly conform to that pattern either. This makes finding training
data difficult. I get around this problem by providing the syllable count of
each line as an input to the network along with the text at training time. Then,
at generation time, I can choose how many syllables I want for each line. This
project is still early, but so far I've gotten some promising results.

Here an examples of 5-7-5 syllable output:

```
early morning sun
from the carried garden fate
stars at the sunset
```

And if I use the same network to get a 10-10-10 poem:
```
just as the street lamp spake the sun is bright
and the soul and the spring are blowing
with every beat of my heart i will love you
```

### Model Version 1

The first version of the model is implemented in [`notebooks/models/v1`](notebooks/models/v1).

![Model V1 Diagram](https://github.com/docmarionum1/haikurnn/raw/master/notebooks/models/v1/diagram.png)

The model is essentially a character-to-character text generation network with a twist. 
The number of syllables for each line is provided to the network, passed through a dense 
layer and then added to the LSTM's internal state. This means that by changing the three numbers 
provided, we can alter the behavior of the network. My hope is that this will still 
allow the network to learn "English" from the whole corpus even though most of the samples 
are not 5–7–5 haiku, while still allowing us to generate haiku of that length later. 

### Repo

The [`notebooks`](notebooks) directory contains the code organized into:
- [`data`](notebooks/data): Jupyter notebooks for working with and preparing the data.
- [`models`](notebooks/models): Jupyter notebooks and python files implementing the different models.

[`input`](input) contains the raw input data as well as [`haikus.csv`](input/poems/haikus.csv) 
which contains the whole corpus and [`sources.txt`](input/poems/sources.txt) describes the
sources used to build that corpus. [`Preprocess Haikus.ipynb`](notebooks/data/Preprocess%20Haikus.ipynb) 
constructs corpus.
