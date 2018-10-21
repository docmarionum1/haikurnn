# Haiku Generation with Deep Learning _(Work in Progress)_

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
Hot afternoon haze
I tell the conversation
dandelion seeds
```

And if I use the same network to get a 5-5-5 poem:
```
Indian summer
packing of the wind
the smell of snow line
```

Clearly there's much work to be done in terms of _content_, but it's a start.

### Model

It is set up as a character level sequence-to-sequence problem.
The model I've been working on so far is 3 layers, one for each of the 3 lines. 
The syllable count for each line is embedded into a 512 dimensional space and 
used to initialize the state of each layer's LSTM, which is used for character
generation. For the second and third lines, the previous LSTM's state is added
with the syllable embedding to initialze the state. The output of each LSTM is
fed through a fully connected layer to generate one-hot-encodings of characters

The model looks something like this:

```
Syllables  (Line 1) -------> Dense
                               V (state)
Characters (Line 1) --------> LSTM -> Dense -> Output (Line 1)
                               V (state)
Syllables  (Line 2) -> Dense---+ (add Dense output with previous state)
                               V
Characters (Line 2) --------> LSTM -> Dense -> Output (Line 2)
                               V (state)
Syllables  (Line 3) -> Dense---+ (add Dense output with previous state)
                               V
Characters (Line 3) --------> LSTM -> Dense -> Output (Line 3)
```

### Repo

The `notebooks` directory contains ipython notebooks with the code.  They are
not well organized yet.

`input/dictionaries` contains phoneme dictionaries from The [CMU Pronouncing
Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict).

`input/poems` contains various poetry datasets.  
