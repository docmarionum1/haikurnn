from keras.layers import Add, Dense, Input, LSTM
from keras.models import Model
from keras.utils import np_utils
import numpy as np

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    # From https://github.com/llSourcell/keras_explained/blob/master/gentext.py
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

class TrainingLine:
    def __init__(self, name, previous_line, lstm, n_tokens):
        self.char_input = Input(shape=(None, n_tokens), name='char_input_%s' % name)

        self.syllable_input = Input(shape=(1,), name='syllable_input_%s' % name)
        self.syllable_dense = Dense(lstm.units, activation='relu', name='syllable_dense_%s' % name)
        self.syllable_dense_output = self.syllable_dense(self.syllable_input)

        #self.lstm = LSTM(latent_dim, return_state=True, return_sequences=True, name='lstm_%s' % name)

        if previous_line:
            initial_state = [
                Add(name='add_h_%s' % name)([
                    previous_line.lstm_h,
                    self.syllable_dense_output
                ]),
                Add(name='add_c_%s' % name)([
                    previous_line.lstm_c,
                    self.syllable_dense_output
                ])
            ]
        else:
            initial_state = [self.syllable_dense_output, self.syllable_dense_output]

        self.lstm_out, self.lstm_h, self.lstm_c = lstm(self.char_input, initial_state=initial_state)

        self.output_dense = Dense(n_tokens, activation='softmax', name='output_%s' % name)
        self.output = self.output_dense(self.lstm_out)

def create_training_model(latent_dim, n_tokens):
    lstm = LSTM(latent_dim, return_state=True, return_sequences=True, name='lstm')
    lines = []
    inputs = []
    outputs = []

    for i in range(3):
        previous_line = lines[-1] if lines else None
        lines.append(TrainingLine('line_%s' % i, previous_line, lstm, n_tokens))
        inputs += [lines[-1].char_input, lines[-1].syllable_input]
        outputs.append(lines[-1].output)

    training_model = Model(inputs, outputs)
    training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    return training_model, lstm, lines, inputs, outputs

class GeneratorLine:
    def __init__(self, name, training_line, lstm, n_tokens):
        self.char_input = Input(shape=(None, n_tokens), name='char_input_%s' % name)

        self.syllable_input = Input(shape=(1,), name='syllable_input_%s' % name)
        self.syllable_dense = Dense(lstm.units, activation='relu', name='syllable_dense_%s' % name)
        self.syllable_dense_output = self.syllable_dense(self.syllable_input)

        self.h_input = Input(shape=(lstm.units,), name='h_input_%s' % name)
        self.c_input = Input(shape=(lstm.units,), name='c_input_%s' % name)
        initial_state = [self.h_input, self.c_input]

        self.lstm = lstm

        self.lstm_out, self.lstm_h, self.lstm_c = self.lstm(self.char_input, initial_state=initial_state)

        self.output_dense = Dense(n_tokens, activation='softmax', name='output_%s' % name)
        self.output = self.output_dense(self.lstm_out)

        self.syllable_dense.set_weights(training_line.syllable_dense.get_weights())
        #self.lstm.set_weights(lstm.get_weights())
        self.output_dense.set_weights(training_line.output_dense.get_weights())

class Generator:
    def __init__(self, lstm, lines, tf_session, tokenizer, n_tokens, max_line_length):
        self.tf_session = tf_session
        self.tokenizer = tokenizer
        self.n_tokens = n_tokens
        self.max_line_length = max_line_length

        self.lstm = LSTM(
            lstm.units, return_state=True, return_sequences=True,
            name='generator_lstm'
        )
        self.lines = [
            GeneratorLine(
                'generator_line_%s' % i,
                lines[i], self.lstm, self.n_tokens
            ) for i in range(3)
        ]
        self.lstm.set_weights(lstm.get_weights())

    def generate_haiku(self, syllables=[5, 7, 5], temperature=.1, first_char=None):
        output = []
        h = None
        c = None

        if first_char is None:
            first_char = chr(int(np.random.randint(ord('a'), ord('z')+1)))

        next_char = self.tokenizer.texts_to_sequences(first_char)[0][0]

        for i in range(3):
            line = self.lines[i]
            s = self.tf_session.run(
                line.syllable_dense_output,
                feed_dict={
                    line.syllable_input: [[syllables[i]]]
                }
            )

            if h is None:
                h = s
                c = s
            else:
                h = h + s
                c = c + s

            line_output = [next_char]

            end = False
            next_char = None
            for i in range(self.max_line_length):
                char, h, c = self.tf_session.run(
                    [line.output, line.lstm_h, line.lstm_c],
                    feed_dict={
                        line.char_input: [[
                            np_utils.to_categorical(
                                line_output[-1],
                                num_classes=self.n_tokens
                            )
                        ]],
                        line.h_input: h,
                        line.c_input: c
                    }
                )

                char = sample(char[0,0], temperature)
                if char == 1 and not end:
                    end = True
                if char != 1 and end:
                    next_char = char
                    char = 1

                line_output.append(char)

            cleaned_text = self.tokenizer.sequences_to_texts([
                line_output
            ])[0].strip()[1:].replace(
                '   ', '\n'
            ).replace(' ', '').replace('\n', ' ')

            print(cleaned_text)
            output.append(cleaned_text)

        return output
