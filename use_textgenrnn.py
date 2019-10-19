from textgenrnn import textgenrnn

input_file = "horror_plots.txt"
epochs = 50
weights_file = "textgenrnn_weights.hdf5"

def train_generator(input_file, epochs=0):
    textgen_model = textgenrnn()
    textgen_model.train_from_file(input_file, num_epochs=epochs)
    textgen_model.save("textgenrnn_weights.hdf5")

#train_generator(input_file, epochs)

def gen_text(weights_file):
    textgen_model = textgenrnn()
    textgen_model.load(weights_file)
    textgen_model.generate()

gen_text(weights_file=weights_file)