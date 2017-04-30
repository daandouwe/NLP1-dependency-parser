def load_glove_model(glove_file):
    """
    loads pretrained word embeddings from file into dictionary
    :param glove_file: path to glove file
    :return: dictionary mapping words to 50-dimensional vectors
    """
    print("Loading GloVe Model")
    
    f = open(glove_file,'r', encoding='utf8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding

    # handle empty positions and root:
    model['_'] = 50*[-0.5]
    model['ROOT'] = 50*[0.5]
    return model

