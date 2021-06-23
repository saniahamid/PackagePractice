"""MLP - machine-learning-production

Usage: 
    mlp-cli train <dataset-dir> <model-file> [--vocab-size=,vocab-size>]
    mlp-cli ask <model-file> <question>
    mlp-cli (-h | --help)
    
Arguments:
    <dataset-dir>   Directory with dataset.
    <model-file>    Serialized model file.
    <question>      Text to be classified.
    
Options:
    --vocab-size=<vocab-size>   Vocabulary size. [default: 10000]
    -h --help                   Show this screen.

"""

from docopt import docopt
import os
from sklearn.metrics import classification_report

from mlp import DumbModel, Dataset

def train_model(dataset_dir, model_file, vocab_size):
    print("Trainig model from directory {}".format(dataset_dir))
    print("Vocabulary size: {}".format(vocab_size))
    
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    dset = Dataset(train_dir, test_dir)
    X, y = dset.get_train_set()
    
    model = DumbModel(vocab_size=vocab_size)
    model.train(X, y)
    
    print("Storing model to {}".format(model_file))
    model.serialize(model_file)
    
    X_test, y_test = dset.get_test_set()
    y_pred = model.predict(X_test)
    
    print(classification_report(y_test, y_pred))
    
def ask_model(model_file,question):
    print("Asking model {} aboout '{}'".format(model_file, question))
    
    model = DumbModel.deserialize(model_file)
    
    y_pred = model.predict_proba([question])
    print(y_pred[0])
    
def main():
    arguments = docopt(__doc__)
    
    if arguments['train']:
        train_model(arguments['<dataset-dir>'],
                    arguments['<model-file>'],
                    arguments['--vocab-size'])
        
    elif arguments['ask']:
        ask_model(arguments['<model-file>'],
                  arguments['<question>'])
        
if __name__ == '__main__':
    main()
    