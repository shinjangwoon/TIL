from dataloader import load_data
from preprocess import scaling
from utils import load_config

from sklearn.ensemble import RandomForestClassifier


def main():
    # TODO python argparse package(library)
    # python main.py --config rf.json
    # python main.py --config svm.json
    # python main.py --config lr.json

    X, y = load_data()
    cfg = load_config(model_name='rf')

    preprocess_cfg = cfg['preprocess']
    model_cfg = cfg['params']

    new_X = X
    if preprocess_cfg['scaling']:
        new_X = scaling(X)

    model = RandomForestClassifier(**model_cfg)
    model.fit(new_X, y)
    print("Done")
    # TODO evaluate


if __name__ == "__main__":
    main()