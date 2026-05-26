# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:45:40 2017

@author: NishitP
"""

import pickle
import os
from sklearn.exceptions import NotFittedError

#doc_new = ['obama is running for president in 2016']

var = input("Please enter the news text you want to verify: ")
print("You entered: " + str(var))


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'sklearn.linear_model.logistic':
            module = 'sklearn.linear_model._logistic'
        return super().find_class(module, name)


def detecting_fake_news(var):
    # Try to load the pipeline-style model first
    model_path = 'final_model.sav'
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                load_model = CompatUnpickler(f).load()
            try:
                prediction = load_model.predict([var])
                print("The given statement is ", prediction[0])
                try:
                    prob = load_model.predict_proba([var])
                    print("The truth probability score is ", prob[0][1])
                except Exception:
                    print("Probability score not available for this model.")
                return
            except NotFittedError:
                # Fall through to fallback loading
                pass
        except Exception:
            pass

    # Fallback: load vectorizer + classifier separately
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
    except Exception:
        vectorizer = None

    for clf_name in ['model_new.pkl', 'model.pkl']:
        if not os.path.exists(clf_name):
            continue
        try:
            with open(clf_name, 'rb') as f:
                clf = CompatUnpickler(f).load()
            # transform input if vectorizer available
            X = [var]
            if vectorizer is not None:
                try:
                    X = vectorizer.transform([var])
                except Exception:
                    pass

            # Ensure shape expected
            try:
                pred = clf.predict(X)
                print("The given statement is ", pred[0])
                try:
                    prob = clf.predict_proba(X)
                    print("The truth probability score is ", prob[0][1])
                except Exception:
                    print("Probability score not available for this classifier.")
                return
            except Exception:
                continue
        except Exception:
            continue

    print("No compatible model found. Try running `setup_vectorizer.py` and training a model.")


if __name__ == '__main__':
    detecting_fake_news(var)