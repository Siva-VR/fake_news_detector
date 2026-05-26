import pickle, os
files = ['final_model.sav','model.pkl','model_new.pkl']
class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'sklearn.linear_model.logistic':
            module = 'sklearn.linear_model._logistic'
        return super().find_class(module, name)

for p in files:
    print('\n--', p, 'exists', os.path.exists(p))
    if not os.path.exists(p):
        continue
    try:
        with open(p,'rb') as f:
            m = CompatUnpickler(f).load()
        print(' loaded:', type(m))
        if hasattr(m, 'predict'):
            try:
                print(' predict->', m.predict(['test']))
            except Exception as e:
                print(' predict error', repr(e))
    except Exception as e:
        print('load failed:', repr(e))
