import pickle, os

p = 'final_model.sav'
print('exists', os.path.exists(p))

class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'sklearn.linear_model.logistic':
            module = 'sklearn.linear_model._logistic'
        return super().find_class(module, name)

with open(p, 'rb') as f:
    load_model = CompatUnpickler(f).load()

print('type', type(load_model))
print('has predict', hasattr(load_model, 'predict'))
try:
    steps = getattr(load_model, 'named_steps', None)
    print('named_steps:', steps)
    if steps:
        for name, step in steps.items():
            print('step:', name, type(step))
            if hasattr(step, 'get_params'):
                print(' params keys:', list(step.get_params().keys())[:5])
except Exception as e:
    print('inspect error', repr(e))
try:
    print('predict->', load_model.predict(['test']))
except Exception as e:
    print('predict error', repr(e))
try:
    print('predict_proba->', load_model.predict_proba(['test']))
except Exception as e:
    print('predict_proba error', repr(e))
