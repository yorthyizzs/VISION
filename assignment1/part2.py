import model
import pickle


"""

m = model.ImageRetrieval('Part2-dataset')
m.train()

with open('model.pickle', 'wb') as handle:
    pickle.dump(m, handle)



"""
with open('model.pickle', 'rb') as handle:
    m = pickle.load(handle)


print(m.calculateMap(type='color_hist', k=30))
