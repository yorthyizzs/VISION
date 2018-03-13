import model
import pickle


"""



with open('model.pickle', 'rb') as handle:
    m = pickle.load(handle)

"""

m = model.ImageRetrieval('Part2-dataset')
m.train()


with open('model.pickle', 'wb') as handle:
    pickle.dump(m, handle)

print(m.calculateMap(type = 'color_hist', k=10))
