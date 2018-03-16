import model


def calculateMap(model):
    k_list = [10, 20, 30, 50, 100, 300]
    rgb_list = []
    hsv_list = []
    gabor_list = []

    for k in k_list:
        rgb_list.append(model.calculateMap(type='color_hist', k=k))
        hsv_list.append(model.calculateMap(type='hsv', k=k))
        gabor_list.append(model.calculateMap(type='gabor', k=k))

    print(k_list)
    print(rgb_list)
    print(hsv_list)
    print(gabor_list)



if __name__ == '__main__':
    import sys

    print("Please give your dataset path :\n")
    dataset_path = sys.stdin.readline()

    m = model.ImageRetrieval(dataset_path.strip())
    m.train()


    ans = 'Yes '
    while ans.strip() == 'Yes':
        print("Please give your query image location :\n")
        query_path = sys.stdin.readline()
        print("Please write the feature you want :\n")
        print("(Available choices : 'color_hist' / 'gabor' / 'hsv' )\n")
        type = sys.stdin.readline()
        m.query(query_path.strip(), type.strip())
        print("Do you want to query another image ?(Yes/No) :\n")
        ans = sys.stdin.readline()


