from scipy.spatial.distance import cosine
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize


def label_vecs_quelity(label_vecs):

    sim_mat = cosine_similarity(label_vecs)
    sim_mat_half = np.tril(sim_mat, k=-1)
    sim_mat_half = sim_mat_half.ravel()[np.flatnonzero(sim_mat_half)]
    print("mean cos sim:", np.mean(sim_mat_half))
    print("max cos sim:", np.max(sim_mat_half))
    print("min cos sim:", np.min(sim_mat_half))

    dis_mat = euclidean_distances(label_vecs)
    dis_mat_half = np.tril(dis_mat, k=-1)
    dis_mat_half = dis_mat_half.ravel()[np.flatnonzero(dis_mat_half)]
    print("mean euclid dis:", np.mean(dis_mat_half))
    print("max euclid dis:", np.max(dis_mat_half))
    print("min euclid dis:", np.min(dis_mat_half))


def label_random_mapping(class_num, mapping_dim, normal=False):

    assert class_num > 0

    label_vecs = np.random.normal(size=(class_num, mapping_dim))
    label_vecs = label_vecs.astype(np.float32)
    if normal:
        label_vecs = normalize(label_vecs, axis=1)

    label_vecs_quelity(label_vecs)

    return label_vecs


def label_greedy_mapping(class_num, mapping_dim, threshold=0.15, normal=False):

    assert class_num > 0

    print("start search label vecs...")

    first_vec = np.random.normal(size=(1, mapping_dim))
    if normal:
        first_vec = normalize(first_vec, axis=1)
    label_vecs = [first_vec]

    cnt = 1
    dummy_loop = 0
    dummy_loop_upper_bound = 10000

    while cnt < class_num:
        #
        if dummy_loop > dummy_loop_upper_bound:
            print("reach the dummy loop upper bound")
            break

        candidate_vec = np.random.normal(size=(1, mapping_dim))
        if normal:
            candidate_vec = normalize(candidate_vec, axis=1)
        for vec in label_vecs:
            simlarity = 1 - cosine(candidate_vec, vec)  # cos distance of scipy is actually 1-cosine, we have to reverse it
            if simlarity > threshold:
                dummy_loop += 1
                break
        else:
            print(cnt)
            label_vecs.append(candidate_vec)
            cnt += 1

    label_vecs = np.array(label_vecs).reshape(len(label_vecs), mapping_dim)
    label_vecs = label_vecs.astype(np.float32)

    label_vecs_quelity(label_vecs)

    return label_vecs


def label_greedy_mapping_online(label_dict, mapping_dim, threshold=0.15, normal=True):

    if len(label_dict) == 0:
        first_vec = np.random.normal(size=(1, mapping_dim))
        if normal:
            first_vec = normalize(first_vec, axis=1)
        return first_vec.astype(np.float32)

    else:

        while True:
            candidate_vec = np.random.normal(size=(1, mapping_dim))
            if normal:
                candidate_vec = normalize(candidate_vec, axis=1)
            for vec in label_dict.values():
                simlarity = 1 - cosine(candidate_vec, vec)  
                if simlarity > threshold:
                    break
            else:
                return candidate_vec.astype(np.float32)


def label_binary_mapping(class_num, mapping_dim):

    assert class_num > 0

    pre_labels=[]
    n = pow(2, mapping_dim)  # upper bound

    for i in range(class_num):
        pre_label = []
        for y in range(mapping_dim-1, -1, -1):
            num = str((i >> y) & 1)
            num = int(num)
            if num == 0:
                num = -1
            pre_label.append(num)

        pre_labels.append(pre_label)

    label_vecs = np.array(pre_labels)

    label_vecs_quelity(label_vecs)

    return label_vecs


def labels2Vec(y_train, label_dict, dim):

    n = y_train.shape[0]
    y_vec_train = []
    for label in y_train:
        y_vec_train.append(label_dict[label])
    y_vec_train = np.array(y_vec_train).reshape((n, dim))
    return y_vec_train


def evaluation(model, x_test, y_test, label_vecs):

    pred_vecs = model.predict(x_test)

    dis = euclidean_distances(pred_vecs, label_vecs)
    pred = np.argmin(dis, axis=1)

    acc = accuracy_score(y_test, pred)
    print("acc", acc)

# def monitor_evaluation(label_vecs):
#
#     def loss(y_test, pred_vecs):
#
#         dis = K.sqrt(K.sum(K.square(label_vecs - pred_vecs), axis=-1, keepdims=True))
#         pred = K.argmin(dis, axis=1)
#
#         acc = K.mean(K.equal(y_test, pred))
#         return acc
#
#     return loss


if __name__ == '__main__':

    class_num = 100
    vec_dim = 100

    # labee_vecs = label_random_mapping(class_num, vec_dim)
    # label_vecs = label_greedy_mapping(class_num, vec_dim, threshold=0.15, normal=True)
    # label_vecs = label_binary_mapping(class_num, vec_dim)

    label_dict = {}
    for i in range(class_num):
        label_vec = label_greedy_mapping_online(label_dict, vec_dim, threshold=0.15, normal=True)
        label_dict[i] = label_vec
    label_vecs = list(label_dict.values())
    label_vecs = np.array(label_vecs).reshape(class_num, vec_dim)
    label_vecs_quelity(label_vecs)

