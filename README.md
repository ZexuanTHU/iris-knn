# iris-knn
Apply the KNN (K Nearest Neighbors) algorithms on the iris classification problem

## Datasets
You can use the built-in iris dataset(train_iris.data.csv for training while test_iris.data.csv for testing).
Or you can feel free to use any iris classification datasets which you can found online. They are all pretty similar.
If you want to use your own dataset, you may need some littie modify on the data loader part.

## How-to-use
```bash
# First, clone this repository
git clone https://github.com/ZexuanTHU/iris-knn

# Then enter the knn folder and run the main.py
cd knn
python main.py
```
If everything OK, you should get an output on your terminal like this
```bash
Distribution: [x1, y1, z1]
Predict: Iris-setosa TEST_LABEL: Iris-verginica
Distribution: [x2, y2, z2]
Predict: balabala TEST_LABEL: balabala
...
Distribution: [x3, y3, z3]
Predict: balabala TEST_LABEL: balabala

k:1 Accuracy: 0.667
k:2 Accuracy: 0.833
...
k:140: Accuracy: 0.333
```
After terminal ouput ending, the program will show a figure on your screen, which the x-axis is `k` and the y-axis is `Accuracy`.
It shows how the accuracy of KNN change with the increase of parameter k.

## About the KNN
Please look at the [wiki](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).

## License
MIT

If you have any problems ot advices, please feel free to open an issue.

