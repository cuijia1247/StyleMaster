The experiment data on Painting 91 for optimal parameter finding.
1. test base_lr： 0.05, 0.03, 0.01, 0.005, 0.001

| base_lr |best_accuracy|last_accuracy|information|
|---------|-|-|-|
| 0.05    |0.670168|0.653361|lr is too large, the loss fluctuates|
| 0.03    |0.697479|0.634454|the loss fluctuates|
| 0.01    |0.710084|0.649160|no fluctuates, but loss reduce slowly, maybe test 0.012, 0.01, 0.008 with longer epoch will be better|
| 0.005   |0.691176|0.661765|loss reduce slowly|
| 0.001   |0.695378|0.682773|loss reduce slowly very much, but the lower bound is much better|
second-round:0.012, 0.01, 0.008, 0.0005

| base_lr |best_accuracy|last_accuracy|information|
|---------|-|-|-|
| 0.01    |0.701681|0.613445|loss fluctuates|
| 0.008   |0.707983|0.644958|loss reduce slowly|
| 0.0005  |0.686975|...|too slow|

2. test epochs: 201, 501, 701, 1001, 1501, 3001, 6001

| epochs |best_accuracy|last_accuracy| information         |
|---------|-|-|---------------------|
|201|0.689076|0.663866| unoptimal, 3.395692 |
|501|0.689076|0.680672| unoptimal, 3.362261 |
|701|0.686975|0.640756| upoptimal, 3.337603 |
|1001|0.693277|0.647059|3.282877|
|1501|0.699580|0.640756|2.946231|
|3001|0.707983|0.636555|2.129976|
|6001|0.716387|0.684874|not finished|

3. test image size: 32, 64, 128, 156

|image_size|best_accuracy|last_accuracy| information|
|-----------|-|-|-|
| 32        |0.686975|0.636555|2.286774, classifier is less trained|
| 64        |0.705882|0.638655|3.136801, the patch size is bigger, the loss reduction is slower|
| 128       |0.697479|0.592437|3.316861, the same with '64'|
| 156       |0.678571|0.630252|not finished|
Another test with more clearly test correctness and wrongness samples is required later.

4. test classifier iteration: 50, 100, 150, 200, 300


| classifier_iteration|best_accuracy|last_accuracy| information |
|---------------------|-|-|-|
| 50                  |0.668067|0.579832|classifier is not trained well|
| 100                 |0.668067|0.626050|2.950647|
| 150                 |0.693277|0.630252|2.502351|
| 200                 |0.680672|0.609244|1.042544|
| 300                 |0.697479|0.630252|2.511317|
longger iteration is still useful

5. test classifier lr: 0.0001, 0.0005, 0.00001, 0.00005, 0.000001

| classifier_lr        |best_accuracy|last_accuracy| information |
|----------------------|-|-|-|
| 0.0001               |0.676471|0.661765|1797/1862|
| 0.0005               |0.705882|0.638655|1739/1862|
| 0.00001              |0.653361|0.611345|1422/1862, not trained enough|
| 0.00005              |0.672269|0.634454|1718/1862|
| 0.000001             |0.495798|0.399160|805/1862, not trained enough|
| 0.00005+500iteration |0.670168|0.640756|1802/1862, overfitting|

6. test activation function: ReLU, SiLU, SELU, Softsign

| classifier_activation |best_accuracy|last_accuracy| information |
|-----------------------|-|-|-|
| ReLU                  |0.693277|0.632353|
| SiLU                  |0.701681|0.626050|
| SELU                  |0.697479|0.638655|
| Softsign              |0.680672|0.632353|

7. test classifier structures

| classifier_structures |best_accuracy|last_accuracy| information |
|-----------------------|-|-|-|
| 2048-4096-1024-512-13 |0.693277|0.638655|overfitting at 1000 epochs|
| 2048-1024-512-13      |0.705882|0.619748|overfitting at 1400 epochs|
| 2048-1024-512-256-13  |0.693277|0.607143|under-traind|
|2048-1024-512-13 |0.707983|0.617647|hard to increase the accuracy|



When the SSC loss is lower and lower, the cls accuracy is lower too.
The potential reason is the loss of SSC which is calculated on contents instead of style
Another option: when the loss is lower, the classifer training iterations should be increased.
The classifier training iterations could be adjustablely changed.
also, the classifier lr should be smaller than the current one.
Dropuot is also should be considered later.
